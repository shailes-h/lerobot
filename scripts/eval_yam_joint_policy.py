#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-robot eval client for the RoboColosseum joint-space YAM HTTP policy servers (GR00T,
G0.5, MolmoAct2, pi0.5, LingBot-VLA v2 -- see EVAL_LOCAL_ROBOCOLOSSEUM.md).

This is a sibling of `eval_yam_http_policy.py`, NOT a replacement for it: that script talks
to the old goal-pose-prior server, which is 16-D ABSOLUTE end-effector state/action (IK'd onto
the arms). These new fine-tunes are different -- they were trained directly on **14-D absolute
joint angles** (state in, action out), in exactly the follower's own key order:

    [left_joint_0..5.pos, left_gripper.pos, right_joint_0..5.pos, right_gripper.pos]

There is no FK/IK anywhere in the control loop itself: the state sent to the server is read
straight off the follower's observation, and each returned action row is split into a 6-DoF
arm target plus a gripper value per side and sent to the robot as-is (through the same
hardware safety rails -- `max_joint_step_rad` / `max_gripper_step` clamps -- as the EEF
client). FK is only ever used for `--record_dataset=true` recordings, to fill in the
`*_eef.*`/`*_eef_delta.*` dataset columns -- see `_augment_action_with_eef` below.

Same interactive multi-episode session (b/s/f/n/r/z on stdin), same per-episode mp4s + session
log + optional `--record_dataset=true` LeRobotDataset recording as `eval_yam_http_policy.py` --
see that script's module docstring for the full behavior description; only the state/action
representation differs here.

`chunk_gate_enabled` and `home_snap_enabled` are reimplemented in plain joint space (max
per-joint angle deviation instead of an EEF pose distance) and default OFF, matching the EEF
client's own default-off home_snap; turn them on if you want the same near-home
idle-noise/snap behavior here.

`BiYamFollowerConfig.record_eef_pose` defaults to `True` (same as the EEF client), which makes
`BiYamFollower` populate `observation.state_eef_absolute` via FK on every tick. With
`--record_dataset=true`, this script's own `_augment_action_with_eef` fills in the matching
`action_eef_absolute`/`action_eef_delta` columns too (FK on the commanded joint target, same
`eef_pose_delta` formula `BiYamLeader.augment_action_with_observation` uses during teleop) --
so a recorded dataset lands in the exact same
`observation.state_eef_absolute` / `action_eef_absolute` / `action_eef_delta` /
`action_joint_angles` / `observation.state_joint_angles` schema as the teleop-collected
datasets (dustpan/cups/drawer/microwave), and `scripts/encode_pending_videos.py` alone is
enough afterward -- no separate schema conversion needed. Pass `--robot.record_eef_pose=false`
if you'd rather record the older plain `action`/`observation.state` (14-D joint only) schema;
the control loop itself never touches `*_eef.*` either way (it's dataset-only, not sent to the
server/robot).

Usage:

```shell
python scripts/eval_yam_joint_policy.py \
  --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260322275072", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260322271881", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262522074294", "width": 640, "height": 360, "fps": 30}
}' \
  --server_url=http://localhost:8000/act \
  --task="Clean the table." \
  --actions_per_chunk=8 \
  --max_episodes=50 \
  --record_dataset=true
```
"""

import json
import logging
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.robots.bi_yam_follower.bi_yam_follower import BiYamFollower
from lerobot.robots.bi_yam_follower.config_bi_yam_follower import BiYamFollowerConfig
from lerobot.robots.bi_yam_follower.eef_kinematics import (
    EEF_POSE_NAMES_WITH_GRIPPER,
    eef_pose_delta,
    eef_pose_from_joint_pos,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say

SIDES = ("left", "right")
CAM_KEYS = ("top", "left", "right")
VIDEO_TILE_WH = (320, 240)  # per-camera tile size in the composed mp4 frame

# Chunk gate thresholds (see EvalYamJointPolicyConfig.chunk_gate_enabled), now in plain joint
# space -- max absolute per-joint deviation, no FK/IK involved.
CHUNK_GATE_NEAR_HOME_ANGLE_DEG = 10.0
CHUNK_GATE_MIN_MOVE_ANGLE_DEG = 5.0


@dataclass
class DatasetRecordConfig:
    """LeRobot dataset output settings, intentionally aligned with lerobot-record.

    Not exposed on the CLI directly -- built internally by `_build_dataset_record_config`
    from `--record_dataset`/`--output_dir`/`--task`/the session timestamp, so a dataset
    (when requested) always lands next to that session's eval_logs, named after it."""

    repo_id: str
    root: str | Path | None = None
    fps: int = 30
    video: bool = True
    push_to_hub: bool = False
    private: bool = True
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1


@dataclass
class EvalYamJointPolicyConfig:
    robot: BiYamFollowerConfig
    # If true, also record eval rollouts as a native LeRobotDataset episode (the lightweight
    # tiled-mp4/session-log artifacts are always written regardless). The dataset is written
    # under this same run's `<output_dir>/<timestamp>/dataset` directory, named after the
    # session timestamp -- see `_build_dataset_record_config`. Saved RAW (no video encoding
    # during or at the end of this session) -- run `scripts/encode_pending_videos.py`
    # afterward.
    record_dataset: bool = False
    server_url: str = "http://localhost:8000/act"
    task: str = "Clean the table."
    # How many of the returned chunk's rows to actually execute before re-querying the server
    # (receding horizon). Lower = more reactive/robust to model error, higher = fewer round
    # trips.
    actions_per_chunk: int = 8
    control_hz: float = 30.0
    # Soft cap on one episode's rollout time -- once exceeded, the control loop stops
    # querying the server / moving the arms but still waits for you to label the episode
    # `s`/`f` (labeling is a human judgment, not something a timeout can decide). None (the
    # default) means no cap: the episode runs until you label it.
    duration_s: float | None = None
    play_sounds: bool = True
    # Diagnostic passthrough to the server; None lets it use its own default.
    num_steps: int | None = None
    # Slowly interpolate both arms' 6 joints from wherever they currently are to
    # `reset_joint_pos` before each episode's policy loop starts (and after each episode
    # ends), instead of snapping -- avoids a jerky move if the arms were left in an
    # arbitrary pose. Gripper is left untouched.
    reset_joint_pos: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    reset_duration_s: float = 4.0
    reset_hz: float = 20.0
    # If a predicted joint target for an arm is already within this threshold of that arm's
    # home joint pos, snap that arm's target to EXACTLY home instead of the model's (likely
    # just noisy) near-home output. The policy's own gripper command is left untouched either
    # way.
    #
    # DEFAULT OFF -- ported from the EEF client, where this was confirmed on real hardware to
    # cause a visible snap/jump rather than a smooth settle (see `eval_yam_http_policy.py`).
    # Same hard-cutoff caveat applies here: needs a continuous blend before it's safe to
    # enable.
    home_snap_enabled: bool = False
    home_snap_angle_threshold_deg: float = 3.0
    # HARDWARE SAFETY RAIL: hard-clamps every commanded joint target to at most this much
    # (radians) away from the PREVIOUSLY COMMANDED target for that joint, every control
    # step, regardless of where the target came from (a garbage/stale server response, a
    # discontinuity like the one that tripped a motor fault on 2026-09-09). This is
    # independent of and in addition to `--robot.left/right_arm_max_relative_target` (which
    # clips against the arm's live feedback position instead of the last command, and is
    # None/off unless you set it) -- defense in depth, not a replacement. 0.15 rad (~8.6 deg)
    # per step at the default 30 Hz control rate is a hard cap of ~258 deg/s; tune down if
    # that's still enough to jolt your payload. Applies in both the policy loop and
    # `_slow_reset`/homing.
    max_joint_step_rad: float = 0.15
    # Same idea for the gripper (0-1 range) -- much less safety-critical, but a max_joint_
    # step_rad-only rail sending a full-speed gripper snap alongside slowed joints is still a
    # smaller version of the same bug.
    max_gripper_step: float = 0.15
    # If an arm is near home AND the chunk's last row barely moves it (see CHUNK_GATE_*
    # constants, now plain joint-space deviations), hold that arm at rest instead of
    # executing the chunk -- filters idle noise without touching mid-task motion. DEFAULT OFF
    # here (unlike the EEF client) since it hasn't been validated against these new joint-
    # space checkpoints yet.
    chunk_gate_enabled: bool = False
    # Session bookkeeping.
    max_episodes: int = 50
    output_dir: str = "eval_logs"
    video_fps: float | None = None  # None -> defaults to control_hz


def _current_arm_joint_pos(obs: dict, side: str) -> np.ndarray:
    return np.array([obs[f"{side}_joint_{i}.pos"] for i in range(6)])


def _state14_from_obs(obs: dict) -> np.ndarray:
    """Build the server's 14-D ABSOLUTE joint state: left [joint_0..5, gripper] ++ right
    [same] -- exactly the follower's own observation keys, no FK/IK involved."""
    row = []
    for side in SIDES:
        row.extend(_current_arm_joint_pos(obs, side))
        row.append(obs[f"{side}_gripper.pos"])
    return np.asarray(row, dtype=np.float32)


def _slow_reset(
    robot: BiYamFollower,
    robot_action_processor,
    target_joint_pos: np.ndarray,
    duration_s: float,
    hz: float,
    max_joint_step_rad: float,
    max_gripper_step: float,
) -> None:
    """Linearly interpolate both arms' 6 joints from their current position to
    `target_joint_pos` over `duration_s`, in joint space. The gripper is interpolated open
    (1.0 -- see `bi_yam_leader.py`'s "not pressed = open (1)") over the same window, so every
    reset ends with both grippers open.

    `max_joint_step_rad`/`max_gripper_step` are the same hardware safety rail applied in the
    policy loop (see `EvalYamJointPolicyConfig.max_joint_step_rad`) -- normally a no-op here
    since the interpolation's own per-step delta is already small and known ahead of time,
    but it's cheap insurance against a misconfigured `duration_s`/`hz` producing a jump."""
    obs = robot.get_observation()
    start = {side: _current_arm_joint_pos(obs, side) for side in SIDES}
    gripper_start = {side: obs[f"{side}_gripper.pos"] for side in SIDES}
    prev_q = dict(start)
    prev_gripper = dict(gripper_start)

    n_steps = max(1, int(duration_s * hz))
    period_s = 1.0 / hz
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        alpha = step / n_steps
        action: dict[str, float] = {}
        for side in SIDES:
            q = (1 - alpha) * start[side] + alpha * target_joint_pos
            q = _clamp_step(prev_q[side], q, max_joint_step_rad)
            prev_q[side] = q
            for j, val in enumerate(q):
                action[f"{side}_joint_{j}.pos"] = float(val)

            gripper = (1 - alpha) * gripper_start[side] + alpha * 1.0
            gripper = float(np.clip(gripper, prev_gripper[side] - max_gripper_step, prev_gripper[side] + max_gripper_step))
            prev_gripper[side] = gripper
            action[f"{side}_gripper.pos"] = gripper

        robot_obs = robot.get_observation()
        processed_action = robot_action_processor((action, robot_obs))
        robot.send_action(processed_action)
        busy_wait(period_s - (time.perf_counter() - t0))


def _action_row_to_joint_gripper(row: np.ndarray, side_idx: int) -> tuple[np.ndarray, float]:
    """One arm's 6 joint targets + gripper out of a flat 14-D action row -- left is row[0:6]
    (joints) + row[6] (gripper), right is row[7:13] (joints) + row[13] (gripper)."""
    start = side_idx * 7
    return row[start : start + 6], float(row[start + 6])


def _snap_to_home_if_close(q6: np.ndarray, home_q6: np.ndarray, angle_threshold_rad: float) -> np.ndarray:
    """If `q6` is already within threshold of `home_q6` on every joint, return `home_q6`
    exactly instead of the model's raw, possibly-jittery near-home output."""
    if np.max(np.abs(q6 - home_q6)) < angle_threshold_rad:
        return home_q6
    return q6


def _clamp_step(prev: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    """Hard per-element rate limit: `target` clipped to within `max_delta` of `prev`. The
    hardware safety rail against any single-step jump -- see `max_joint_step_rad`'s docs on
    `EvalYamJointPolicyConfig` for why this exists independent of anything upstream."""
    return np.clip(target, prev - max_delta, prev + max_delta)


def _augment_action_with_eef(action: dict[str, float], robot_obs: dict) -> dict[str, float]:
    """If `--robot.record_eef_pose=true` (the default), `robot_obs` already carries
    `{side}_eef.*` (FK on the follower's own joint positions, done by `BiYamFollower.
    get_observation()`). Mirror what `BiYamLeader.get_action()` +
    `augment_action_with_observation()` do during teleop: FK the commanded joint target into
    `action[f"{side}_eef.{axis}"]` (-> `action_eef_absolute`), then
    `eef_pose_delta(target, current)` into `action[f"{side}_eef_delta.{axis}"]` (->
    `action_eef_delta`) -- so `--record_dataset=true` recordings get the same EEF columns a
    teleop-collected dataset has. No-op (returns `action` unchanged) when `record_eef_pose`
    is off, same guard as `augment_action_with_observation`."""
    for side in SIDES:
        if f"{side}_eef.x" not in robot_obs:
            continue
        q6 = np.array([action[f"{side}_joint_{i}.pos"] for i in range(6)])
        gripper = action[f"{side}_gripper.pos"]
        target_pose = eef_pose_from_joint_pos(np.append(q6, gripper))
        for axis in EEF_POSE_NAMES_WITH_GRIPPER:
            action[f"{side}_eef.{axis}"] = target_pose[axis]

        current_pose = {axis: robot_obs[f"{side}_eef.{axis}"] for axis in EEF_POSE_NAMES_WITH_GRIPPER}
        delta = eef_pose_delta(target_pose, current_pose)
        for axis in EEF_POSE_NAMES_WITH_GRIPPER:
            action[f"{side}_eef_delta.{axis}"] = delta[axis]
    return action


def _build_dataset_record_config(cfg: EvalYamJointPolicyConfig, timestamp: str) -> DatasetRecordConfig:
    """Only called when `--record_dataset=true`. Places the dataset inside this run's own
    `<output_dir>/<timestamp>/` session folder (alongside `videos/`, `log.jsonl`,
    `summary.txt`) and names it after that same timestamp -- so it always lives right next
    to the eval_logs it belongs to and never needs a separate `--dataset.repo_id`/`--dataset.root`
    from you. `repo_id` embeds both the timestamp and the task prompt you're evaling with
    (slugified), so the dataset is self-describing without opening it."""
    task_slug = "".join(c if c.isalnum() else "_" for c in cfg.task.strip().lower()).strip("_")
    task_slug = "_".join(filter(None, task_slug.split("_")))  # collapse repeated underscores
    repo_id = f"local/{timestamp}_{task_slug}" if task_slug else f"local/{timestamp}"
    root = Path(cfg.output_dir) / timestamp / "dataset"
    # DatasetRecordConfig.fps must be an int -- PyAV's add_stream() needs a Fraction-
    # convertible fps (int/Fraction), not a bare float; control_hz is a float (default 30.0)
    # to allow fractional control rates, so it's cast down here at the dataset boundary only.
    #
    # video_encoding_batch_size is set absurdly high on purpose: this session never wants to
    # pay for video encoding mid-run (it would stall the control loop for however long ffmpeg
    # takes, right in the middle of an eval episode). Episodes are saved RAW -- images kept
    # as loose per-frame PNGs on disk, never touched by `_batch_save_episode_video` -- and
    # `eval_policy()`'s `close_dataset()` deliberately does NOT encode at session end either.
    # Run `scripts/encode_pending_videos.py --repo-id ... --root ...` afterward to encode.
    return DatasetRecordConfig(
        repo_id=repo_id, root=root, fps=int(round(cfg.control_hz)), video_encoding_batch_size=10**6
    )


def _create_dataset(
    dataset_cfg: DatasetRecordConfig,
    robot: BiYamFollower,
    teleop_action_processor,
    robot_observation_processor,
) -> LeRobotDataset:
    """Build exactly the feature schema used by lerobot-record for this robot."""
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=dataset_cfg.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=dataset_cfg.video,
        ),
    )
    features = combine_feature_dicts(features, robot.extra_dataset_features or {})
    for old_key, new_key in (robot.dataset_feature_renames or {}).items():
        if old_key in features:
            features[new_key] = features.pop(old_key)

    return LeRobotDataset.create(
        dataset_cfg.repo_id,
        dataset_cfg.fps,
        root=dataset_cfg.root,
        robot_type=robot.name,
        features=features,
        use_videos=dataset_cfg.video,
        image_writer_processes=dataset_cfg.num_image_writer_processes,
        image_writer_threads=dataset_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=dataset_cfg.video_encoding_batch_size,
    )


def _request_with_retries(fn, retries: int = 5, backoff_s: float = 2.0):
    """Retry a `requests` call a few times with linear backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            logging.warning(f"request attempt {attempt}/{retries} failed: {e!r}; retrying in {backoff_s}s")
            time.sleep(backoff_s)
    raise RuntimeError(f"request failed after {retries} attempts") from last_exc


def _request_next_chunk(
    requests_module,
    server_url: str,
    payload: dict,
    robot: BiYamFollower,
    robot_action_processor,
    recorder: "EpisodeRecorder",
    dataset: LeRobotDataset | None,
    robot_observation_processor,
    task: str,
    init_q: dict[str, np.ndarray],
    prev_gripper: dict[str, float],
    period_s: float,
) -> tuple[dict, float]:
    """POST for the next action chunk on a background thread, and while waiting for the
    server's response keep sampling+recording the robot at the control period instead of
    silently losing those frames: the round-trip (inference + network) latency between chunks
    otherwise leaves a real-time gap the dataset never captures. The robot is re-commanded to
    hold its last applied joint targets (`init_q`/`prev_gripper`) for the duration, since that's
    what it's already doing and there's no new policy output yet to act on."""
    result: dict = {}

    def _do_request() -> None:
        result["resp"] = _request_with_retries(
            lambda: requests_module.post(
                server_url, json=payload, headers={"ngrok-skip-browser-warning": "1"}, timeout=60
            ).json()
        )

    t0 = time.perf_counter()
    thread = threading.Thread(target=_do_request, daemon=True)
    thread.start()

    hold_action: dict[str, float] = {}
    for side in SIDES:
        for j, q in enumerate(init_q[side]):
            hold_action[f"{side}_joint_{j}.pos"] = float(q)
        hold_action[f"{side}_gripper.pos"] = prev_gripper[side]

    while thread.is_alive():
        row_start = time.perf_counter()
        hold_obs = robot.get_observation()
        recorder.write(hold_obs)

        processed_action = robot_action_processor((hold_action, hold_obs))
        robot.send_action(processed_action)

        if dataset is not None:
            observation_frame = build_dataset_frame(
                dataset.features, robot_observation_processor(hold_obs), prefix=OBS_STR
            )
            action_frame = build_dataset_frame(
                dataset.features, _augment_action_with_eef(dict(hold_action), hold_obs), prefix=ACTION
            )
            dataset.add_frame({**observation_frame, **action_frame, "task": task})

        busy_wait(period_s - (time.perf_counter() - row_start))

    thread.join()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return result["resp"], dt_ms


# ---------------------------------------------------------------------------
# Keyboard input: a background thread blocks on stdin `input()` (so the control loop never
# does), pushing each line into a queue the main loop polls / blocks on as appropriate.
# ---------------------------------------------------------------------------


def _start_key_listener() -> queue.Queue:
    q: queue.Queue = queue.Queue()

    def _reader():
        while True:
            try:
                line = input()
            except EOFError:
                break
            q.put(line.strip().lower())

    threading.Thread(target=_reader, daemon=True).start()
    return q


def _wait_for_key(q: queue.Queue, valid: set[str]) -> str:
    """Block until one of `valid` is typed (+ Enter), ignoring/echoing anything else."""
    while True:
        line = q.get()
        if line in valid:
            return line
        print(f"[eval] unrecognized input {line!r}; expected one of {sorted(valid)}")


def _poll_key(q: queue.Queue, valid: set[str]) -> str | None:
    """Non-blocking: returns a valid key if one is queued, else None. Drains (and warns
    about) anything not in `valid` so stray input doesn't pile up unseen."""
    result = None
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line in valid:
            result = line
        else:
            print(f"[eval] unrecognized input {line!r}; expected one of {sorted(valid)}")
    return result


# ---------------------------------------------------------------------------
# Per-episode mp4 recording: all 3 camera views tiled into one frame, plus a second,
# separate max-quality recording of just the `top` camera at native resolution/fps. The
# outcome is only known after the episode ends, so recording writes to raw temp files first
# and they're simply moved to their final `..._success[_top].mp4` / `..._failure[_top].mp4`
# names once the outcome is known.
# ---------------------------------------------------------------------------


def _compose_frame(obs: dict) -> np.ndarray:
    tiles = []
    for key in CAM_KEYS:
        img = cv2.resize(np.asarray(obs[key]), VIDEO_TILE_WH)
        tiles.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # cameras hand back RGB; cv2 wants BGR
    return cv2.hconcat(tiles)


def _top_frame(obs: dict) -> np.ndarray:
    """The `top` camera frame at its native resolution, un-resized/un-tiled -- exactly what
    the robot saw, for the standalone max-quality per-episode top-camera video."""
    return cv2.cvtColor(np.asarray(obs["top"]), cv2.COLOR_RGB2BGR)  # cameras hand back RGB; cv2 wants BGR


class EpisodeRecorder:
    """Writes two mp4s per episode: the tiled 3-camera overview (small, for a quick look),
    and a second recording of just the `top` camera at its native resolution/fps and with a
    high-quality codec setting -- as close to what the robot's camera actually saw as
    `cv2.VideoWriter` allows (mp4v's default lossy quantization noticeably softens fine
    detail at the tiles' downscaled 320x240)."""

    def __init__(self, raw_path: Path, raw_top_path: Path, fps: float, obs: dict):
        tiled_shape = _compose_frame(obs).shape
        top_shape = _top_frame(obs).shape
        h, w = tiled_shape[:2]
        top_h, top_w = top_shape[:2]

        self.raw_path = raw_path
        self.raw_top_path = raw_top_path
        self.fps = fps
        self._writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self._top_writer = cv2.VideoWriter(
            str(raw_top_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (top_w, top_h)
        )
        # Bump the top-camera writer's JPEG quality knob to its max (100) -- mp4v internally
        # quantizes per-frame like JPEG, and OpenCV honors this property for it. Silently a
        # no-op on builds/codecs that don't support the property.
        self._top_writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
        self.n_frames = 0

    def write(self, obs: dict) -> None:
        self._writer.write(_compose_frame(obs))
        self._top_writer.write(_top_frame(obs))
        self.n_frames += 1

    def close(self) -> None:
        self._writer.release()
        self._top_writer.release()

    def finalize(self, final_path: Path, final_top_path: Path, outcome: str) -> None:
        """Move the raw recordings to their final paths. `outcome` is unused for the videos
        themselves (the success/failure is already encoded in the filenames) but kept in the
        signature for callers/logging."""
        self.close()
        self.raw_path.replace(final_path)
        self.raw_top_path.replace(final_top_path)


# ---------------------------------------------------------------------------
# Session logging: one timestamped directory per run, so consecutive sessions never clobber
# each other's videos/logs.
# ---------------------------------------------------------------------------


class SessionLog:
    def __init__(self, output_dir: str, timestamp: str | None = None):
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_dir) / self.timestamp
        self.videos_dir = self.session_dir / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.session_dir / "log.jsonl"
        self.summary_path = self.session_dir / "summary.txt"
        self.records: list[dict] = []
        logging.info(f"Session log directory: {self.session_dir}")

    def record_episode(self, episode: int, outcome: str, video_path: Path, n_frames: int, fps: float) -> None:
        entry = {
            "episode": episode,
            "outcome": outcome,
            "video": str(video_path.relative_to(self.session_dir)),
            "n_frames": n_frames,
            "duration_s": n_frames / fps if fps else None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.records.append(entry)
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self._write_summary(final=False)

    def _write_summary(self, final: bool) -> None:
        n = len(self.records)
        n_success = sum(1 for r in self.records if r["outcome"] == "success")
        rate = (n_success / n * 100.0) if n else 0.0
        lines = [
            f"{'FINAL' if final else 'Running'} summary -- {n} episode(s), {n_success} success, "
            f"success rate {rate:.1f}%",
            "",
        ]
        for r in self.records:
            lines.append(f"  ep {r['episode']:03d}: {r['outcome']:7s} ({r['video']})")
        self.summary_path.write_text("\n".join(lines) + "\n")

    def finalize(self) -> str:
        self._write_summary(final=True)
        n = len(self.records)
        n_success = sum(1 for r in self.records if r["outcome"] == "success")
        rate = (n_success / n * 100.0) if n else 0.0
        return f"{n} episode(s), {n_success} success, success rate {rate:.1f}%"


_CONTROLS = {
    "start": "b: begin episode 1 (homes first)   |   r: home/reset now   |   z: quit (0 episodes run)",
    "running": "s: mark SUCCESS, end episode   |   f: mark FAILURE, end episode",
    "waiting": "n: next episode (homes first)   |   r: home/reset now   |   z: finish session now",
}


def _print_controls(state: str) -> None:
    print(f"[eval] controls -- {_CONTROLS[state]}")


def _home(cfg: EvalYamJointPolicyConfig, robot: BiYamFollower, robot_action_processor) -> None:
    log_say("Resetting to home position", cfg.play_sounds, blocking=True)
    _slow_reset(
        robot,
        robot_action_processor,
        np.asarray(cfg.reset_joint_pos, dtype=np.float32),
        cfg.reset_duration_s,
        cfg.reset_hz,
        cfg.max_joint_step_rad,
        cfg.max_gripper_step,
    )


def _cleanup_and_exit(
    cfg: EvalYamJointPolicyConfig,
    robot: BiYamFollower,
    robot_action_processor,
    session: "SessionLog",
    reason: str,
) -> None:
    """Best-effort: home the arms and finalize the session log/summary, even if something
    (Ctrl+C, an unhandled error mid-episode) interrupted the normal flow. Never raises --
    this is the last thing that runs."""
    print(f"\n[eval] {reason} -- homing arms and closing out the session cleanly.")
    try:
        _home(cfg, robot, robot_action_processor)
    except Exception as e:  # noqa: BLE001
        logging.warning(f"home-on-cleanup failed: {e!r}")
    summary = session.finalize()
    print(f"[eval] {summary}")
    print(f"[eval] session log: {session.session_dir}")
    try:
        robot.disconnect()
    except Exception as e:  # noqa: BLE001
        logging.warning(f"disconnect-on-cleanup failed: {e!r}")


def _run_episode(
    cfg: EvalYamJointPolicyConfig,
    robot: BiYamFollower,
    robot_action_processor,
    requests,
    key_q: queue.Queue,
    episode: int,
    session: "SessionLog",
    dataset: LeRobotDataset | None,
    robot_observation_processor,
) -> tuple[str, "EpisodeRecorder"]:
    """Home, then run the policy loop until `s`/`f` is typed. Returns the outcome ('success'
    or 'failure')."""
    _home(cfg, robot, robot_action_processor)

    video_fps = cfg.video_fps or cfg.control_hz
    raw_path = session.session_dir / f"_raw_tmp_episode_{episode:03d}.mp4"  # renamed by recorder.finalize()
    raw_top_path = session.session_dir / f"_raw_tmp_episode_{episode:03d}_top.mp4"
    obs = robot.get_observation()
    recorder = EpisodeRecorder(raw_path, raw_top_path, video_fps, obs)

    log_say(f"Starting episode {episode}: {cfg.task}", cfg.play_sounds, blocking=True)
    print(f"[eval] episode {episode} RUNNING")
    _print_controls("running")

    init_q = {side: _current_arm_joint_pos(obs, side) for side in SIDES}
    prev_gripper = {side: obs[f"{side}_gripper.pos"] for side in SIDES}
    period_s = 1.0 / cfg.control_hz
    t_start = time.perf_counter()
    step = 0
    outcome: str | None = None
    time_capped = False

    # Home joint pos, used by home-snap and the chunk gate (both plain joint-space checks).
    home_q6 = np.asarray(cfg.reset_joint_pos, dtype=np.float32)
    angle_threshold_rad = np.deg2rad(cfg.home_snap_angle_threshold_deg)

    while outcome is None:
        elapsed = time.perf_counter() - t_start
        if cfg.duration_s is not None and elapsed >= cfg.duration_s:
            if not time_capped:
                print(
                    f"[eval] episode {episode}: duration_s={cfg.duration_s} reached, holding position -- "
                    "type 's' or 'f' to label it."
                )
                time_capped = True
            key = _wait_for_key(key_q, {"s", "f"})
            outcome = "success" if key == "s" else "failure"
            break

        obs = robot.get_observation()
        recorder.write(obs)
        state14 = _state14_from_obs(obs)

        payload = {
            "top": np.asarray(obs["top"]),
            "left": np.asarray(obs["left"]),
            "right": np.asarray(obs["right"]),
            "instruction": cfg.task,
            "state": state14,
        }
        if cfg.num_steps is not None:
            payload["num_steps"] = cfg.num_steps

        resp, dt_ms = _request_next_chunk(
            requests,
            cfg.server_url,
            payload,
            robot,
            robot_action_processor,
            recorder,
            dataset,
            robot_observation_processor,
            cfg.task,
            init_q,
            prev_gripper,
            period_s,
        )
        if "error" in resp:
            raise RuntimeError(f"server returned an error: {resp['error']}")

        actions = np.asarray(resp["actions"], dtype=np.float32)  # (chunk_size, 14), ABSOLUTE joints
        logging.info(
            f"episode={episode} step={step} server_dt_ms={dt_ms:.1f} "
            f"resp_dt_ms={resp.get('dt_ms', float('nan')):.1f}"
        )

        # Hold an arm at rest this chunk if it's near home and barely moving (see
        # chunk_gate_enabled).
        chunk_move_enough = {side: True for side in SIDES}
        if cfg.chunk_gate_enabled:
            for side_idx, side in enumerate(SIDES):
                current_q = _current_arm_joint_pos(obs, side)
                near_home = np.max(np.abs(current_q - home_q6)) < np.deg2rad(CHUNK_GATE_NEAR_HOME_ANGLE_DEG)
                if not near_home:
                    continue
                last_q6, _ = _action_row_to_joint_gripper(actions[-1], side_idx)
                chunk_move_enough[side] = np.max(np.abs(last_q6 - current_q)) >= np.deg2rad(
                    CHUNK_GATE_MIN_MOVE_ANGLE_DEG
                )

        n_exec = min(cfg.actions_per_chunk, len(actions))
        for i in range(n_exec):
            row_start = time.perf_counter()

            key = _poll_key(key_q, {"s", "f"})
            if key is not None:
                outcome = "success" if key == "s" else "failure"
                break

            action: dict[str, float] = {}
            for side_idx, side in enumerate(SIDES):
                if not chunk_move_enough[side]:
                    # Hold: re-send the last commanded target instead of this row's prediction.
                    for j, q in enumerate(init_q[side]):
                        action[f"{side}_joint_{j}.pos"] = float(q)
                    action[f"{side}_gripper.pos"] = prev_gripper[side]
                    continue

                q6, gripper_val = _action_row_to_joint_gripper(actions[i], side_idx)
                if cfg.home_snap_enabled:
                    q6 = _snap_to_home_if_close(q6, home_q6, angle_threshold_rad)

                # HARDWARE SAFETY RAIL -- see `max_joint_step_rad`'s docstring. Clamp against
                # the previously COMMANDED joint target (init_q[side]), regardless of source
                # (a discontinuity, a stale/garbage response) -- never send a jump bigger than
                # this in one control step.
                q6_step = np.max(np.abs(q6 - init_q[side]))
                q6 = _clamp_step(init_q[side], q6, cfg.max_joint_step_rad)
                if q6_step > cfg.max_joint_step_rad:
                    logging.warning(
                        f"episode={episode} step={step} row={i} {side}: joint step {q6_step:.3f} rad "
                        f"clamped to {cfg.max_joint_step_rad} rad/step"
                    )
                init_q[side] = q6

                gripper_val = float(
                    np.clip(gripper_val, prev_gripper[side] - cfg.max_gripper_step, prev_gripper[side] + cfg.max_gripper_step)
                )
                prev_gripper[side] = gripper_val

                for j, q in enumerate(q6):
                    action[f"{side}_joint_{j}.pos"] = float(q)
                action[f"{side}_gripper.pos"] = gripper_val

            robot_obs = robot.get_observation()
            recorder.write(robot_obs)

            if dataset is not None:
                observation_frame = build_dataset_frame(
                    dataset.features, robot_observation_processor(robot_obs), prefix=OBS_STR
                )
                action_frame = build_dataset_frame(
                    dataset.features, _augment_action_with_eef(dict(action), robot_obs), prefix=ACTION
                )

            processed_action = robot_action_processor((action, robot_obs))
            robot.send_action(processed_action)

            if dataset is not None:
                dataset.add_frame({**observation_frame, **action_frame, "task": cfg.task})

            busy_wait(period_s - (time.perf_counter() - row_start))

        step += 1

    print(f"[eval] episode {episode} marked {outcome.upper()} -- homing and holding.")
    log_say(f"Episode {outcome}", cfg.play_sounds, blocking=True)
    _home(cfg, robot, robot_action_processor)

    return outcome, recorder


@parser.wrap()
def eval_policy(cfg: EvalYamJointPolicyConfig):
    init_logging()
    logging.info(cfg)

    # Imported here (not at module top) so a plain `--help` / config-only invocation doesn't
    # require json_numpy/requests to be importable.
    import json_numpy
    import requests

    # NOTE: do NOT call json_numpy.patch() until after the first get_observation() call
    # below. json_numpy monkeypatches the stdlib `json` module process-wide, and even though
    # this joint-space client itself has no FK/IK, BiYamFollower.get_observation() still does
    # its own internal FK for EEF-pose logging fields (`record_eef_pose`), which lazily
    # imports scipy -> numpy.testing, and numpy.testing does its own unrelated json.loads() at
    # import time that crashes under the patched decoder hook. Same issue (and same fix) as
    # `eval_yam_http_policy.py`'s module docstring describes -- connect + one get_observation()
    # call first (forces the scipy import), patch only after.
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    robot = BiYamFollower(cfg.robot)
    robot.connect()
    robot.get_observation()  # forces the lazy scipy import before patching json

    json_numpy.patch()

    # Sanity-check the server is up and its contract matches what this client assumes,
    # before moving any hardware.
    health = _request_with_retries(
        lambda: requests.get(cfg.server_url, headers={"ngrok-skip-browser-warning": "1"}, timeout=15).json()
    )
    logging.info(f"Server health: {health}")
    if health.get("state_dim") != 14 or health.get("action_dim") != 14:
        raise RuntimeError(f"unexpected server state/action dims: {health}")
    if list(health.get("camera_keys", [])) != ["top", "left", "right"]:
        raise RuntimeError(f"unexpected server camera_keys: {health.get('camera_keys')}")

    # Shared by both the session log dir and (if enabled) the dataset dir/name, so they're
    # always found together under `<output_dir>/<timestamp>/`.
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if cfg.record_dataset:
        dataset_cfg = _build_dataset_record_config(cfg, session_timestamp)
        logging.info(
            f"Recording dataset RAW (no video encoding this session) to {dataset_cfg.root} "
            f"(repo_id={dataset_cfg.repo_id})"
        )
        dataset = _create_dataset(dataset_cfg, robot, teleop_action_processor, robot_observation_processor)
    else:
        logging.info("--record_dataset not set -- eval rollouts will NOT be recorded as a dataset.")
        dataset_cfg = None
        dataset = None
    dataset_closed = False

    def close_dataset(interrupted: bool = False) -> None:
        """Close the raw dataset without encoding any video. `video_encoding_batch_size` is
        set absurdly high (see `_build_dataset_record_config`) precisely so episodes are
        never auto-encoded mid-run; deliberately skip `VideoEncodingManager`'s exit-time
        batch-encode too, so no video encoding happens in this process at all -- only
        `scripts/encode_pending_videos.py`, run afterward, ever calls
        `_batch_save_episode_video`. This mirrors that script's cleanup of a still-open
        (never `save_episode()`-d) episode's raw images on interrupt, then just closes the
        parquet writers."""
        nonlocal dataset_closed
        if dataset_closed or dataset is None:
            dataset_closed = True
            return
        if interrupted:
            interrupted_episode_index = dataset.num_episodes
            for key in dataset.meta.video_keys:
                img_dir = dataset._get_image_file_path(
                    episode_index=interrupted_episode_index, image_key=key, frame_index=0
                ).parent
                if img_dir.exists():
                    logging.debug(f"Cleaning up interrupted episode images for camera {key}")
                    shutil.rmtree(img_dir)
        dataset.finalize()
        logging.info(
            f"Dataset saved RAW at {dataset_cfg.root} -- {dataset.num_episodes} episode(s) still need "
            f"encoding. Run:\n  python scripts/encode_pending_videos.py "
            f"--repo-id {dataset_cfg.repo_id} --root {dataset_cfg.root}"
        )
        if dataset_cfg.push_to_hub:
            logging.warning(
                "--dataset.push_to_hub was set but the dataset is raw/unencoded -- skipping push. "
                "Run encode_pending_videos.py first, then push manually."
            )
        dataset_closed = True

    session = SessionLog(cfg.output_dir, timestamp=session_timestamp)
    key_q = _start_key_listener()

    try:
        print(f"\n[eval] Session ready ({cfg.max_episodes} episode(s) max).")
        _print_controls("start")
        while True:
            key = _wait_for_key(key_q, {"b", "r", "z"})
            if key == "r":
                _home(cfg, robot, robot_action_processor)
                _print_controls("start")
                continue
            break  # 'b' or 'z'

        if key == "z":
            print("[eval] finishing with 0 episodes run.")
            close_dataset()
            _cleanup_and_exit(cfg, robot, robot_action_processor, session, "quit before any episode")
            return

        episode = 1
        while True:
            outcome, recorder = _run_episode(
                cfg,
                robot,
                robot_action_processor,
                requests,
                key_q,
                episode,
                session,
                dataset,
                robot_observation_processor,
            )

            final_path = session.videos_dir / f"episode_{episode:03d}_{outcome}.mp4"
            final_top_path = session.videos_dir / f"episode_{episode:03d}_{outcome}_top.mp4"
            recorder.finalize(final_path, final_top_path, outcome)
            if dataset is not None:
                dataset.save_episode()
            session.record_episode(episode, outcome, final_path, recorder.n_frames, recorder.fps)
            print(f"[eval] saved {final_path}")
            print(f"[eval] saved {final_top_path}")

            if episode >= cfg.max_episodes:
                print(f"[eval] max_episodes={cfg.max_episodes} reached -- finishing.")
                break

            print(f"[eval] episode {episode} logged. Reset the scene, then:")
            _print_controls("waiting")
            while True:
                key = _wait_for_key(key_q, {"n", "r", "z"})
                if key == "r":
                    _home(cfg, robot, robot_action_processor)
                    _print_controls("waiting")
                    continue
                break  # 'n' or 'z'

            if key == "z":
                print(f"[eval] finishing early at episode {episode} (of {cfg.max_episodes} max).")
                break
            episode += 1

        summary = session.finalize()
        print(f"[eval] {summary}")
        print(f"[eval] session log: {session.session_dir}")
        close_dataset()
        robot.disconnect()

    except KeyboardInterrupt:
        close_dataset(interrupted=True)
        _cleanup_and_exit(cfg, robot, robot_action_processor, session, "Ctrl+C received")

    except Exception:
        close_dataset(interrupted=True)
        _cleanup_and_exit(cfg, robot, robot_action_processor, session, "evaluation aborted by an error")
        raise


def main():
    eval_policy()


if __name__ == "__main__":
    main()
