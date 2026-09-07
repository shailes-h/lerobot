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

"""Real-robot eval client for the custom goal-pose-prior YAM HTTP policy server
(see `host_policy_server_reference.py` for the server-side contract this implements).

This is NOT `lerobot.async_inference.robot_client` -- that client speaks a different
protocol (gRPC, joint-space, delta actions, for the released `allenai/MolmoAct2-BimanualYAM`
HF checkpoint). This server is a different artifact: a plain HTTP `POST /act`, json_numpy
encoded, that takes 16-D ABSOLUTE end-effector state and returns 16-D ABSOLUTE end-effector
action chunks (30 x 16, w-first quats, left arm 0:8 / right arm 8:16 -- see
`lerobot.robots.bi_yam_follower.eef_kinematics.EEF_POSE_NAMES_WITH_GRIPPER`, which is exactly
this layout per arm). So this script builds that state, POSTs it, and drives the arms via
`ik_from_eef_pose` on each returned row -- reusing the same FK/IK helpers and IK-warm-start
pattern as `lerobot.scripts.lerobot_replay_bi_yam`'s `eef_absolute` method.

This is an interactive multi-episode eval session, driven from stdin (type a letter, press
Enter):

    b  -- begin: home the arms (if not already there), then start policy inference for
          episode 1. Only valid before the first episode.
    s  -- mark the CURRENTLY RUNNING episode a success, stop inference/recording, home the
          arms, and hold there until you've physically reset the scene.
    f  -- while an episode is RUNNING: mark it a failure (same home-and-hold behavior as
          `s`). While WAITING between episodes (after `s`/`f`, before `n`): finish the whole
          session instead (no more episodes).
    n  -- next: after you've reset the scene, start inference for the next episode.

Every episode's lightweight eval artifacts are always written: two mp4s (all 3 camera views
tiled into one frame, plus a second recording of just the `top` camera at its native
resolution/fps and max quality -- exactly as the robot saw it; the outcome is reflected only
in the filenames, not burned into the video), and every episode's outcome plus a running
success rate is appended to a per-session log directory:

    <output_dir>/<YYYYMMDD_HHMMSS>/
      videos/episode_001_success.mp4
      videos/episode_001_success_top.mp4
      videos/episode_002_failure.mp4
      videos/episode_002_failure_top.mp4
      log.jsonl        # one JSON line per episode
      summary.txt       # human-readable running + final success rate

Pass `--record_dataset=true` to ALSO record every episode as a native LeRobotDataset episode
(the same Parquet, per-camera-frames, and metadata layout produced by `lerobot-record`, with
each frame's `task` set to your `--task` prompt) -- so eval rollouts can be replayed with
`lerobot-replay-bi-yam`, inspected the same way as a teleop recording, or folded into a
training set. No `--dataset.*` flags needed: it's written to
`<output_dir>/<YYYYMMDD_HHMMSS>/dataset` (same session folder as the artifacts above) and
named `local/<timestamp>_<task_slug>`.

Episodes are saved RAW: no video encoding happens during the session (nothing to stall an
eval episode mid-rollout) or even at the end when you close the session -- camera frames are
just loose per-frame PNGs on disk. Once you're done evaling, encode them into the dataset's
mp4s with:

```shell
python scripts/encode_pending_videos.py --repo-id local/<timestamp>_<task_slug> \
  --root <output_dir>/<YYYYMMDD_HHMMSS>/dataset
```

Usage:

```shell
python scripts/eval_yam_http_policy.py \
  --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260322275072", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260322271881", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262522074294", "width": 640, "height": 360, "fps": 30}
}' \
  --server_url=https://untaken-eskimo-penholder.ngrok-free.dev/act \
  --task="Put all blocks into the box." \
  --actions_per_chunk=15 \
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
    ik_from_eef_pose,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say

SIDES = ("left", "right")
CAM_KEYS = ("top", "left", "right")
VIDEO_TILE_WH = (320, 240)  # per-camera tile size in the composed mp4 frame


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
class EvalYamHttpPolicyConfig:
    robot: BiYamFollowerConfig
    # If true, also record eval rollouts as a native LeRobotDataset episode (the lightweight
    # tiled-mp4/session-log artifacts are always written regardless). The dataset is written
    # under this same run's `<output_dir>/<timestamp>/dataset` directory, named after the
    # session timestamp -- see `_build_dataset_record_config`. Saved RAW (no video encoding
    # during or at the end of this session) -- run `scripts/encode_pending_videos.py`
    # afterward.
    record_dataset: bool = False
    server_url: str = "https://untaken-eskimo-penholder.ngrok-free.dev/act"
    task: str = "Put all blocks into the box."
    # How many of the returned chunk's 30 rows to actually execute before re-querying the
    # server (receding horizon). Lower = more reactive/robust to model error, higher =
    # fewer round trips. The reference server's chunk is 1.0s @ 30 rows -> 30fps.
    actions_per_chunk: int = 15
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
    # Session bookkeeping.
    max_episodes: int = 50
    output_dir: str = "eval_logs"
    video_fps: float | None = None  # None -> defaults to control_hz


def _current_arm_joint_pos(obs: dict, side: str) -> np.ndarray:
    return np.array([obs[f"{side}_joint_{i}.pos"] for i in range(6)])


def _current_arm_joint_pos_with_gripper(obs: dict, side: str) -> np.ndarray:
    return np.array([*_current_arm_joint_pos(obs, side), obs[f"{side}_gripper.pos"]])


def _state16_from_obs(obs: dict) -> np.ndarray:
    """Build the server's 16-D ABSOLUTE EEF state: left [x,y,z,qw,qx,qy,qz,gripper] ++
    right [same], via FK on the follower's live joint positions -- same FK helper and same
    per-arm layout (`EEF_POSE_NAMES_WITH_GRIPPER`) the training dataset's
    `observation.state_eef_absolute` column used (see `bi_yam_follower.py`)."""
    row = []
    for side in SIDES:
        pose = eef_pose_from_joint_pos(_current_arm_joint_pos_with_gripper(obs, side))
        row.extend(pose[axis] for axis in EEF_POSE_NAMES_WITH_GRIPPER)
    return np.asarray(row, dtype=np.float32)


def _slow_reset(
    robot: BiYamFollower,
    robot_action_processor,
    target_joint_pos: np.ndarray,
    duration_s: float,
    hz: float,
) -> None:
    """Linearly interpolate both arms' 6 joints from their current position to
    `target_joint_pos` over `duration_s`, in joint space (no IK needed). The gripper is
    interpolated open (1.0 -- see `bi_yam_leader.py`'s "not pressed = open (1)") over the
    same window, so every reset ends with both grippers open."""
    obs = robot.get_observation()
    start = {side: _current_arm_joint_pos(obs, side) for side in SIDES}
    gripper_start = {side: obs[f"{side}_gripper.pos"] for side in SIDES}

    n_steps = max(1, int(duration_s * hz))
    period_s = 1.0 / hz
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        alpha = step / n_steps
        action: dict[str, float] = {}
        for side in SIDES:
            q = (1 - alpha) * start[side] + alpha * target_joint_pos
            for j, val in enumerate(q):
                action[f"{side}_joint_{j}.pos"] = float(val)
            action[f"{side}_gripper.pos"] = float((1 - alpha) * gripper_start[side] + alpha * 1.0)

        robot_obs = robot.get_observation()
        processed_action = robot_action_processor((action, robot_obs))
        robot.send_action(processed_action)
        busy_wait(period_s - (time.perf_counter() - t0))


def _action_row_to_pose(row: np.ndarray, side_idx: int) -> dict[str, float]:
    """One arm's 8 values out of a flat 16-D action row -- `EEF_POSE_NAMES_WITH_GRIPPER` order."""
    values = row[side_idx * 8 : side_idx * 8 + 8]
    return dict(zip(EEF_POSE_NAMES_WITH_GRIPPER, (float(v) for v in values)))


def _augment_action_for_dataset(action: dict[str, float], row: np.ndarray, obs: dict) -> None:
    """Add the EEF action fields declared by BiYamFollower's native dataset schema."""
    for side_idx, side in enumerate(SIDES):
        target_pose = _action_row_to_pose(row, side_idx)
        for axis, value in target_pose.items():
            action[f"{side}_eef.{axis}"] = value

        if f"{side}_eef.x" in obs:
            current_pose = {axis: obs[f"{side}_eef.{axis}"] for axis in EEF_POSE_NAMES_WITH_GRIPPER}
            delta = eef_pose_delta(target_pose, current_pose)
            for axis, value in delta.items():
                action[f"{side}_eef_delta.{axis}"] = float(value)


def _build_dataset_record_config(cfg: EvalYamHttpPolicyConfig, timestamp: str) -> DatasetRecordConfig:
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
    """Retry a `requests` call a few times with linear backoff. This robot's WiFi DNS
    resolver has been observed to intermittently time out (`Temporary failure in name
    resolution`) for a few seconds at a time even though the tunnel/server are up -- not
    worth aborting a whole rollout, or even just the pre-flight health check, over that."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            logging.warning(f"request attempt {attempt}/{retries} failed: {e!r}; retrying in {backoff_s}s")
            time.sleep(backoff_s)
    raise RuntimeError(f"request failed after {retries} attempts") from last_exc


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


def _home(cfg: EvalYamHttpPolicyConfig, robot: BiYamFollower, robot_action_processor) -> None:
    log_say("Resetting to home position", cfg.play_sounds, blocking=True)
    _slow_reset(
        robot,
        robot_action_processor,
        np.asarray(cfg.reset_joint_pos, dtype=np.float32),
        cfg.reset_duration_s,
        cfg.reset_hz,
    )


def _cleanup_and_exit(
    cfg: EvalYamHttpPolicyConfig,
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
    cfg: EvalYamHttpPolicyConfig,
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
    period_s = 1.0 / cfg.control_hz
    t_start = time.perf_counter()
    step = 0
    outcome: str | None = None
    time_capped = False

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
        state16 = _state16_from_obs(obs)

        payload = {
            "top": np.asarray(obs["top"]),
            "left": np.asarray(obs["left"]),
            "right": np.asarray(obs["right"]),
            "instruction": cfg.task,
            "state": state16,
        }
        if cfg.num_steps is not None:
            payload["num_steps"] = cfg.num_steps

        t0 = time.perf_counter()
        resp = _request_with_retries(
            lambda: requests.post(
                cfg.server_url,
                json=payload,
                headers={"ngrok-skip-browser-warning": "1"},
                timeout=60,
            ).json()
        )
        if "error" in resp:
            raise RuntimeError(f"server returned an error: {resp['error']}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        actions = np.asarray(resp["actions"], dtype=np.float32)  # (chunk_size, 16), ABSOLUTE
        logging.info(
            f"episode={episode} step={step} server_dt_ms={dt_ms:.1f} "
            f"resp_dt_ms={resp.get('dt_ms', float('nan')):.1f} "
            f"quat_norm_dev={resp.get('quat_norm_dev', float('nan')):.2e}"
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
                target_pose = _action_row_to_pose(actions[i], side_idx)
                gripper_val = target_pose["gripper"]

                success, q6 = ik_from_eef_pose(target_pose, init_q[side])
                if not success:
                    logging.warning(f"episode={episode} step={step} row={i} {side}: IK did not converge")
                init_q[side] = q6

                for j, q in enumerate(q6):
                    action[f"{side}_joint_{j}.pos"] = float(q)
                action[f"{side}_gripper.pos"] = float(gripper_val)

            robot_obs = robot.get_observation()
            recorder.write(robot_obs)

            if dataset is not None:
                dataset_action = action.copy()
                _augment_action_for_dataset(dataset_action, actions[i], robot_obs)
                observation_frame = build_dataset_frame(
                    dataset.features, robot_observation_processor(robot_obs), prefix=OBS_STR
                )
                action_frame = build_dataset_frame(dataset.features, dataset_action, prefix=ACTION)

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
def eval_policy(cfg: EvalYamHttpPolicyConfig):
    init_logging()
    logging.info(cfg)

    # Imported here (not at module top) so a plain `--help` / config-only invocation
    # doesn't require json_numpy/requests to be importable.
    import json_numpy
    import requests

    # NOTE: do NOT call json_numpy.patch() until every heavy import (lerobot -> scipy ->
    # numpy.testing, pulled in lazily by eef_kinematics's FK/IK calls and by camera/robot
    # connect()) is done. json_numpy monkeypatches the stdlib `json` module process-wide,
    # and numpy.testing does its own unrelated json.loads() at import time that crashes
    # under the patched decoder hook. Same issue called out in
    # `host_policy_server_reference.py`'s module docstring. So: connect + do one FK call
    # first (forces the scipy import), patch only after.
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    robot = BiYamFollower(cfg.robot)
    robot.connect()

    obs = robot.get_observation()
    _state16_from_obs(obs)  # forces the lazy scipy import before patching json

    json_numpy.patch()

    # Sanity-check the server is up and its contract matches what this client assumes,
    # before moving any hardware.
    health = _request_with_retries(
        lambda: requests.get(cfg.server_url, headers={"ngrok-skip-browser-warning": "1"}, timeout=15).json()
    )
    logging.info(f"Server health: {health}")
    if health.get("state_dim") != 16 or health.get("action_dim") != 16:
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
