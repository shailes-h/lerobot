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
  --duration_s=60
```
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots.bi_yam_follower.bi_yam_follower import BiYamFollower
from lerobot.robots.bi_yam_follower.config_bi_yam_follower import BiYamFollowerConfig
from lerobot.robots.bi_yam_follower.eef_kinematics import (
    EEF_POSE_NAMES_WITH_GRIPPER,
    eef_pose_from_joint_pos,
    ik_from_eef_pose,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say

SIDES = ("left", "right")


@dataclass
class EvalYamHttpPolicyConfig:
    robot: BiYamFollowerConfig
    server_url: str = "https://untaken-eskimo-penholder.ngrok-free.dev/act"
    task: str = "Put all blocks into the box."
    # How many of the returned chunk's 30 rows to actually execute before re-querying the
    # server (receding horizon). Lower = more reactive/robust to model error, higher =
    # fewer round trips. The reference server's chunk is 1.0s @ 30 rows -> 30fps.
    actions_per_chunk: int = 15
    control_hz: float = 30.0
    duration_s: float = 60.0
    play_sounds: bool = True
    # Diagnostic passthrough to the server; None lets it use its own default.
    num_steps: int | None = None
    # Slowly interpolate both arms' 6 joints from wherever they currently are to
    # `reset_joint_pos` before the policy loop starts, instead of snapping/letting the
    # policy's first chunk do it -- avoids a jerky first move if the arms were left in an
    # arbitrary teleop/previous-episode pose. Gripper is left untouched. Set
    # `--no-reset_before_start` to skip.
    reset_before_start: bool = True
    reset_joint_pos: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    reset_duration_s: float = 4.0
    reset_hz: float = 20.0


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
    `target_joint_pos` over `duration_s`, in joint space (no IK needed). Gripper is left
    alone -- each step's action carries forward whatever the gripper's live position is, so
    it doesn't jump open/closed during the reset."""
    obs = robot.get_observation()
    start = {side: _current_arm_joint_pos(obs, side) for side in SIDES}
    gripper = {side: obs[f"{side}_gripper.pos"] for side in SIDES}

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
            action[f"{side}_gripper.pos"] = float(gripper[side])

        robot_obs = robot.get_observation()
        processed_action = robot_action_processor((action, robot_obs))
        robot.send_action(processed_action)
        busy_wait(period_s - (time.perf_counter() - t0))


def _action_row_to_pose(row: np.ndarray, side_idx: int) -> dict[str, float]:
    """One arm's 8 values out of a flat 16-D action row -- `EEF_POSE_NAMES_WITH_GRIPPER` order."""
    values = row[side_idx * 8 : side_idx * 8 + 8]
    return dict(zip(EEF_POSE_NAMES_WITH_GRIPPER, (float(v) for v in values)))


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
    robot_action_processor = make_default_robot_action_processor()
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

    if cfg.reset_before_start:
        log_say("Resetting to home position", cfg.play_sounds, blocking=True)
        _slow_reset(
            robot,
            robot_action_processor,
            np.asarray(cfg.reset_joint_pos, dtype=np.float32),
            cfg.reset_duration_s,
            cfg.reset_hz,
        )

    log_say(f"Starting eval: {cfg.task}", cfg.play_sounds, blocking=True)

    # IK warm-start, carried frame-to-frame -- same pattern as lerobot_replay_bi_yam.
    obs = robot.get_observation()
    init_q = {side: _current_arm_joint_pos(obs, side) for side in SIDES}

    period_s = 1.0 / cfg.control_hz
    t_start = time.perf_counter()
    step = 0

    while time.perf_counter() - t_start < cfg.duration_s:
        obs = robot.get_observation()
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
            f"step={step} server_dt_ms={dt_ms:.1f} resp_dt_ms={resp.get('dt_ms', float('nan')):.1f} "
            f"quat_norm_dev={resp.get('quat_norm_dev', float('nan')):.2e}"
        )

        n_exec = min(cfg.actions_per_chunk, len(actions))
        for i in range(n_exec):
            row_start = time.perf_counter()
            action: dict[str, float] = {}
            for side_idx, side in enumerate(SIDES):
                target_pose = _action_row_to_pose(actions[i], side_idx)
                gripper_val = target_pose["gripper"]

                success, q6 = ik_from_eef_pose(target_pose, init_q[side])
                if not success:
                    logging.warning(f"step={step} row={i} {side}: IK did not converge, using best attempt")
                init_q[side] = q6

                for j, q in enumerate(q6):
                    action[f"{side}_joint_{j}.pos"] = float(q)
                action[f"{side}_gripper.pos"] = float(gripper_val)

            robot_obs = robot.get_observation()
            processed_action = robot_action_processor((action, robot_obs))
            robot.send_action(processed_action)

            busy_wait(period_s - (time.perf_counter() - row_start))

        step += 1

    log_say("Eval done", cfg.play_sounds, blocking=True)
    robot.disconnect()


def main():
    eval_policy()


if __name__ == "__main__":
    main()
