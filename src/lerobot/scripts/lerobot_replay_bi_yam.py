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

"""
Replays a `bi_yam_follower` episode recorded with `--robot.record_eef_pose=true`, using one
of three action representations. This exists as a separate script from `lerobot-replay`
(rather than a `--method` bolted onto it) because that script assumes a single generic
`action` column sent straight to `robot.send_action()`; this one needs to read one of three
differently-named/shaped columns and, for the two EEF methods, run IK to turn a pose back
into joint angles before sending.

Examples:

```shell
# Sanity check: replay the exact joint-space actions that were recorded.
lerobot-replay-bi-yam \
  --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --dataset.repo_id=local/sanity --dataset.root=./datasets/sanity --dataset.episode=0 \
  --method=joint

# IK-based: replay by driving the arm to each frame's absolute FK'd pose.
lerobot-replay-bi-yam ... --method=eef_absolute

# IK-based, closed-loop: replay by adding each frame's recorded delta to the arm's live
# current pose (matches how the delta was recorded relative to the follower's observation).
lerobot-replay-bi-yam ... --method=eef_delta
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots.bi_yam_follower.bi_yam_follower import BiYamFollower
from lerobot.robots.bi_yam_follower.config_bi_yam_follower import BiYamFollowerConfig
from lerobot.robots.bi_yam_follower.eef_kinematics import (
    EEF_POSE_NAMES_WITH_GRIPPER,
    apply_eef_pose_delta,
    eef_pose_from_joint_pos,
    ik_from_eef_pose,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say

SIDES = ("left", "right")


class ReplayMethod(str, Enum):
    JOINT = "joint"
    EEF_ABSOLUTE = "eef_absolute"
    EEF_DELTA = "eef_delta"


@dataclass
class DatasetReplayConfig:
    repo_id: str
    episode: int
    root: str | Path | None = None
    fps: int = 30


@dataclass
class ReplayBiYamConfig:
    robot: BiYamFollowerConfig
    dataset: DatasetReplayConfig
    method: ReplayMethod = ReplayMethod.JOINT
    play_sounds: bool = True
    # For eef_absolute/eef_delta: before replaying, move to the episode's frame-0
    # observation.state_joint_angles (the ORIGINAL episode's actual starting follower pose)
    # and let it settle. Matters most for eef_delta, since its recorded deltas were computed
    # against that specific starting state — any offset from it doesn't self-correct and
    # carries straight through every subsequent frame's reconstructed target.
    snap_to_start: bool = True
    snap_settle_s: float = 2.0
    # eef_delta only: after sending each frame's delta-reconstructed action, immediately
    # also send that same frame's recorded observation.state_joint_angles — i.e. force the
    # arm back to the ORIGINAL episode's ground-truth position before computing the next
    # frame's delta. This isolates whether the delta representation reconstructs correctly
    # frame-by-frame (each one computed from the exact `current` it was recorded against),
    # decoupled from real tracking error/drift accumulating across frames.
    realign_to_recording: bool = True


def _current_arm_joint_pos(obs: dict, side: str) -> np.ndarray:
    """[joint_0..joint_5] for one arm, read from a `robot.get_observation()` dict."""
    return np.array([obs[f"{side}_joint_{i}.pos"] for i in range(6)])


def _current_arm_joint_pos_with_gripper(obs: dict, side: str) -> np.ndarray:
    """[joint_0..joint_5, gripper] for one arm, read from a `robot.get_observation()` dict."""
    return np.array([*_current_arm_joint_pos(obs, side), obs[f"{side}_gripper.pos"]])


def _row_pose(row: np.ndarray, names: list[str], side: str) -> dict[str, float]:
    """Pull one side's {x,y,z,qw,qx,qy,qz,gripper} out of a flat action_eef_absolute/_delta row."""
    row_by_name = dict(zip(names, row))
    return {axis: row_by_name[f"{side}_eef.{axis}"] for axis in EEF_POSE_NAMES_WITH_GRIPPER}


def _row_pose_delta(row: np.ndarray, names: list[str], side: str) -> dict[str, float]:
    row_by_name = dict(zip(names, row))
    return {axis: row_by_name[f"{side}_eef_delta.{axis}"] for axis in EEF_POSE_NAMES_WITH_GRIPPER}


@parser.wrap()
def replay(cfg: ReplayBiYamConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()
    robot = BiYamFollower(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.episode)

    joint_names = dataset.features["action_joint_angles"]["names"]
    joint_rows = np.stack(episode_frames["action_joint_angles"])

    if cfg.method != "joint":
        pose_key = "action_eef_absolute" if cfg.method == "eef_absolute" else "action_eef_delta"
        if pose_key not in dataset.features:
            raise ValueError(
                f"Dataset has no '{pose_key}' column — was it recorded with "
                f"--robot.record_eef_pose=true?"
            )
        pose_names = dataset.features[pose_key]["names"]
        pose_rows = np.stack(episode_frames[pose_key])

    state_names = dataset.features["observation.state_joint_angles"]["names"]
    state_rows = np.stack(episode_frames["observation.state_joint_angles"])

    robot.connect()

    if cfg.method != "joint" and cfg.snap_to_start:
        start_action = dict(zip(state_names, state_rows[0]))
        log_say("Snapping to episode start pose", cfg.play_sounds, blocking=True)
        robot.send_action(robot_action_processor((start_action, robot.get_observation())))
        time.sleep(cfg.snap_settle_s)

    log_say(f"Replaying episode ({cfg.method})", cfg.play_sounds, blocking=True)

    # IK warm-start: seeded from the robot's live state (post-snap, if any), then carried
    # frame-to-frame so consecutive solves stay near each other instead of each
    # re-searching from scratch.
    init_q = {side: _current_arm_joint_pos(robot.get_observation(), side) for side in SIDES} if cfg.method != "joint" else None

    for idx in range(len(episode_frames)):
        start_t = time.perf_counter()

        if cfg.method == "joint":
            action = dict(zip(joint_names, joint_rows[idx]))
        else:
            obs = robot.get_observation()
            action = {}
            for side in SIDES:
                if cfg.method == "eef_absolute":
                    target_pose = _row_pose(pose_rows[idx], pose_names, side)
                else:  # eef_delta: recorded relative to the follower's own observation, so
                    # reconstruct the absolute target from *this tick's* live pose, not the
                    # leader's — that's the whole point of testing this representation.
                    current_pose = eef_pose_from_joint_pos(_current_arm_joint_pos_with_gripper(obs, side))
                    delta_pose = _row_pose_delta(pose_rows[idx], pose_names, side)
                    target_pose = apply_eef_pose_delta(current_pose, delta_pose)

                # Gripper rides along in the EEF pose dict itself (it isn't part of the IK
                # problem — see ik_from_eef_pose — but the target/reconstructed pose still
                # carries it, same control authority as the joint-space `_gripper.pos` action).
                gripper_val = target_pose["gripper"]

                success, q6 = ik_from_eef_pose(target_pose, init_q[side])
                if not success:
                    logging.warning(f"[{cfg.method}] frame {idx} {side}: IK did not converge, using best attempt")
                init_q[side] = q6

                for i, q in enumerate(q6):
                    action[f"{side}_joint_{i}.pos"] = float(q)
                action[f"{side}_gripper.pos"] = float(gripper_val)

        robot_obs = robot.get_observation()
        processed_action = robot_action_processor((action, robot_obs))
        robot.send_action(processed_action)

        if cfg.method == "eef_delta" and cfg.realign_to_recording:
            # Snap back to this frame's actual recorded follower state before the next
            # frame's delta gets reconstructed against it (see `realign_to_recording` docs).
            realign_action = dict(zip(state_names, state_rows[idx]))
            robot.send_action(robot_action_processor((realign_action, robot.get_observation())))
            # We know exactly where this command is driving the arm, so seed the next frame's
            # IK warm-start from it directly rather than re-reading (possibly not-yet-settled)
            # live joint positions.
            for side in SIDES:
                init_q[side] = np.array([realign_action[f"{side}_joint_{i}.pos"] for i in range(6)])

        dt_s = time.perf_counter() - start_t
        busy_wait(1 / dataset.fps - dt_s)

    robot.disconnect()


def main():
    replay()


if __name__ == "__main__":
    main()
