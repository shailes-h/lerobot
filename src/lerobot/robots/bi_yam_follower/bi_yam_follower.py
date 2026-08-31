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

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np
import portal

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .config_bi_yam_follower import BiYamFollowerConfig
from .eef_kinematics import EEF_POSE_NAMES_WITH_GRIPPER, eef_pose_from_joint_pos

logger = logging.getLogger(__name__)


class YamArmClient:
    """Client interface for a single Yam arm using the portal RPC framework."""

    def __init__(self, port: int, host: str = "localhost"):
        """
        Initialize the Yam arm client.

        Args:
            port: Server port for the arm
            host: Server host address
        """
        self.port = port
        self.host = host
        self._client = None

    def connect(self):
        """Connect to the arm server."""
        logger.info(f"Connecting to Yam arm server at {self.host}:{self.port}")
        self._client = portal.Client(f"{self.host}:{self.port}")
        logger.info(f"Successfully connected to Yam arm server at {self.host}:{self.port}")

    def disconnect(self):
        """Disconnect from the arm server."""
        if self._client is not None:
            logger.info(f"Disconnecting from Yam arm server at {self.host}:{self.port}")
            self._client = None

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._client is not None

    def num_dofs(self) -> int:
        """Get the number of degrees of freedom."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command joint positions."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        self._client.command_joint_pos(joint_pos)

    def get_observations(self) -> dict[str, np.ndarray]:
        """Get current observations including joint positions, velocities, etc."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_observations().result()


class BiYamFollower(Robot):
    """
    Bimanual Yam Arms follower robot using the i2rt library.

    This robot controls two Yam arms simultaneously. Each arm communicates via
    the portal RPC framework with servers running on different ports.

    Expected setup:
    - Two Yam follower arms connected via CAN interfaces
    - Server processes running for each arm (see bimanual_lead_follower.py)
    - Left arm server on port 1235 (default)
    - Right arm server on port 1234 (default)
    """

    config_class = BiYamFollowerConfig
    name = "bi_yam_follower"

    def __init__(self, config: BiYamFollowerConfig):
        super().__init__(config)
        self.config = config

        # Create clients for left and right arms
        self.left_arm = YamArmClient(port=config.left_arm_port, host=config.server_host)
        self.right_arm = YamArmClient(port=config.right_arm_port, host=config.server_host)

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

        # Store number of DOFs (will be set after connection)
        self._left_dofs = None
        self._right_dofs = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Define motor feature types for both arms (positions)."""
        return {
            **self._build_per_arm_features("left", "pos"),
            **self._build_per_arm_features("right", "pos"),
        }

    def _build_per_arm_features(self, side: str, suffix: str) -> dict[str, type]:
        """Build a flat per-motor feature dict for one arm, keyed `{side}_{joint|gripper}.{suffix}`.

        7 DOFs are assumed to be 6 joints + gripper; otherwise all motors are named
        `joint_i`. Positions (`.pos`) and torques (`.eff`) share this layout so they
        line up 1-to-1 by motor.
        """
        dofs_attr = self._left_dofs if side == "left" else self._right_dofs
        dofs = dofs_attr if dofs_attr is not None else 7

        features: dict[str, type] = {}
        for i in range(dofs):
            if dofs == 7 and i == dofs - 1:
                features[f"{side}_gripper.{suffix}"] = float
            else:
                features[f"{side}_joint_{i}.{suffix}"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Define camera feature types."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Return observation features including motors and cameras.

        Note: torques are intentionally exposed as a separate dataset column via
        `extra_dataset_features` (gated on `config.record_torques`) rather than
        folded into `observation.state`, so pretrained policies trained on
        positions-only datasets remain compatible.
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Return action features (motor positions)."""
        return self._motors_ft

    @property
    def extra_dataset_features(self) -> dict[str, dict]:
        """Extra columns that bypass the standard hw_to_dataset_features lumping.

        Gated on two independent config flags:
        - `record_torques`: `observation.left_torques` / `observation.right_torques`,
          from the flat `.eff` keys produced by `get_observation()`.
        - `record_eef_pose`: `observation.state_eef_absolute` (both arms, from FK on the
          follower's own joint positions), `action_eef_absolute` (both arms, from FK on the
          leader's commanded joint positions), and `action_eef_delta` (target minus this
          tick's observed follower pose). Each carries the gripper DOF alongside the 7 pose
          values (same control authority as the joint-space columns) — see
          `EEF_POSE_NAMES_WITH_GRIPPER`. See `dataset_feature_renames` for the matching
          joint-space renames. Consumed by `lerobot.scripts.lerobot_record` via
          `getattr(robot, "extra_dataset_features", {})`.
        """
        features: dict[str, dict] = {}

        if getattr(self.config, "record_torques", False):
            for side in ("left", "right"):
                names = list(self._build_per_arm_features(side, "eff").keys())
                features[f"observation.{side}_torques"] = {
                    "dtype": "float32",
                    "shape": (len(names),),
                    "names": names,
                }

        if getattr(self.config, "record_eef_pose", False):
            abs_names = [
                f"{side}_eef.{axis}" for side in ("left", "right") for axis in EEF_POSE_NAMES_WITH_GRIPPER
            ]
            features["observation.state_eef_absolute"] = {
                "dtype": "float32",
                "shape": (len(abs_names),),
                "names": abs_names,
            }
            # The leader teleoperator's get_action() is what actually populates these
            # `{side}_eef.*` values for the action side (see bi_yam_leader.py); this
            # just declares the resulting dataset column.
            features["action_eef_absolute"] = {
                "dtype": "float32",
                "shape": (len(abs_names),),
                "names": abs_names,
            }
            # `{side}_eef_delta.*` is populated by BiYamLeader.augment_action_with_observation
            # (target_eef - this tick's observed follower eef), which only has something to
            # compute once this same flag has put `{side}_eef.*` into the observation dict.
            delta_names = [
                f"{side}_eef_delta.{axis}" for side in ("left", "right") for axis in EEF_POSE_NAMES_WITH_GRIPPER
            ]
            features["action_eef_delta"] = {
                "dtype": "float32",
                "shape": (len(delta_names),),
                "names": delta_names,
            }

        return features

    @property
    def dataset_feature_renames(self) -> dict[str, str]:
        """Rename the standard `action` / `observation.state` columns when EEF columns are on.

        Keeps the joint-space data (still the full, unmodified DOF set) but names it
        `action_joint_angles` / `observation.state_joint_angles` so it sits clearly
        alongside the `*_eef_absolute` / `*_eef_delta` columns from `extra_dataset_features`
        instead of colliding with the generic `action` / `observation.state` names.
        """
        if not getattr(self.config, "record_eef_pose", False):
            return {}
        return {"action": "action_joint_angles", "observation.state": "observation.state_joint_angles"}

    @property
    def is_connected(self) -> bool:
        """Check if both arms and all cameras are connected."""
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to both arm servers and cameras.

        Args:
            calibrate: Not used for Yam arms (kept for API compatibility)
        """
        logger.info("Connecting to bimanual Yam follower robot")

        # Connect to arm servers
        self.left_arm.connect()
        self.right_arm.connect()

        # Get number of DOFs from each arm
        self._left_dofs = self.left_arm.num_dofs()
        self._right_dofs = self.right_arm.num_dofs()

        logger.info(f"Left arm DOFs: {self._left_dofs}, Right arm DOFs: {self._right_dofs}")

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        logger.info("Successfully connected to bimanual Yam follower robot")

    @property
    def is_calibrated(self) -> bool:
        """Yam arms don't require calibration in the lerobot sense."""
        return self.is_connected

    def calibrate(self) -> None:
        """Yam arms don't require calibration in the lerobot sense."""
        pass

    def configure(self) -> None:
        """Configure the robot (not needed for Yam arms)."""
        pass

    def setup_motors(self) -> None:
        """Setup motors (not needed for Yam arms)."""
        pass

    def get_observation(self) -> dict[str, Any]:
        """
        Get current observation from both arms and cameras.

        Returns:
            Dictionary with joint positions (and optionally torques when
            `config.record_torques=True`) for both arms as flat
            `{side}_{joint|gripper}.{pos|eff}` keys, plus camera images.
        """
        obs_dict: dict[str, Any] = {}

        left_obs = self.left_arm.get_observations()
        self._populate_arm_obs(obs_dict, side="left", arm_obs=left_obs)

        right_obs = self.right_arm.get_observations()
        self._populate_arm_obs(obs_dict, side="right", arm_obs=right_obs)

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def _populate_arm_obs(self, obs_dict: dict[str, Any], side: str, arm_obs: dict[str, np.ndarray]) -> None:
        """Emit flat `.pos` (and optionally `.eff`) keys for one arm from a single RPC response.

        The i2rt server returns joint and gripper components separately; we concatenate
        them so a 7-DOF (6 joints + gripper) arm yields keys 0..5 + `gripper`. Torques
        come from `joint_eff` / `gripper_eff` (motor effort feedback over CAN) and are
        only emitted when `config.record_torques=True`.
        """
        joint_pos = arm_obs["joint_pos"]
        has_gripper = "gripper_pos" in arm_obs

        if has_gripper:
            joint_pos = np.concatenate([joint_pos, arm_obs["gripper_pos"]])
        else:
            # Some follower servers (e.g. i2rt's minimum_gello.py) fold the gripper into
            # a single flat `joint_pos` array instead of exposing a separate `gripper_pos`
            # key. Fall back to the arm's known DOF count so the last element is still
            # labeled `{side}_gripper.pos`, matching the schema built by `_build_per_arm_features`.
            dofs = self._left_dofs if side == "left" else self._right_dofs
            has_gripper = dofs == 7 and len(joint_pos) == 7

        for i, pos in enumerate(joint_pos):
            if has_gripper and i == len(joint_pos) - 1:
                obs_dict[f"{side}_gripper.pos"] = pos
            else:
                obs_dict[f"{side}_joint_{i}.pos"] = pos

        if getattr(self.config, "record_torques", False):
            joint_eff = arm_obs.get("joint_eff")
            if joint_eff is not None:
                if has_gripper and "gripper_eff" in arm_obs:
                    joint_eff = np.concatenate([joint_eff, arm_obs["gripper_eff"]])

                for i, eff in enumerate(joint_eff):
                    if has_gripper and i == len(joint_eff) - 1:
                        obs_dict[f"{side}_gripper.eff"] = eff
                    else:
                        obs_dict[f"{side}_joint_{i}.eff"] = eff

        if getattr(self.config, "record_eef_pose", False):
            eef_pose = eef_pose_from_joint_pos(joint_pos)
            for axis in EEF_POSE_NAMES_WITH_GRIPPER:
                obs_dict[f"{side}_eef.{axis}"] = eef_pose[axis]

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action commands to both arms.

        Args:
            action: Dictionary with joint positions for both arms

        Returns:
            The action that was sent
        """
        # Extract left arm actions
        left_action = []
        for i in range(self._left_dofs):
            # Last DOF is gripper if we have 7 DOFs
            key = "left_gripper.pos" if (self._left_dofs == 7 and i == self._left_dofs - 1) else f"left_joint_{i}.pos"
            if key in action:
                left_action.append(action[key])

        # Extract right arm actions
        right_action = []
        for i in range(self._right_dofs):
            # Last DOF is gripper if we have 7 DOFs
            key = "right_gripper.pos" if (self._right_dofs == 7 and i == self._right_dofs - 1) else f"right_joint_{i}.pos"
            if key in action:
                right_action.append(action[key])

        # Apply max_relative_target if configured
        if self.config.left_arm_max_relative_target is not None:
            left_current = self.left_arm.get_joint_pos()
            left_action = self._clip_relative_target(
                np.array(left_action), left_current, self.config.left_arm_max_relative_target
            )

        if self.config.right_arm_max_relative_target is not None:
            right_current = self.right_arm.get_joint_pos()
            right_action = self._clip_relative_target(
                np.array(right_action), right_current, self.config.right_arm_max_relative_target
            )

        # Send commands to arms
        if len(left_action) > 0:
            self.left_arm.command_joint_pos(np.array(left_action))

        if len(right_action) > 0:
            self.right_arm.command_joint_pos(np.array(right_action))

        return action

    def _clip_relative_target(
        self, target: np.ndarray, current: np.ndarray, max_relative: float | dict[str, float]
    ) -> np.ndarray:
        """
        Clip target positions to be within max_relative distance from current position.

        Args:
            target: Target joint positions
            current: Current joint positions
            max_relative: Maximum relative change allowed (per joint or global)

        Returns:
            Clipped target positions
        """
        if isinstance(max_relative, dict):
            # Per-joint limits
            clipped = target.copy()
            for i in range(len(target)):
                key = f"joint_{i}.pos"
                if key in max_relative:
                    max_delta = max_relative[key]
                    clipped[i] = np.clip(target[i], current[i] - max_delta, current[i] + max_delta)
            return clipped
        else:
            # Global limit for all joints
            return np.clip(target, current - max_relative, current + max_relative)

    def disconnect(self):
        """Disconnect from both arms and cameras."""
        logger.info("Disconnecting from bimanual Yam follower robot")

        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info("Disconnected from bimanual Yam follower robot")

