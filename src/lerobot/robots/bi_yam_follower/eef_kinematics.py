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

"""Shared forward-kinematics helper for the bimanual YAM setup.

Both `BiYamFollower` (for `observation.{side}_eef_pose`) and `BiYamLeader` (for
`action.{side}_eef_pose`) need to turn a 7-DOF joint_pos array (6 arm joints + gripper)
into an end-effector pose. They share one `i2rt.robots.kinematics.Kinematics` model built
from the YAM arm + linear_4310 gripper XML — the arm chain (joints 1-6) and the `grasp_site`
it exposes are identical regardless of which gripper/teaching-handle is physically mounted,
so this single model is valid for both the follower (which has a linear_4310 gripper) and
the leader (which has a teaching handle): the leader's action is a target for the
follower's gripper anyway.
"""

from functools import lru_cache

import numpy as np

# EEF pose feature layout shared by the observation and action columns.
EEF_POSE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]

# Full dataset column layout: pose + the gripper DOF, so the EEF action/observation
# columns carry the same control authority as the joint-space ones (gripper doesn't
# affect `grasp_site`/FK at all, but a policy predicting EEF actions still needs to
# command the gripper, so it rides alongside the pose as a plain extra scalar).
EEF_POSE_NAMES_WITH_GRIPPER = EEF_POSE_NAMES + ["gripper"]


@lru_cache(maxsize=1)
def _get_kinematics():
    """Lazily build (and cache) the shared FK model. Imports i2rt/mujoco/mink on first use."""
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    return Kinematics(xml_path, "grasp_site")


def eef_pose_to_matrix(pose: dict[str, float]) -> np.ndarray:
    """Inverse of `eef_pose_from_joint_pos`'s pose extraction: dict -> 4x4 world-frame transform."""
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([pose["qx"], pose["qy"], pose["qz"], pose["qw"]]).as_matrix()
    matrix[:3, 3] = [pose["x"], pose["y"], pose["z"]]
    return matrix


def apply_eef_pose_delta(current_pose: dict[str, float], delta_pose: dict[str, float]) -> dict[str, float]:
    """Inverse of `eef_pose_delta`: `current_pose + delta_pose` -> the absolute target pose.

    i.e. `apply_eef_pose_delta(current, eef_pose_delta(target, current)) == target`.

    If both poses carry a `"gripper"` key, it's combined the same way as position (plain
    sum) since it's just a scalar joint value, not part of the rotation; omitted from the
    result if either pose lacks it.
    """
    from scipy.spatial.transform import Rotation

    x = current_pose["x"] + delta_pose["x"]
    y = current_pose["y"] + delta_pose["y"]
    z = current_pose["z"] + delta_pose["z"]

    r_current = Rotation.from_quat(
        [current_pose["qx"], current_pose["qy"], current_pose["qz"], current_pose["qw"]]
    )
    r_delta = Rotation.from_quat([delta_pose["qx"], delta_pose["qy"], delta_pose["qz"], delta_pose["qw"]])
    qx, qy, qz, qw = (r_delta * r_current).as_quat()

    result = {"x": x, "y": y, "z": z, "qw": qw, "qx": qx, "qy": qy, "qz": qz}
    if "gripper" in current_pose and "gripper" in delta_pose:
        result["gripper"] = current_pose["gripper"] + delta_pose["gripper"]
    return result


def ik_from_eef_pose(target_pose: dict[str, float], init_joint_pos: np.ndarray) -> tuple[bool, np.ndarray]:
    """Solve for the 6 arm joint angles that reach `target_pose` at `grasp_site`.

    Args:
        target_pose: dict with keys `EEF_POSE_NAMES` (a `"gripper"` key, if present, is
            ignored here — the gripper doesn't affect `grasp_site`, so it isn't part of the
            IK problem; callers need to command it separately from `target_pose["gripper"]`).
        init_joint_pos: warm-start joint configuration; only the first 6 (arm) entries are
            used, the gripper DOFs don't affect `grasp_site` (see `eef_pose_from_joint_pos`).

    Returns:
        (success, joint_pos_6): `success` is False if the solver didn't converge (see
        `i2rt.robots.kinematics.Kinematics.ik`'s `max_iters`/thresholds) — the returned
        joint_pos_6 is still its best attempt and callers should decide whether to use it
        (e.g. skip the frame, or fall back to a slower blend) based on `success`.
    """
    init_q = np.zeros(8)
    init_q[:6] = init_joint_pos[:6]

    matrix = eef_pose_to_matrix(target_pose)
    success, q = _get_kinematics().ik(matrix, "grasp_site", init_q=init_q)
    return success, q[:6]


def eef_pose_delta(target_pose: dict[str, float], current_pose: dict[str, float]) -> dict[str, float]:
    """Compute `target_pose - current_pose` as a relative pose.

    Position: plain difference. Orientation: the relative rotation R_delta such that
    R_delta * R_current == R_target (i.e. applying R_delta to the current orientation
    yields the target orientation) — a plain quaternion subtraction is not a valid rotation.
    Gripper (if present in both poses): plain difference, same as position — it's just a
    joint value, not part of the rotation.

    Both `target_pose` and `current_pose` are dicts with keys `EEF_POSE_NAMES` (as returned
    by `eef_pose_from_joint_pos`), optionally plus `"gripper"`. Returns a dict with the same
    keys that were present in both inputs.
    """
    from scipy.spatial.transform import Rotation

    dx = target_pose["x"] - current_pose["x"]
    dy = target_pose["y"] - current_pose["y"]
    dz = target_pose["z"] - current_pose["z"]

    r_target = Rotation.from_quat([target_pose["qx"], target_pose["qy"], target_pose["qz"], target_pose["qw"]])
    r_current = Rotation.from_quat(
        [current_pose["qx"], current_pose["qy"], current_pose["qz"], current_pose["qw"]]
    )
    qx, qy, qz, qw = (r_target * r_current.inv()).as_quat()

    result = {"x": dx, "y": dy, "z": dz, "qw": qw, "qx": qx, "qy": qy, "qz": qz}
    if "gripper" in target_pose and "gripper" in current_pose:
        result["gripper"] = target_pose["gripper"] - current_pose["gripper"]
    return result


def eef_pose_from_joint_pos(joint_pos: np.ndarray) -> dict[str, float]:
    """Compute the end-effector pose for one arm from its 7-DOF joint_pos (6 joints + gripper).

    Returns a flat dict with keys "x", "y", "z", "qw", "qx", "qy", "qz" (world-frame
    position in meters and orientation as a wxyz quaternion), plus `"gripper"` when
    `joint_pos` carries a 7th element. The gripper DOF does not affect `grasp_site`/the FK
    result at all (it's fixed relative to the joint6/gripper-mount body, unlike the finger
    joints) — it's just carried through verbatim so EEF-space actions/observations keep the
    same control authority over the gripper as the joint-space ones.
    """
    from scipy.spatial.transform import Rotation

    q = np.zeros(8)
    q[:6] = joint_pos[:6]
    if len(joint_pos) > 6:
        q[6] = q[7] = joint_pos[6]

    transform = _get_kinematics().fk(q)
    pos = transform[:3, 3]
    # scipy quats are (x, y, z, w); we store (w, x, y, z).
    qx, qy, qz, qw = Rotation.from_matrix(transform[:3, :3]).as_quat()

    pose = {"x": pos[0], "y": pos[1], "z": pos[2], "qw": qw, "qx": qx, "qy": qy, "qz": qz}
    if len(joint_pos) > 6:
        pose["gripper"] = joint_pos[6]
    return pose
