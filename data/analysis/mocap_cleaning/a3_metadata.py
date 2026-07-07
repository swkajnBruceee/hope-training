"""Lightweight A3 metadata for retarget/refinement tools.

This module intentionally avoids Isaac/IsaacLab imports so data-prep CLIs can
run in a plain Python environment.
"""

from __future__ import annotations

A3_POLICY_JOINT_ORDER = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

A3_WRIST_BODY = "right_wrist_yaw_Link"
A3_RACKET_BODY = "pingpang_red_Link"
A3_MOUNT_OFFSET_M = (0.210211399202899, 0.0320784994676765, 0.0320358706296689)
A3_MOUNT_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)
A3_MOUNT_NORMAL_AXIS = 1
A3_MOUNT_NORMAL_SIGN = 1.0
A3_RACKET_TANGENT_AXIS = 0
A3_RACKET_UP_AXIS = 2

A3_ACTIVE_JOINTS_FIRST_PASS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

A3_LOCKED_JOINTS_FIRST_PASS = [
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

A3_WEAK_TRACK_JOINTS_FIRST_PASS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

A3_DEFAULT_JOINT_POS = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.12,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.8,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.3,
    "right_shoulder_roll_joint": -0.12,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.8,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "left_hip_pitch_joint": -0.1311,
    "left_hip_roll_joint": 0.0056,
    "left_hip_yaw_joint": -0.0348,
    "left_knee_joint": 0.2468,
    "left_ankle_pitch_joint": -0.1204,
    "left_ankle_roll_joint": -0.0078,
    "right_hip_pitch_joint": -0.1311,
    "right_hip_roll_joint": -0.0056,
    "right_hip_yaw_joint": 0.0348,
    "right_knee_joint": 0.2468,
    "right_ankle_pitch_joint": -0.1204,
    "right_ankle_roll_joint": 0.0078,
}
