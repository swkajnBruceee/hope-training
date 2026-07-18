#!/usr/bin/env python3
"""Write the frozen first-round A3 fixed-stand body-drive executor contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a3_strike_contract import validate_executor_contract


OFFICIAL_31_DOF = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
STRIKE_JOINTS = {
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
}


def build_contract() -> dict:
    joints = []
    for index, name in enumerate(OFFICIAL_31_DOF):
        is_head = name.startswith("head_")
        is_strike = name in STRIKE_JOINTS
        joints.append({
            "joint_name": name,
            "sdk_index": index,
            "ownership": "strike_optimized" if is_strike else "fixed_stand_baseline",
            "q_source": "actuator_aware_trajectory" if is_strike else ("zero_head_hold" if is_head else "a3_default_angles"),
            "dq_source": "zero",
            "tau_ff_source": "zero",
            "kp_source": "head_hold_kp_40" if is_head else "a3_policy_parameters.a3_kps",
            "kd_source": "head_hold_kd_2" if is_head else "a3_policy_parameters.a3_kds",
            "stand_to_strike_blend": "pd_stand_3s_then_linear_100ms",
            "post_hit_recovery": "linear_return_to_baseline_120ms",
        })
    return {
        "schema_version": 1,
        "executor_contract_id": "a3_t2d5_body_drive_fixed_stand_diag_v1",
        "runtime_class": "official_standalone_sil_diagnostic",
        "deployment_approved": False,
        "policy_hz": 50.0,
        "startup_state_machine": ["PASSIVE", "PD_STAND", "MOTION_OR_POLICY"],
        "pd_stand_duration_s": 3.0,
        "transport": "iceoryx",
        "backend": "a3_deploy_example::RobotIOBackend/A3AimrtBackend",
        "state_topics": [
            "/body_drive/waist_joint_state", "/body_drive/leg_joint_state", "/body_drive/arm_joint_state", "/body_drive/neck_joint_state", "/body_drive/pelvis_imu/data", "/body_drive/torso_imu/data",
        ],
        "command_topics": [
            "/body_drive/waist_joint_command", "/body_drive/leg_joint_command", "/body_drive/arm_joint_command", "/body_drive/neck_joint_command",
        ],
        "joints": joints,
        "limitations": [
            "Fixed stand is diagnostic only; it is not a feedback-balance or real-robot deployment contract.",
            "A new contract and complete requalification are required for native MOTION ownership or a whole-body balance policy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = build_contract()
    validate_executor_contract(contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
