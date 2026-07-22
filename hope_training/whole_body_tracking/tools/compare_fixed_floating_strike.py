#!/usr/bin/env python3
"""Compare fixed-teacher and floating-Base realized joints at strike time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STRIKE_JOINTS = (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-bank", type=Path, required=True)
    parser.add_argument("--floating-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixed_manifest = json.loads(
        args.fixed_bank.joinpath("rsi_capture_manifest.json").read_text(encoding="utf-8")
    )
    floating_manifest = json.loads(
        args.floating_bank.joinpath("rsi_capture_manifest.json").read_text(encoding="utf-8")
    )
    fixed_by_episode = {entry["episode_id"]: entry for entry in fixed_manifest["entries"]}
    names = list(floating_manifest["joint_names"])
    ids = [names.index(name) for name in STRIKE_JOINTS]
    actual_differences = []
    target_differences = []
    rows = []
    for floating_entry in floating_manifest["entries"]:
        episode_id = floating_entry["episode_id"]
        fixed_entry = fixed_by_episode[episode_id]
        with np.load(args.floating_bank / floating_entry["state_file"], allow_pickle=False) as floating:
            with np.load(args.fixed_bank / fixed_entry["state_file"], allow_pickle=False) as fixed:
                floating_hit = int(np.argmin(np.abs(floating["time_to_strike_s"])))
                fixed_hit = int(np.argmin(np.abs(fixed["time_to_strike_s"])))
                actual_delta = floating["joint_pos"][floating_hit, ids] - fixed["joint_pos"][fixed_hit, ids]
                target_delta = (
                    floating["joint_pos_target"][floating_hit, ids]
                    - fixed["joint_pos_target"][fixed_hit, ids]
                )
                actual_differences.append(actual_delta)
                target_differences.append(target_delta)
                rows.append(
                    {
                        "episode_id": episode_id,
                        "actual_joint_delta_rad": {
                            name: float(value) for name, value in zip(STRIKE_JOINTS, actual_delta)
                        },
                        "target_joint_delta_rad": {
                            name: float(value) for name, value in zip(STRIKE_JOINTS, target_delta)
                        },
                    }
                )

    actual = np.asarray(actual_differences)
    target = np.asarray(target_differences)
    per_joint = {
        name: {
            "actual_delta_mean_rad": float(np.mean(actual[:, index])),
            "actual_delta_rmse_rad": float(np.sqrt(np.mean(np.square(actual[:, index])))),
            "target_delta_abs_max_rad": float(np.max(np.abs(target[:, index]))),
        }
        for index, name in enumerate(STRIKE_JOINTS)
    }
    dominant_joint = max(per_joint, key=lambda name: per_joint[name]["actual_delta_rmse_rad"])
    report = {
        "schema_version": 1,
        "stage": "fixed_vs_floating_strike_contract_comparison_v1",
        "training_eligible": False,
        "motions": len(rows),
        "all_reference_targets_identical": bool(np.max(np.abs(target)) <= 1.0e-7),
        "all_strike_joint_actual_delta_rmse_rad": float(np.sqrt(np.mean(np.square(actual)))),
        "dominant_actual_difference_joint": dominant_joint,
        "per_joint": per_joint,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "motions": report["motions"],
                "targets_identical": report["all_reference_targets_identical"],
                "dominant_joint": dominant_joint,
                "dominant_rmse_rad": per_joint[dominant_joint]["actual_delta_rmse_rad"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
