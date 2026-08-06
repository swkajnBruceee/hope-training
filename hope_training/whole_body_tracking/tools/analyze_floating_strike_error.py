#!/usr/bin/env python3
"""Decompose floating-base strike error into root-pose and internal tracking terms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rotate(q_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q_vec = q_wxyz[1:]
    t = 2.0 * np.cross(q_vec, vector)
    return vector + q_wxyz[0] * t + np.cross(q_vec, t)


def _rotate_inverse(q_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    inverse = q_wxyz.copy()
    inverse[1:] *= -1.0
    return _rotate(inverse, vector)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1.0e-12))
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.bank.joinpath("rsi_capture_manifest.json").read_text(encoding="utf-8"))
    joint_names = list(manifest["joint_names"])
    action_ids = [joint_names.index(name) for name in manifest["action_joint_names"]]
    strike_ids = [
        index
        for index, name in enumerate(joint_names)
        if name in {
            "waist_yaw_joint",
            "waist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        }
    ]
    rows = []
    for entry in manifest["entries"]:
        with np.load(args.bank / entry["state_file"], allow_pickle=False) as data:
            hit = int(np.argmin(np.abs(data["time_to_strike_s"])))
            root0 = data["root_state_w"][0]
            root = data["root_state_w"][hit]
            racket = data["racket_pos_w"][hit]
            target = data["racket_target_pos_w"][hit]
            normal = data["racket_normal_w"][hit]
            target_normal = data["racket_target_normal_w"][hit]

            desired_pos_rel0 = _rotate_inverse(root0[3:7], target - root0[:3])
            actual_pos_rel = _rotate_inverse(root[3:7], racket - root[:3])
            root_only_racket = root[:3] + _rotate(root[3:7], desired_pos_rel0)
            desired_normal_rel0 = _rotate_inverse(root0[3:7], target_normal)
            actual_normal_rel = _rotate_inverse(root[3:7], normal)
            root_only_normal = _rotate(root[3:7], desired_normal_rel0)

            joint_error = data["joint_pos"][hit] - data["joint_pos_target"][hit]
            rows.append(
                {
                    "episode_id": entry["episode_id"],
                    "motion_step": int(data["motion_step"][hit]),
                    "time_to_strike_s": float(data["time_to_strike_s"][hit]),
                    "world_position_error_m": float(np.linalg.norm(racket - target)),
                    "world_velocity_error_mps": float(
                        np.linalg.norm(data["racket_lin_vel_w"][hit] - data["racket_target_vel_w"][hit])
                    ),
                    "world_normal_error_deg": _angle_deg(normal, target_normal),
                    "root_translation_from_start_m": float(np.linalg.norm(root[:3] - root0[:3])),
                    "root_only_position_error_m": float(np.linalg.norm(root_only_racket - target)),
                    "internal_relative_position_error_m": float(
                        np.linalg.norm(actual_pos_rel - desired_pos_rel0)
                    ),
                    "root_only_normal_error_deg": _angle_deg(root_only_normal, target_normal),
                    "internal_relative_normal_error_deg": _angle_deg(actual_normal_rel, desired_normal_rel0),
                    "base14_joint_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(joint_error[action_ids])))),
                    "strike_joint_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(joint_error[strike_ids])))),
                    "root_linear_velocity_mps": float(np.linalg.norm(root[7:10])),
                    "root_angular_velocity_rad_s": float(np.linalg.norm(root[10:13])),
                }
            )

    numeric_keys = [key for key in rows[0] if key not in {"episode_id", "motion_step"}]
    summary = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "min": float(np.min([row[key] for row in rows])),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in numeric_keys
    }
    report = {
        "schema_version": 1,
        "stage": "floating_base_strike_error_decomposition_v1",
        "training_eligible": False,
        "interpretation": (
            "root_only assumes the desired racket pose is rigidly carried by root motion; "
            "internal_relative compares the realized root-relative racket pose with the frame-zero target contract"
        ),
        "rows": rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "motions": len(rows),
                "position_error_mean_m": summary["world_position_error_m"]["mean"],
                "root_only_position_error_mean_m": summary["root_only_position_error_m"]["mean"],
                "internal_relative_position_error_mean_m": summary["internal_relative_position_error_m"]["mean"],
                "strike_joint_tracking_rmse_mean_rad": summary["strike_joint_tracking_rmse_rad"]["mean"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
