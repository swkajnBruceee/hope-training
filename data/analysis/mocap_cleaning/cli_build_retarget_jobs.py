#!/usr/bin/env python3
"""Build retarget job manifests from retarget-ready sample manifests."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    del _ROOT

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_name(text: str) -> str:
    return text.replace("/", "_")


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _a3_constraint_profile() -> dict[str, Any]:
    return {
        "priority_order": [
            "racket_pose_at_hit",
            "racket_velocity_direction_at_hit",
            "swing_timing_alignment",
            "upper_body_kinematic_consistency",
            "torso_support_motion",
            "lower_body_stability",
        ],
        "constraints": {
            "hard": [
                "a3_joint_order",
                "joint_limits",
                "retarget_csv_schema",
                "motion_npz_schema",
                "position_unit_m",
                "fps_consistency",
                "finite_values_only",
            ],
            "strong_soft": [
                "racket_center_position",
                "racket_face_normal",
                "racket_velocity_direction",
                "hit_frame_alignment",
            ],
            "medium_soft": [
                "right_wrist_trajectory",
                "right_elbow_trajectory",
                "right_shoulder_trajectory",
                "torso_yaw_and_lean",
            ],
            "weak_soft": [
                "pelvis_tracking",
                "leg_tracking",
                "head_orientation",
                "global_motion_naturalness",
            ],
        },
        "phase_weights": {
            "pre_hit": {
                "frame_window": [-24, -4],
                "weights": {
                    "racket_pose": 2.0,
                    "racket_velocity": 2.5,
                    "arm_posture": 1.0,
                    "torso_support": 1.2,
                    "leg_tracking": 0.3,
                },
            },
            "hit": {
                "frame_window": [-3, 3],
                "weights": {
                    "racket_pose": 5.0,
                    "racket_velocity": 5.0,
                    "arm_posture": 1.2,
                    "torso_support": 1.5,
                    "leg_tracking": 0.2,
                },
            },
            "post_hit": {
                "frame_window": [4, 20],
                "weights": {
                    "racket_pose": 1.5,
                    "racket_velocity": 1.0,
                    "arm_posture": 1.0,
                    "torso_support": 1.0,
                    "leg_tracking": 0.3,
                },
            },
        },
        "refinement_policy": {
            "generic_retarget_output_used_as_init": True,
            "run_upper_body_and_torso_refinement": True,
            "allow_small_torso_motion": True,
            "lock_legs_first_pass": True,
            "post_refinement_limit_enforcement": True,
        },
        "racket_mount_transforms": {
            "human_wrist_to_racket_center": {
                "status": "todo_calibrate",
                "translation_m": [None, None, None],
                "quat_xyzw": [None, None, None, None],
            },
            "a3_ee_to_racket_center": {
                "status": "todo_calibrate",
                "translation_m": [None, None, None],
                "quat_xyzw": [None, None, None, None],
            },
        },
        "quality_metrics": [
            "racket_position_error_at_hit_m",
            "racket_orientation_error_at_hit_deg",
            "racket_velocity_direction_error_at_hit_deg",
            "max_joint_limit_violation_before_clamp_rad",
            "max_joint_velocity_radps",
            "max_joint_acceleration_radps2",
            "ik_residual_rms",
            "csv_to_npz_passed",
        ],
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DATA260703 Retarget Jobs",
        "",
        f"- robot: `{report['robot']}`",
        f"- source manifest: `{report['source_manifest']}`",
        f"- jobs: `{report['job_count']}`",
        "",
        "## Label Counts",
        "",
        "| label | count |",
        "|---|---:|",
    ]
    for label, count in sorted(report["label_counts"].items()):
        lines.append(f"| {label} | {count} |")

    lines.extend(["", "## Source CSV Counts", "", "| source | count |", "|---|---:|"])
    for source, count in sorted(report["source_csv_counts"].items()):
        lines.append(f"| {source} | {count} |")

    lines.extend(["", "## Outputs", ""])
    lines.append(f"- `jobs_manifest_json`: `{report['jobs_manifest_json']}`")
    lines.append(f"- `jobs_summary_md`: `{report['jobs_summary_md']}`")
    lines.append(f"- `jobs_root`: `{report['jobs_root']}`")

    lines.extend(["", "## Refinement", ""])
    lines.append("- Priority is racket-first: pose, face normal, velocity direction, then human-like arm/torso motion.")
    lines.append("- The expected pipeline is generic retarget init -> A3 constrained refinement -> validation -> csv_to_npz.")

    lines.extend(["", "## Notes", ""])
    lines.append("- Each job declares the input BVH, source clean sample, target retarget CSV path, and target motion NPZ path.")
    lines.append("- Job status is initialized as `pending`; this script defines the refinement contract but does not solve A3 IK yet.")
    lines.append("- This is the handoff point between dataset curation and robot-specific retarget implementation.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_ready/retarget_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3"),
    )
    parser.add_argument("--robot", default="agibot_a3")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    samples = manifest["samples"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    constraint_profile = _a3_constraint_profile()

    jobs = []
    label_counts = Counter()
    source_csv_counts = Counter()
    for sample in samples:
        label = str(sample["label"])
        episode_id = str(sample["episode_id"])
        source_csv = str(sample["source_csv"])
        source_csv_key = _safe_name(source_csv.replace("Csv/", "").replace(".csv", ""))
        label_dir = args.output_dir / label
        init_dir = label_dir / "generic_init_csv"
        csv_dir = label_dir / "retarget_csv"
        npz_dir = label_dir / "motion_npz"
        init_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        npz_dir.mkdir(parents=True, exist_ok=True)

        generic_retarget_csv = init_dir / f"{episode_id}.csv"
        retarget_csv = csv_dir / f"{episode_id}.csv"
        motion_npz = npz_dir / f"{episode_id}.npz"
        job = {
            "job_id": f"{args.robot}__{label}__{episode_id}",
            "status": "pending",
            "robot": args.robot,
            "label": label,
            "confidence": float(sample["confidence"]),
            "episode_id": episode_id,
            "source_csv": source_csv,
            "source_csv_group": source_csv_key,
            "source_bvh": str(sample["source_bvh"]),
            "source_sample_npz": str(sample["sample_path"]),
            "source_clean_npz": str(sample["source_npz"]),
            "source_debug_npz": str(sample["debug_npz"]),
            "racket": str(sample["racket"]),
            "candidate": str(sample["candidate"]),
            "handedness": str(sample["handedness"]),
            "hit_time": float(sample["hit_time"]),
            "hit_index": int(sample["hit_index"]),
            "frames": int(sample["frames"]),
            "fps": float(sample["fps"]),
            "generic_retarget_csv": str(generic_retarget_csv),
            "retarget_csv": str(retarget_csv),
            "motion_npz": str(motion_npz),
            "pipeline": [
                "generic_retarget",
                "a3_joint_mapping",
                "a3_constrained_refinement",
                "limit_velocity_smoothing_validation",
                "csv_to_npz",
            ],
            "constraint_profile": constraint_profile,
            "motion_contract": {
                "required_fields": [
                    "fps",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ],
                "csv_to_npz_robot": args.robot,
            },
            "refinement_inputs": {
                "human_skeleton_source": str(sample["source_bvh"]),
                "clean_sample_npz": str(sample["sample_path"]),
                "racket_reference_source": "clean_sample_npz",
                "hit_frame_index": int(sample["hit_index"]),
                "label": label,
            },
            "refinement_outputs": {
                "retarget_csv": str(retarget_csv),
                "motion_npz": str(motion_npz),
                "quality_report_json": str(label_dir / "quality_reports" / f"{episode_id}.json"),
            },
            "quality_template": {
                metric: None for metric in constraint_profile["quality_metrics"]
            },
            "notes": [
                "Run human-to-A3 retarget on source_bvh first.",
                "Use the generic retarget output as the initialization, not the final answer.",
                "Refine around the hit window with racket-first constraints before csv_to_npz conversion.",
                "Then convert retarget_csv to motion_npz via hope_training/whole_body_tracking/scripts/csv_to_npz.py.",
            ],
        }
        jobs.append(job)
        label_counts[label] += 1
        source_csv_counts[source_csv] += 1

    jobs.sort(key=lambda item: (item["label"], -item["confidence"], item["source_csv"], item["episode_id"]))
    jobs_manifest = {
        "robot": args.robot,
        "source_manifest": str(args.manifest),
        "job_count": len(jobs),
        "jobs_root": str(args.output_dir),
        "constraint_profile": constraint_profile,
        "jobs": jobs,
    }
    jobs_manifest_json = args.output_dir / "jobs_manifest.json"
    _write_json(jobs_manifest_json, jobs_manifest)

    report = {
        "robot": args.robot,
        "source_manifest": str(args.manifest),
        "job_count": len(jobs),
        "label_counts": dict(label_counts),
        "source_csv_counts": dict(source_csv_counts),
        "jobs_manifest_json": str(jobs_manifest_json),
        "jobs_root": str(args.output_dir),
    }
    summary_md = args.output_dir / "jobs_summary.md"
    report["jobs_summary_md"] = str(summary_md)
    _write_markdown(summary_md, report)

    print(f"Prepared {len(jobs)} retarget jobs")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")
    print(f"Wrote {jobs_manifest_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
