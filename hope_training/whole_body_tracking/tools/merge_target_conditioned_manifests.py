#!/usr/bin/env python3
"""Merge independently audited target-conditioned banks without copying data."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", action="append", required=True)
    parser.add_argument("--goal-manifest", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.source_manifest) != len(args.goal_manifest):
        raise ValueError("source and goal manifest counts must match")

    merged: list[dict] = []
    seen_goals: set[str] = set()
    for source_value, goal_value in zip(args.source_manifest, args.goal_manifest):
        source_path = Path(source_value).expanduser().resolve()
        goal_path = Path(goal_value).expanduser().resolve()
        source = json.loads(source_path.read_text(encoding="utf-8"))
        goal_manifest = json.loads(goal_path.read_text(encoding="utf-8"))
        split_by_goal = {str(g["goal_id"]): str(g["split"]) for g in goal_manifest.get("goals", [])}
        for entry in source.get("motions", []):
            goal_id = str(entry["source_goal_id"])
            if goal_id in seen_goals:
                raise ValueError(f"duplicate target goal: {goal_id}")
            split = split_by_goal.get(goal_id)
            if split not in ("training", "validation"):
                raise ValueError(f"missing explicit split for target goal: {goal_id}")
            copied = copy.deepcopy(entry)
            motion_path = Path(str(copied["motion_npz"]))
            if not motion_path.is_absolute():
                motion_path = source_path.parent / motion_path
            copied["motion_npz"] = str(motion_path.resolve())
            copied["library_motion_npz"] = str(motion_path.resolve())
            copied["source_bank_manifest"] = str(source_path)
            copied["split"] = split
            copied["split_group_id"] = goal_id
            copied["split_contract"] = "explicit_target_tuple_holdout"
            merged.append(copied)
            seen_goals.add(goal_id)

    merged.sort(key=lambda entry: (str(entry["split"]), str(entry["stroke_type"]), str(entry["source_goal_id"])))
    for index, entry in enumerate(merged):
        entry["episode_id"] = f"candidate_{index:05d}"
        entry["motion_id"] = entry["episode_id"]
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "a3_target_conditioned_merged_bank/v1",
        "status": "candidate_reference_pending_training_split",
        "source_manifests": [str(Path(x).expanduser().resolve()) for x in args.source_manifest],
        "goal_manifests": [str(Path(x).expanduser().resolve()) for x in args.goal_manifest],
        "notice_contract": "THIRD_PARTY_NOTICES.md",
        "reference_semantics": "fixed-base FK-expanded offline IK candidate reference",
        "coordinate_contract": "current_root_relative_initial_heading",
        "root_pose_contract": {
            "root_position_w_m": [0.0, 0.0, 1.0684],
            "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "tcp_contract": {
            "body": "right_wrist_yaw_Link",
            "mount_offset_local_m": [0.210211399202899, 0.0320784994676765, 0.0320358706296689],
            "normal_axis": "+Y",
        },
        "waist_contract": {
            "waist_pitch": "forward_only_nonnegative_joint_pitch",
            "backward_tilt_allowed": False,
            "forward_tilt_limit_deg": 20.0,
            "waist_roll_abs_limit_deg": 20.0,
        },
        "physics_qualified": False,
        "teacher_approved": False,
        "training_admission": False,
        "floating_base_replay_done": False,
        "self_collision_observable": False,
        "weights": {"core": 1.0, "boundary": 0.25},
        "counts": {
            "completed": len(merged),
            "core": sum(e.get("dataset_role") == "core" for e in merged),
            "boundary": sum(e.get("dataset_role") == "boundary" for e in merged),
            "training": sum(e.get("split") == "training" for e in merged),
            "validation": sum(e.get("split") == "validation" for e in merged),
            "backhand": sum(e.get("stroke_type") == "backhand" for e in merged),
            "forehand": sum(e.get("stroke_type") == "forehand" for e in merged),
        },
        "split_contract": "explicit_target_tuple_holdout",
        "motions": merged,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split in ("training", "validation"):
        split_payload = copy.deepcopy(payload)
        split_payload["split"] = split
        split_payload["motions"] = [entry for entry in merged if entry["split"] == split]
        split_payload["counts"] = {
            **payload["counts"],
            "completed": len(split_payload["motions"]),
            "core": sum(e.get("dataset_role") == "core" for e in split_payload["motions"]),
            "boundary": sum(e.get("dataset_role") == "boundary" for e in split_payload["motions"]),
            "training": len(split_payload["motions"]) if split == "training" else 0,
            "validation": len(split_payload["motions"]) if split == "validation" else 0,
            "backhand": sum(e.get("stroke_type") == "backhand" for e in split_payload["motions"]),
            "forehand": sum(e.get("stroke_type") == "forehand" for e in split_payload["motions"]),
        }
        split_payload["motion_count"] = len(split_payload["motions"])
        (output_path.parent / f"{split}_manifest.json").write_text(
            json.dumps(split_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
