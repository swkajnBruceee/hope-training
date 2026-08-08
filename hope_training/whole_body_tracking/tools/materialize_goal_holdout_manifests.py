#!/usr/bin/env python3
"""Materialize train/validation manifests from an explicit goal holdout."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--goal-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    goal_path = args.goal_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    goals = json.loads(goal_path.read_text(encoding="utf-8"))
    split_by_goal = {
        str(goal["goal_id"]): str(goal["split"])
        for goal in goals.get("goals", [])
    }
    if not split_by_goal:
        raise ValueError("goal manifest has no goals")

    split_entries: dict[str, list[dict]] = {"training": [], "validation": []}
    for entry in source.get("motions", []):
        goal_id = str(entry.get("source_goal_id", ""))
        split = split_by_goal.get(goal_id)
        if split not in split_entries:
            raise ValueError(f"missing or invalid split for source_goal_id={goal_id!r}")
        copied = copy.deepcopy(entry)
        copied["split"] = split
        copied["split_group_id"] = goal_id
        copied["split_group_key"] = goal_id
        copied["split_contract"] = "explicit_target_tuple_holdout"
        split_entries[split].append(copied)

    all_ids = {str(e["source_goal_id"]) for e in source.get("motions", [])}
    if not all_ids.issubset(split_by_goal):
        raise ValueError("source manifest contains an unrecognized goal")
    overlap = {
        str(e["source_goal_id"]) for e in split_entries["training"]
    } & {
        str(e["source_goal_id"]) for e in split_entries["validation"]
    }
    if overlap:
        raise RuntimeError(f"goal leakage detected: {sorted(overlap)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": "a3_target_conditioned_goal_holdout/v1",
        "status": "candidate_reference_pending_training_split",
        "source_manifest": str(source_path),
        "goal_manifest": str(goal_path),
        "source_index": source.get("source_index"),
        "notice_contract": "THIRD_PARTY_NOTICES.md",
        "reference_semantics": source.get("reference_semantics"),
        "physics_qualified": False,
        "teacher_approved": False,
        "training_admission": False,
        "floating_base_replay_done": False,
        "self_collision_observable": False,
        "weights": source.get("weights", {"core": 1.0, "boundary": 0.25}),
        "split_contract": "explicit_target_tuple_holdout",
    }
    report = {"source_manifest": str(source_path), "goal_manifest": str(goal_path), "status": "completed"}
    for split, entries in split_entries.items():
        payload = dict(common)
        payload["split"] = split
        payload["motion_count"] = len(entries)
        payload["group_count"] = len({str(e["source_goal_id"]) for e in entries})
        payload["counts"] = {
            "motions": len(entries),
            "core": sum(e.get("dataset_role") == "core" for e in entries),
            "boundary": sum(e.get("dataset_role") == "boundary" for e in entries),
            "backhand": sum(e.get("stroke_type") == "backhand" for e in entries),
            "forehand": sum(e.get("stroke_type") == "forehand" for e in entries),
        }
        payload["motions"] = entries
        (output_dir / f"{split}_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    report["counts"] = {
        split: {
            "motions": len(entries),
            "goals": len({str(e["source_goal_id"]) for e in entries}),
            "by_stroke": dict(Counter(str(e["stroke_type"]) for e in entries)),
        }
        for split, entries in split_entries.items()
    }
    (output_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
