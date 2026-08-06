#!/usr/bin/env python3
"""Freeze a safety-aware reoptimization queue; never relabels canonical goals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--split-report", required=True)
    ap.add_argument("--replay-summary", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    bank_path = Path(args.bank).resolve()
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    split = json.loads(Path(args.split_report).resolve().read_text(encoding="utf-8"))
    replay = json.loads(Path(args.replay_summary).resolve().read_text(encoding="utf-8"))
    projected = set(split["projected_episode_ids"])
    replay_rows = {row["episode_id"]: row for row in replay.get("rows", [])}
    queue = []
    for item in bank["motions"]:
        eid = item["episode_id"]
        if eid not in projected:
            continue
        row = replay_rows.get(eid, {})
        queue.append(
            {
                "episode_id": eid,
                "source_motion_npz": item["motion_npz"],
                "category": item.get("p5d2_bank", {}).get("category"),
                "split": item.get("p5d2_bank", {}).get("split"),
                "source_seed_motion_id": item.get("p5d2_bank", {}).get("source_seed_motion_id"),
                "canonical_goal_10d": item.get("canonical_goal_10d"),
                "hit_frame": item.get("reference_contract", {}).get("hit_frame"),
                "safety_projection_max_rad": row.get("safety_projection_max_rad"),
                "safety_projection_threshold_rad": split["safety_projection_threshold_rad"],
                "reoptimization_status": "QUEUED_SAFETY_AWARE_OFFLINE_REOPTIMIZATION",
                "required_objective": [
                    "use_deployed_runtime_safety_filter_or_exact_equivalent",
                    "preserve_canonical_goal_10d_and_tcp_frame_contract",
                    "preserve_hit_time_and_complete_recovery_segment",
                    "reduce_reference_to_processed_command_projection",
                    "recheck_soft_limits_collision_velocity_acceleration_jerk",
                ],
                "actual_tracking_error_is_not_rejection": True,
                "training_started": False,
            }
        )
    out = {
        "schema_version": "p5d2_safety_reoptimization_queue/v1",
        "source_bank": str(bank_path),
        "source_split_report": str(Path(args.split_report).resolve()),
        "source_replay_summary": str(Path(args.replay_summary).resolve()),
        "queue_role": "safety_filter_reoptimization_only",
        "canonical_goal_relabeling": False,
        "actual_tracking_error_used_as_rejection": False,
        "training_started": False,
        "count": len(queue),
        "items": queue,
    }
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_path), "count": len(queue), "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
