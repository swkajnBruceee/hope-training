#!/usr/bin/env python3
"""Package successful whole-space gap-fill targets for coverage audits.

The manifest deliberately records raw 10-DOF candidates as not yet admitted
to training; 31-DOF FK materialization and floating-base qualification remain
separate gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--physx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    goals = {str(x["goal_id"]): x for x in json.loads(args.goals.read_text(encoding="utf-8"))["goals"]}
    generated = {str(x["goal_id"]): x for x in json.loads(args.generation.read_text(encoding="utf-8"))["results"]}
    physx = json.loads(args.physx.read_text(encoding="utf-8"))
    physx_by_path = {str(x["motion_file"]): x for x in physx["rows"]}
    motions = []
    for goal_id, row in sorted(generated.items()):
        if row.get("returncode") != 0 or row.get("npz_status") != "ok":
            continue
        goal = goals[goal_id]
        trajectory = str((Path(row["output_dir"]).expanduser().resolve() / "trajectory_100hz.npz").resolve())
        audit = physx_by_path[trajectory]
        status = str(audit["status"])
        role = "core" if status == "FIXED_BASE_PHYSX_REPLAY_PASS" else "boundary"
        motions.append({
            "episode_id": f"gap_fill_{goal_id}",
            "motion_id": f"gap_fill_{goal_id}",
            "stroke_type": goal["swing_type"],
            "canonical_goal_10d": {
                "position_m": goal["position_m"],
                "normal_w": goal["racket_normal"],
                "linear_velocity_mps": goal["linear_velocity_mps"],
                "time_to_hit_s": goal["time_to_strike_s"],
            },
            "source_goal_id": goal_id,
            "source_npz": trajectory,
            "fixed_base_physx_status": status,
            "dataset_role": role,
            "sample_weight": 1.0 if role == "core" else 0.25,
            "physics_qualified": False,
            "teacher_approved": False,
            "training_admission": False,
            "fk_materialized": False,
        })
    payload = {
        "schema_version": "a3_gap_fill_screened_manifest/v1",
        "status": "raw_10d_screened_fk_pending",
        "motions": motions,
        "counts": {
            "completed": len(motions),
            "core": sum(x["dataset_role"] == "core" for x in motions),
            "boundary": sum(x["dataset_role"] == "boundary" for x in motions),
        },
        "qualification_notice": "Not admitted to training; fixed-base audit only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **payload["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
