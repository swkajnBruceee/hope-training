#!/usr/bin/env python3
"""Build a weighted index for the backhand seed expansion after PhysX."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-manifest", type=Path, required=True)
    parser.add_argument("--generation-summary", type=Path, required=True)
    parser.add_argument("--physx-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    goal_manifest_path = args.goal_manifest.expanduser().resolve()
    generation_path = args.generation_summary.expanduser().resolve()
    physx_path = args.physx_audit.expanduser().resolve()
    goal_manifest = json.loads(goal_manifest_path.read_text(encoding="utf-8"))
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    physx = json.loads(physx_path.read_text(encoding="utf-8"))
    goals = {str(g["goal_id"]): g for g in goal_manifest["goals"]}
    generated = {str(row["goal_id"]): row for row in generation["results"]}
    physx_by_file = {str(Path(row["motion_file"]).resolve()): row for row in physx["rows"]}

    rows: list[dict] = []
    rejected: list[dict] = []
    for goal_id, goal in goals.items():
        generated_row = generated[goal_id]
        out_dir = Path(str(generated_row["output_dir"])).expanduser().resolve()
        trajectory = (out_dir / "trajectory_100hz.npz").resolve()
        physx_row = physx_by_file.get(str(trajectory))
        if generated_row.get("returncode") != 0 or generated_row.get("npz_status") != "ok":
            rejected.append({"goal_id": goal_id, "reason": "ik_generation_rejected", "status": generated_row.get("status")})
            continue
        if physx_row is None:
            rejected.append({"goal_id": goal_id, "reason": "missing_physx_row"})
            continue
        status = str(physx_row["status"])
        if status == "FIXED_BASE_PHYSX_REPLAY_PASS":
            role, weight = "core", 1.0
        elif status == "FIXED_BASE_PHYSX_SOFT_LIMIT_WARNING":
            role, weight = "boundary", 0.25
        else:
            rejected.append({"goal_id": goal_id, "reason": "physx_rejected", "status": status})
            continue
        rows.append(
            {
                "stroke": "backhand",
                "goal_id": goal_id,
                "seed_episode_id": goal["seed_episode_id"],
                "variant_index": goal["variant_index"],
                "split": goal["split"],
                "trajectory_npz": str(trajectory),
                "x_m": goal["position_m"][0],
                "y_m": goal["position_m"][1],
                "z_m": goal["position_m"][2],
                "linear_velocity_mps": json.dumps(goal["linear_velocity_mps"]),
                "racket_normal": json.dumps(goal["racket_normal"]),
                "strike_time_s": goal["time_to_strike_s"],
                "position_error_m": generated_row.get("position_error_m", ""),
                "normal_error_deg": generated_row.get("normal_error_deg", ""),
                "fixed_base_physx_status": status,
                "dataset_role": role,
                "sample_weight": weight,
                "screening_reason": "backhand_seed_expansion_fixed_base_physx_pass" if role == "core" else "backhand_seed_expansion_soft_limit_boundary",
                "physics_qualified": False,
            }
        )
    rows.sort(key=lambda row: str(row["goal_id"]))
    if not rows:
        raise ValueError("no eligible expansion candidates")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit_path = output.with_name("expansion_index_audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "a3_backhand_seed_expansion_index_audit/v1",
                "goal_manifest": str(goal_manifest_path),
                "generation_summary": str(generation_path),
                "physx_audit": str(physx_path),
                "goal_count": len(goals),
                "eligible_count": len(rows),
                "core_count": sum(r["dataset_role"] == "core" for r in rows),
                "boundary_count": sum(r["dataset_role"] == "boundary" for r in rows),
                "rejected_count": len(rejected),
                "rejected": rejected,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "eligible": len(rows), "core": sum(r["dataset_role"] == "core" for r in rows), "boundary": sum(r["dataset_role"] == "boundary" for r in rows), "rejected": len(rejected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
