#!/usr/bin/env python3
"""Materialize a weighted-index CSV for the target-conditioned IK pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--generation-summary", type=Path, required=True)
    parser.add_argument("--physx-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot = json.loads(args.pilot_manifest.expanduser().read_text(encoding="utf-8"))
    generated = json.loads(args.generation_summary.expanduser().read_text(encoding="utf-8"))
    physx = json.loads(args.physx_audit.expanduser().read_text(encoding="utf-8"))
    goals = {str(g["goal_id"]): g for g in pilot["goals"]}
    rows_by_goal = {str(row["goal_id"]): row for row in generated["results"]}
    physx_by_path = {str(row["motion_file"]): row for row in physx["rows"]}

    rows: list[dict[str, object]] = []
    for goal_id, goal in goals.items():
        generated_row = rows_by_goal[goal_id]
        if generated_row.get("returncode") != 0 or generated_row.get("npz_status") != "ok":
            continue
        output_dir = Path(str(generated_row["output_dir"])).expanduser().resolve()
        trajectory = (output_dir / "trajectory_100hz.npz").resolve()
        audit = physx_by_path.get(str(trajectory))
        if audit is None:
            raise ValueError(f"missing PhysX row for generated trajectory: {trajectory}")
        status = str(audit["status"])
        if status == "FIXED_BASE_PHYSX_REPLAY_PASS":
            role, weight = "core", 1.0
            reason = "target_conditioned_pilot_fixed_base_pass"
        elif status == "FIXED_BASE_PHYSX_SOFT_LIMIT_WARNING":
            role, weight = "boundary", 0.25
            reason = "target_conditioned_pilot_soft_limit_warning"
        else:
            raise ValueError(f"pilot trajectory is not eligible for the pilot bank: {trajectory}: {status}")
        rows.append(
            {
                "stroke": goal["swing_type"],
                "goal_id": goal_id,
                "trajectory_npz": str(trajectory),
                "x_m": goal["position_m"][0],
                "y_m": goal["position_m"][1],
                "z_m": goal["position_m"][2],
                "pitch_deg": goal["pitch_deg"],
                "yaw_deg": goal["yaw_deg"],
                "strike_time_s": goal["time_to_strike_s"],
                "position_error_m": generated_row.get("position_error_m", ""),
                "normal_error_deg": generated_row.get("normal_error_deg", ""),
                "velocity_error_mps": generated_row.get("velocity_error_mps", ""),
                "minimum_clearance_m": "",
                "quality_score": "",
                "fixed_base_physx_status": status,
                "dataset_role": role,
                "sample_weight": weight,
                "screening_reason": reason,
                "physics_qualified": False,
            }
        )

    if not rows:
        raise ValueError("pilot produced no eligible trajectories")
    rows.sort(key=lambda row: str(row["goal_id"]))
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with args.output.expanduser().open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output.expanduser().resolve()), "rows": len(rows), "core": sum(r["dataset_role"] == "core" for r in rows), "boundary": sum(r["dataset_role"] == "boundary" for r in rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
