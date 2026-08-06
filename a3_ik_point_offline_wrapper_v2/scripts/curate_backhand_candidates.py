#!/usr/bin/env python3
"""Filter and deduplicate backhand candidate diagnostics without deleting data."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"missing/non-finite diagnostic field: {key}")
    return float(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "goal_id",
        "source_output_dir",
        "x_m",
        "y_m",
        "z_m",
        "pitch_deg",
        "yaw_deg",
        "strike_time_s",
        "position_error_m",
        "normal_error_deg",
        "velocity_error_mps",
        "minimum_clearance_m",
        "quality_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--swing-type", choices=("forehand", "backhand"), default="backhand")
    parser.add_argument("--max-position-error-m", type=float, default=0.001)
    parser.add_argument("--max-normal-error-deg", type=float, default=0.2)
    parser.add_argument("--max-velocity-error-mps", type=float, default=0.05)
    parser.add_argument("--min-clearance-m", type=float, default=0.02)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    with args.summary.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("success") != "True" or row.get("status") != "KINEMATIC_CANDIDATE":
                continue
            source = Path(row["output_dir"])
            diagnostics = json.loads((source / "diagnostics.json").read_text(encoding="utf-8"))
            if (
                diagnostics.get("requested_swing_type") != args.swing_type
                or diagnostics.get("selected_swing_type") != args.swing_type
                or diagnostics.get("ready_goal_contract_match") is not True
                or diagnostics.get("trajectory_reject_reason") != "none"
                or diagnostics.get("timing_extension_s") != 0
                or any(diagnostics.get(k, 0) != 0 for k in (
                    "rejected_target_count",
                    "solve_reject_count",
                    "velocity_reject_count",
                    "follow_reject_count",
                    "trajectory_reject_count",
                ))
            ):
                continue
            position_error = number(diagnostics, "solved_position_error_m")
            normal_error = number(diagnostics, "solved_normal_error_deg")
            velocity_error = number(diagnostics, "velocity_solve_error_mps")
            clearance = number(diagnostics, "minimum_body_clearance_m")
            if position_error > args.max_position_error_m:
                continue
            if normal_error > args.max_normal_error_deg:
                continue
            if velocity_error > args.max_velocity_error_mps:
                continue
            if clearance < args.min_clearance_m:
                continue
            position = tuple(round(float(value), 8) for value in diagnostics["target_position_m"])
            target_velocity = diagnostics["target_velocity_mps"]
            normal = diagnostics["solved_joint_position_rad"]
            # Use the goal's target normal for the report; diagnostics keeps the
            # authoritative target position and the source directory.
            goal = json.loads((source / "normalized_goal.json").read_text(encoding="utf-8"))
            target_normal = goal["racket_normal"]
            horizontal_normal = math.hypot(float(target_normal[0]), float(target_normal[1]))
            if args.swing_type == "forehand":
                pitch_deg = math.degrees(math.atan2(float(target_normal[2]), horizontal_normal))
                yaw_deg = math.degrees(math.atan2(float(target_normal[1]), float(target_normal[0])))
            else:
                pitch_deg = math.degrees(math.atan2(float(target_normal[2]), horizontal_normal))
                yaw_deg = math.degrees(math.atan2(-float(target_normal[1]), -float(target_normal[0])))
            score = (
                position_error / args.max_position_error_m
                + normal_error / args.max_normal_error_deg
                + velocity_error / args.max_velocity_error_mps
                - clearance / max(args.min_clearance_m, 1e-9)
            )
            accepted.append({
                "goal_id": row["goal_id"],
                "source_output_dir": str(source.resolve()),
                "position": position,
                "pitch_deg": round(pitch_deg, 6),
                "yaw_deg": round(yaw_deg, 6),
                "strike_time_s": number(diagnostics, "planned_strike_time_s"),
                "position_error_m": position_error,
                "normal_error_deg": normal_error,
                "velocity_error_mps": velocity_error,
                "minimum_clearance_m": clearance,
                "quality_score": score,
                "target_velocity_mps": target_velocity,
                "solved_joint_position_rad": normal,
            })

    accepted.sort(key=lambda item: (item["quality_score"], item["goal_id"]))
    unique_position: dict[tuple[float, float, float], dict[str, Any]] = {}
    unique_position_orientation: dict[tuple[float, float, float, float, float], dict[str, Any]] = {}
    unique_position_orientation_time: dict[tuple[float, float, float, float, float, float], dict[str, Any]] = {}
    for item in accepted:
        unique_position.setdefault(item["position"], item)
        unique_position_orientation.setdefault(
            (*item["position"], item["pitch_deg"], item["yaw_deg"]), item
        )
        unique_position_orientation_time.setdefault(
            (*item["position"], item["pitch_deg"], item["yaw_deg"], item["strike_time_s"]), item
        )

    def report_row(item: dict[str, Any]) -> dict[str, Any]:
        x, y, z = item["position"]
        return {
            "goal_id": item["goal_id"],
            "source_output_dir": item["source_output_dir"],
            "x_m": x,
            "y_m": y,
            "z_m": z,
            "pitch_deg": item["pitch_deg"],
            "yaw_deg": item["yaw_deg"],
            "strike_time_s": item["strike_time_s"],
            "position_error_m": item["position_error_m"],
            "normal_error_deg": item["normal_error_deg"],
            "velocity_error_mps": item["velocity_error_mps"],
            "minimum_clearance_m": item["minimum_clearance_m"],
            "quality_score": item["quality_score"],
        }

    all_rows = [report_row(item) for item in accepted]
    unique_rows = [report_row(item) for item in unique_position.values()]
    unique_orientation_rows = [
        report_row(item) for item in unique_position_orientation.values()
    ]
    unique_orientation_time_rows = [
        report_row(item) for item in unique_position_orientation_time.values()
    ]
    write_csv(args.output_root / "quality_candidates.csv", all_rows)
    write_csv(args.output_root / "unique_position_candidates.csv", unique_rows)
    write_csv(
        args.output_root / "unique_position_orientation_candidates.csv",
        unique_orientation_rows,
    )
    write_csv(
        args.output_root / "unique_position_orientation_time_candidates.csv",
        unique_orientation_time_rows,
    )
    (args.output_root / "criteria.json").write_text(
        json.dumps(
            {
                "source_summary": str(args.summary.resolve()),
                "selection": {
                    "status": "KINEMATIC_CANDIDATE",
                    "swing_type": args.swing_type,
                    "ready_goal_contract_match": True,
                    "position_error_max_m": args.max_position_error_m,
                    "normal_error_max_deg": args.max_normal_error_deg,
                    "velocity_error_max_mps": args.max_velocity_error_mps,
                    "minimum_clearance_min_m": args.min_clearance_m,
                    "timing_extension_s": 0,
                    "dedup_keys": {
                        "unique_position": "target_position_xyz",
                        "unique_position_orientation": "target_position_xyz_plus_pitch_deg_plus_yaw_deg",
                        "unique_position_orientation_time": "target_position_xyz_plus_pitch_deg_plus_yaw_deg_plus_strike_time_s",
                    },
                },
                "quality_candidate_count": len(all_rows),
                "unique_position_count": len(unique_rows),
                "unique_position_orientation_count": len(unique_orientation_rows),
                "unique_position_orientation_time_count": len(unique_orientation_time_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"quality_candidates={len(all_rows)}")
    print(f"unique_positions={len(unique_rows)}")
    print(f"output_root={args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
