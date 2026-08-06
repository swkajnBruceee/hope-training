#!/usr/bin/env python3
"""Compare canonical task anchors with actually qualified local P10 anchors."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


GRID_REPORT_BY_MOTION = {
    0: "eval_outputs/target_response/p5_motion0_calibrated_grid1cm.json",
    2: "eval_outputs/target_response/p7_motion2_calibrated_grid1cm.json",
    3: "eval_outputs/target_response/p4c_motion3_calibrated_anchor_grid1cm.json",
    4: "eval_outputs/target_response/p8_motion4_calibrated_grid1cm.json",
    5: "eval_outputs/target_response/p9_motion5_calibrated_grid1cm.json",
}


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument(
        "--center-report-glob",
        default="eval_outputs/strike_goal_p1/p10_center_motion_*.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.canonical_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_by_id = {
        int(entry["motion_id"]): np.asarray(
            entry["strike_target_b0"]["racket_position_b0_m"], dtype=np.float64
        )
        for entry in manifest["motions"]
    }
    p1_anchor = np.asarray((-0.5, -0.7625, 0.3084), dtype=np.float64)
    rows = []
    for filename in sorted(glob.glob(args.center_report_glob)):
        path = Path(filename).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        motion_id = int(report["motion_id"])
        if motion_id not in task_by_id:
            raise ValueError(f"center report motion {motion_id} is absent from canonical manifest")
        trial = report["trials"][0]
        qualified = np.asarray(
            report["auto_motion_selection"]["candidate_anchor_position_b_m"][motion_id],
            dtype=np.float64,
        )
        task = task_by_id[motion_id]
        delta = qualified - task
        grid_path = Path(GRID_REPORT_BY_MOTION[motion_id]).resolve()
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        trials = grid["trials"]
        rows.append(
            {
                "motion_id": motion_id,
                "canonical_task_anchor_b_m": task.tolist(),
                "qualified_local_anchor_b_m": qualified.tolist(),
                "qualified_local_anchor_p1_world_m": (qualified + p1_anchor).tolist(),
                "qualified_minus_task_anchor_b_m": delta.tolist(),
                "anchor_gap_m": float(np.linalg.norm(delta)),
                "center_execution": {
                    "position_error_m": float(trial["position_error_m"]),
                    "normal_error_deg": float(trial["normal_error_deg"]),
                    "velocity_error_mps": float(trial["velocity_error_mps"]),
                    "physics_termination_count": int(report["physical_termination_count"]),
                    "source": str(path),
                },
                "qualified_axis_grid_1cm": {
                    "trial_count": len(trials),
                    "complete": bool(grid["complete"]),
                    "physics_termination_count": int(grid["physical_termination_count"]),
                    "position_error_p50_m": _percentile(
                        [float(item["position_error_m"]) for item in trials], 50.0
                    ),
                    "position_error_p95_m": _percentile(
                        [float(item["position_error_m"]) for item in trials], 95.0
                    ),
                    "position_error_max_m": max(float(item["position_error_m"]) for item in trials),
                    "normal_error_max_deg": max(float(item["normal_error_deg"]) for item in trials),
                    "velocity_error_max_mps": max(float(item["velocity_error_mps"]) for item in trials),
                    "source": str(grid_path),
                },
            }
        )

    qualified_ids = {row["motion_id"] for row in rows}
    missing_ids = sorted(set(task_by_id) - qualified_ids)
    gaps = [row["anchor_gap_m"] for row in rows]
    center_position_errors = [row["center_execution"]["position_error_m"] for row in rows]
    local_half_range = np.asarray((0.01, 0.01, 0.01), dtype=np.float64)
    disconnected_box_gaps = []
    for row_index, lhs in enumerate(rows):
        lhs_position = np.asarray(lhs["qualified_local_anchor_b_m"], dtype=np.float64)
        for rhs in rows[row_index + 1 :]:
            rhs_position = np.asarray(rhs["qualified_local_anchor_b_m"], dtype=np.float64)
            separation = np.maximum(
                np.abs(lhs_position - rhs_position) - 2.0 * local_half_range, 0.0
            )
            disconnected_box_gaps.append(float(np.linalg.norm(separation)))
    output = {
        "purpose": "separate canonical task anchors from actually qualified local execution anchors",
        "canonical_manifest": str(manifest_path),
        "canonical_contract_version": manifest["contract_version"],
        "canonical_motion_count": len(task_by_id),
        "qualified_local_motion_ids": sorted(qualified_ids),
        "unqualified_motion_ids": missing_ids,
        "qualification": "five disconnected position-local skills; not a continuous strike workspace",
        "direct_planner_execution": False,
        "direct_planner_blockers": [
            "10D Planner goal has no action effect",
            "physical ball-center to policy-link contact transform is unresolved",
            "Planner/control clock mapping is unresolved for deployment",
            "qualified anchors differ materially from canonical task anchors",
            "velocity error remains large even on qualified local centers",
            "formal P1 dynamic execution has not been qualified",
        ],
        "aggregate": {
            "anchor_gap_min_m": min(gaps),
            "anchor_gap_max_m": max(gaps),
            "center_position_error_min_m": min(center_position_errors),
            "center_position_error_max_m": max(center_position_errors),
            "minimum_gap_between_qualified_1cm_boxes_m": min(disconnected_box_gaps),
            "maximum_gap_between_qualified_1cm_boxes_m": max(disconnected_box_gaps),
            "center_velocity_error_min_mps": min(
                row["center_execution"]["velocity_error_mps"] for row in rows
            ),
            "center_velocity_error_max_mps": max(
                row["center_execution"]["velocity_error_mps"] for row in rows
            ),
        },
        "motions": sorted(rows, key=lambda row: row["motion_id"]),
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
