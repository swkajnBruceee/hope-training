#!/usr/bin/env python3
"""Analyze P4D nominal hit windows from ``play.py`` trace reports.

This is an evaluation-only tool.  It deliberately reports the independently
best position, normal and velocity samples before an optional normalized
composite score, so a phase error cannot be mistaken for a trajectory-shape
or velocity-direction error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _float(value: Any) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _minimum(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return min(rows, key=lambda row: _float(row["racket_state"].get(key)))


def _state_row(row: dict[str, Any]) -> dict[str, Any] | None:
    state = row.get("post_step_state")
    if not isinstance(state, dict) or "racket_state" not in state:
        return None
    return {
        "control_step": int(row["control_step"]),
        "racket_state": state["racket_state"],
        "minimum_actual_soft_margin_rad": state.get("minimum_actual_soft_joint_margin_rad"),
        "minimum_actual_soft_margin_joint": state.get("minimum_actual_soft_joint_margin_joint"),
        "upper_action_chain": state.get("upper_action_chain", {}),
    }


def _joint_tracking(row: dict[str, Any]) -> dict[str, float]:
    chain = row["upper_action_chain"]
    names = chain.get("joint_names", [])
    actual = chain.get("actual_position_rad", [])
    reference = chain.get("safe_reference_position_rad", [])
    if not (len(names) == len(actual) == len(reference)):
        return {}
    return {str(name): float(ref) - float(act) for name, act, ref in zip(names, actual, reference)}


def _summary(report: dict[str, Any], radius: int, scales: tuple[float, float, float]) -> dict[str, Any]:
    hit_step = int(report["all_hit_control_step"])
    rows = [row for raw in report.get("trace", []) if (row := _state_row(raw)) is not None]
    window = [row for row in rows if abs(row["control_step"] - hit_step) <= radius]
    if not window:
        raise ValueError(f"no racket-state rows in +/-{radius} window around step {hit_step}")

    position_best = _minimum(window, "position_error_m")
    normal_best = _minimum(window, "normal_error_deg")
    velocity_best = _minimum(window, "velocity_error_mps")
    position_scale, normal_scale, velocity_scale = scales
    composite_best = min(
        window,
        key=lambda row: (
            _float(row["racket_state"].get("position_error_m")) / position_scale
            + _float(row["racket_state"].get("normal_error_deg")) / normal_scale
            + _float(row["racket_state"].get("velocity_error_mps")) / velocity_scale
        ),
    )
    tagged = min(window, key=lambda row: abs(row["control_step"] - hit_step))
    tracking = _joint_tracking(tagged)
    ordered_tracking = sorted(tracking.items(), key=lambda item: abs(item[1]), reverse=True)

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        state = row["racket_state"]
        return {
            "control_step": row["control_step"],
            "step_offset_from_tagged_hit": row["control_step"] - hit_step,
            "position_error_m": state["position_error_m"],
            "normal_error_deg": state["normal_error_deg"],
            "velocity_error_mps": state["velocity_error_mps"],
            "actual_speed_mps": state["actual_speed_mps"],
            "target_speed_mps": state["target_speed_mps"],
            "speed_error_mps": state["speed_error_mps"],
            "velocity_direction_error_deg": state["velocity_direction_error_deg"],
            "minimum_actual_soft_margin_rad": row["minimum_actual_soft_margin_rad"],
            "minimum_actual_soft_margin_joint": row["minimum_actual_soft_margin_joint"],
        }

    return {
        "motion_id": int(report["motion_id"]),
        "source_report": report.get("source_report"),
        "tagged_hit_control_step": hit_step,
        "window_radius_control_steps": radius,
        "window_step_range": [window[0]["control_step"], window[-1]["control_step"]],
        "tagged_hit": compact(tagged),
        "best_position": compact(position_best),
        "best_normal": compact(normal_best),
        "best_velocity": compact(velocity_best),
        "best_composite": compact(composite_best),
        "tagged_hit_upper_joint_reference_minus_actual_rad": tracking,
        "tagged_hit_largest_upper_joint_tracking_errors": [
            {"joint": name, "reference_minus_actual_rad": value}
            for name, value in ordered_tracking
        ],
        "window_minimum_global_soft_margin_rad": min(
            _float(row["minimum_actual_soft_margin_rad"]) for row in window
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True, help="MOTION_ID=trace.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--position-scale-m", type=float, default=0.10)
    parser.add_argument("--normal-scale-deg", type=float, default=10.0)
    parser.add_argument("--velocity-scale-mps", type=float, default=1.50)
    args = parser.parse_args()
    if args.radius < 0:
        raise ValueError("radius must be non-negative")
    scales = (args.position_scale_m, args.normal_scale_deg, args.velocity_scale_mps)
    if any(value <= 0.0 for value in scales):
        raise ValueError("all composite scales must be positive")

    summaries: dict[str, Any] = {}
    for spec in args.report:
        motion_text, path_text = spec.split("=", 1)
        path = Path(path_text)
        report = json.loads(path.read_text(encoding="utf-8"))
        report["source_report"] = str(path.resolve())
        summary = _summary(report, args.radius, scales)
        motion_id = int(motion_text)
        if summary["motion_id"] != motion_id:
            raise ValueError(f"motion ID mismatch for {path}: argument={motion_id}, report={summary['motion_id']}")
        summaries[str(motion_id)] = summary

    output = {
        "schema_version": "p4d_hit_window_scan/v1",
        "composite_scales": {
            "position_m": scales[0],
            "normal_deg": scales[1],
            "velocity_mps": scales[2],
        },
        "motions": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
