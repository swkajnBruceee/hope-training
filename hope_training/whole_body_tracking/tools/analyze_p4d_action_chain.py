#!/usr/bin/env python3
"""Summarize the reference -> command -> actual responsibility split.

This is intentionally independent of TCP Jacobian attribution: it first
establishes whether a reference trajectory actually passed through the runtime
safety contract before interpreting tracking or task-space errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _samples(report: dict) -> tuple[list[dict], list[str]]:
    values = [
        row["post_step_state"]["upper_action_chain"]
        for row in report.get("trace", [])
        if isinstance(row.get("post_step_state"), dict)
        and "upper_action_chain" in row["post_step_state"]
    ]
    if not values:
        raise ValueError("report has no upper_action_chain trace samples")
    names = values[0].get("joint_names", [])
    if not names:
        raise ValueError("upper_action_chain has no joint_names")
    return values, list(names)


def _matrix(samples: list[dict], field: str, width: int) -> np.ndarray:
    result = np.asarray([row[field] for row in samples], dtype=np.float64)
    if result.shape != (len(samples), width):
        raise ValueError(f"{field} has shape {result.shape}, expected {(len(samples), width)}")
    if not np.isfinite(result).all():
        raise ValueError(f"{field} contains a non-finite value")
    return result


def _per_joint(names: list[str], values: np.ndarray, window: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {
            "full_cycle_max_abs_rad": float(np.max(np.abs(values[:, index]))),
            "hit_window_max_abs_rad": float(np.max(np.abs(window[:, index]))),
            "hit_window_rms_rad": _rms(window[:, index]),
        }
        for index, name in enumerate(names)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hit-radius", type=int, default=8)
    parser.add_argument("--projection-tolerance-rad", type=float, default=1.0e-6)
    args = parser.parse_args()
    if args.hit_radius < 0 or args.projection_tolerance_rad < 0.0:
        raise ValueError("hit radius and projection tolerance must be non-negative")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    samples, names = _samples(report)
    width = len(names)
    reference = _matrix(samples, "safe_reference_position_rad", width)
    command = _matrix(samples, "processed_command_position_rad", width)
    actual = _matrix(samples, "actual_position_rad", width)
    projection = command - reference
    tracking = actual - command
    total = actual - reference
    if not np.allclose(total, projection + tracking, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("action-chain decomposition does not close")

    hit_step = int(report["all_hit_control_step"])
    steps = np.asarray(
        [
            int(row["control_step"])
            for row in report["trace"]
            if isinstance(row.get("post_step_state"), dict)
            and "upper_action_chain" in row["post_step_state"]
        ]
    )
    window_mask = np.abs(steps - hit_step) <= args.hit_radius
    if not np.any(window_mask):
        raise ValueError("hit window contains no trace samples")
    projection_window = projection[window_mask]
    tracking_window = tracking[window_mask]
    total_window = total[window_mask]
    safety_override = _matrix(samples, "safety_override_rad", width)
    dynamic_override = _matrix(samples, "dynamic_safety_override_rad", width)

    def compact(values: np.ndarray, selected: np.ndarray) -> dict[str, float]:
        return {
            "full_cycle_max_abs_rad": float(np.max(np.abs(values))),
            "hit_window_max_abs_rad": float(np.max(np.abs(selected))),
            "hit_window_rms_rad": _rms(selected),
        }

    result = {
        "schema_version": "p4d_action_chain_responsibility/v1",
        "source_report": str(args.report.resolve()),
        "motion_id": int(report["motion_id"]),
        "execution_mode": report.get("p4c_upper_execution_mode"),
        "control_steps": len(samples),
        "tagged_hit_control_step": hit_step,
        "hit_window_radius_control_steps": args.hit_radius,
        "projection_definition": "processed_command_minus_safe_reference",
        "tracking_definition": "actual_minus_processed_command",
        "total_definition": "actual_minus_safe_reference",
        "decomposition": {
            "identity": "total = projection + tracking",
            "projection": {"aggregate": compact(projection, projection_window), "per_joint": _per_joint(names, projection, projection_window)},
            "tracking": {"aggregate": compact(tracking, tracking_window), "per_joint": _per_joint(names, tracking, tracking_window)},
            "total": {"aggregate": compact(total, total_window), "per_joint": _per_joint(names, total, total_window)},
        },
        "safety_filter": {
            "projection_tolerance_rad": args.projection_tolerance_rad,
            "triggered_control_steps": int(np.count_nonzero(np.any(np.abs(projection) > args.projection_tolerance_rad, axis=1))),
            "hit_window_triggered_control_steps": int(np.count_nonzero(np.any(np.abs(projection_window) > args.projection_tolerance_rad, axis=1))),
            "static_override_peak_abs_rad": float(np.max(np.abs(safety_override))),
            "dynamic_override_peak_abs_rad": float(np.max(np.abs(dynamic_override))),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
