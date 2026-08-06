#!/usr/bin/env python3
"""Validate the frozen P4 motion-3 native-time position-control contract.

Usage:
    python tools/verify_p4_motion3_position_contract.py \
        eval_outputs/target_response/p4c_motion3_calibrated_anchor_grid1cm.json

This is intentionally report-based: the expensive simulator replay remains in
``scripts/play.py``, while CI and release checks can reject a regression in the
recorded seven-point contract without importing Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CALIBRATED_CENTER_B_M = (0.483470643, 0.066296709, -0.082998851)
MAX_POSITION_ERROR_M = 0.007
EXPECTED_CONTROL_STEP = 78
DIAGONAL_GAIN_BOUNDS = {
    "x": (0.75, 1.10),
    "y": (0.40, 0.70),
    "z": (0.85, 1.10),
}
MAX_CROSS_AXIS_GAIN = 0.45


def validate(report: dict) -> list[str]:
    """Return contract violations; an empty list means the audit passes."""

    failures: list[str] = []
    if report.get("motion_id") != 3:
        failures.append(f"expected motion_id=3, got {report.get('motion_id')!r}")
    trials = report.get("trials") or []
    if len(trials) != 7:
        failures.append(f"expected seven calibrated grid trials, got {len(trials)}")
        return failures
    if report.get("physical_termination_count") != 0:
        failures.append(
            "expected zero physical terminations, got "
            f"{report.get('physical_termination_count')!r}"
        )

    for trial in trials:
        trial_id = trial.get("trial_id")
        if trial.get("control_step") != EXPECTED_CONTROL_STEP:
            failures.append(
                f"trial {trial_id}: expected native control step {EXPECTED_CONTROL_STEP}, "
                f"got {trial.get('control_step')!r}"
            )
        error = float(trial.get("position_error_m", math.inf))
        if error > MAX_POSITION_ERROR_M:
            failures.append(
                f"trial {trial_id}: position error {error:.4f} m exceeds "
                f"{MAX_POSITION_ERROR_M:.4f} m"
            )
        if trial.get("first_physical_termination_control_step") is not None:
            failures.append(f"trial {trial_id}: physical termination was recorded")

    baseline = min(
        trials,
        key=lambda row: sum(
            float(value) ** 2
            for value in row.get("control_offset_from_calibrated_anchor_b_m", ())
        ),
    )
    control_offset = baseline.get("control_offset_from_calibrated_anchor_b_m", ())
    if len(control_offset) != 3 or max(abs(float(value)) for value in control_offset) > 1e-6:
        failures.append("no calibrated-control zero target was found")
    center = baseline.get("target_position_b_m", ())
    if len(center) != 3 or max(
        abs(float(actual) - expected)
        for actual, expected in zip(center, CALIBRATED_CENTER_B_M)
    ) > 2e-4:
        failures.append(f"calibrated centre drifted: {center!r}")

    pairs = {entry.get("axis"): entry for entry in report.get("axis_pairs") or []}
    for axis, (low, high) in DIAGONAL_GAIN_BOUNDS.items():
        column = pairs.get(axis, {}).get("position_jacobian_column", ())
        if len(column) != 3:
            failures.append(f"missing {axis}-axis response column")
            continue
        axis_index = "xyz".index(axis)
        gain = float(column[axis_index])
        if not low <= gain <= high:
            failures.append(
                f"{axis} diagonal gain {gain:.3f} outside [{low:.2f}, {high:.2f}]"
            )
        cross = max(abs(float(value)) for index, value in enumerate(column) if index != axis_index)
        if cross > MAX_CROSS_AXIS_GAIN:
            failures.append(f"{axis} cross-axis gain {cross:.3f} exceeds {MAX_CROSS_AXIS_GAIN:.2f}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    with args.report.open(encoding="utf-8") as stream:
        failures = validate(json.load(stream))
    if failures:
        print("P4 motion-3 position contract: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("P4 motion-3 position contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
