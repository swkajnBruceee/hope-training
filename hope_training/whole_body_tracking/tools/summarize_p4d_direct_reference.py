#!/usr/bin/env python3
"""Summarize P4D reference-only qualification reports by motion prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(spec: str) -> tuple[int, Path, dict]:
    motion_text, path_text = spec.split("=", 1)
    path = Path(path_text)
    return int(motion_text), path, json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True, help="MOTION_ID=report.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    motions = {}
    for motion_id, path, report in sorted(_load(item) for item in args.report):
        states = [
            row["post_step_state"]
            for row in report["trace"]
            if row.get("post_step_state")
            and "upper_action_chain" in row["post_step_state"]
        ]
        chain0 = states[0]["upper_action_chain"]
        names = chain0["joint_names"]
        upper_minimum = min(
            (
                min(state["upper_action_chain"]["actual_soft_margin_rad"][index] for state in states),
                name,
            )
            for index, name in enumerate(names)
        )
        command_minimum = min(
            min(state["upper_action_chain"]["processed_command_soft_margin_rad"])
            for state in states
        )
        global_state = min(
            states, key=lambda state: state["minimum_actual_soft_joint_margin_rad"]
        )
        hit = report["trials"][0]
        dynamic_override_peak = max(
            max(abs(value) for value in state["upper_action_chain"].get(
                "dynamic_safety_override_rad", [0.0] * len(names)
            ))
            for state in states
        )
        motions[str(motion_id)] = {
            "source_report": str(path.resolve()),
            "physical_termination_count": report["physical_termination_count"],
            "position_error_m": hit["position_error_m"],
            "normal_error_deg": hit["normal_error_deg"],
            "velocity_error_mps": hit["velocity_error_mps"],
            "minimum_upper_command_soft_margin_rad": command_minimum,
            "minimum_upper_actual_soft_margin_rad": upper_minimum[0],
            "minimum_upper_actual_soft_margin_joint": upper_minimum[1],
            "minimum_global_actual_soft_margin_rad": global_state[
                "minimum_actual_soft_joint_margin_rad"
            ],
            "minimum_global_actual_soft_margin_joint": global_state[
                "minimum_actual_soft_joint_margin_joint"
            ],
            "dynamic_guard_position_override_peak_rad": dynamic_override_peak,
            "stability": report["stability"],
            "upper_execution_safety_pass": (
                report["physical_termination_count"] == 0
                and upper_minimum[0] > 0.0
                and command_minimum > 0.0
            ),
        }

    values = list(motions.values())
    result = {
        "schema_version": "p4d_direct_reference_qualification/v1",
        "motions": motions,
        "bank_upper_execution_safety_pass": all(
            value["upper_execution_safety_pass"] for value in values
        ),
        "maximum_position_error_m": max(value["position_error_m"] for value in values),
        "maximum_normal_error_deg": max(value["normal_error_deg"] for value in values),
        "maximum_velocity_error_mps": max(value["velocity_error_mps"] for value in values),
        "minimum_upper_actual_soft_margin_rad": min(
            value["minimum_upper_actual_soft_margin_rad"] for value in values
        ),
        "adapter_teacher_generation_approved": False,
        "remaining_blockers": [
            "reference-to-actual strike errors remain large and motion-dependent",
            "velocity error remains especially large for motions 4 and 5",
            "left_shoulder_roll READY contract is slightly outside the global soft limit",
            "the bank has not yet been tested under initial-state/dynamics perturbations",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
