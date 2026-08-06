#!/usr/bin/env python3
"""Summarize paired P4C upper-action-chain PhysX audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _states(report: dict) -> list[tuple[int, str, dict]]:
    states: list[tuple[int, str, dict]] = []
    for record in report.get("trace", []):
        for phase in ("pre_step_state", "post_step_state"):
            state = record.get(phase)
            if state is not None and "upper_action_chain" in state:
                states.append((int(record["control_step"]), phase, state))
    if not states:
        raise ValueError("report contains no upper_action_chain trace")
    return states


def _minimum(states, key: str) -> dict:
    value, step, phase, state = min(
        (float(state[key]), step, phase, state) for step, phase, state in states
    )
    return {
        "value": value,
        "control_step": step,
        "sample_phase": phase,
        "joint": state.get(key.replace("_rad", "_joint")),
    }


def _summarize_one(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    states = _states(report)
    names = states[0][2]["upper_action_chain"]["joint_names"]
    per_joint = {}
    minimum_command = (float("inf"), None, None, None)
    minimum_actual = (float("inf"), None, None, None)
    for index, name in enumerate(names):
        actual_event = min(
            (
                float(state["upper_action_chain"]["actual_soft_margin_rad"][index]),
                step,
                phase,
                state,
            )
            for step, phase, state in states
        )
        command_event = min(
            (
                float(
                    state["upper_action_chain"][
                        "processed_command_soft_margin_rad"
                    ][index]
                ),
                step,
                phase,
                state,
            )
            for step, phase, state in states
        )
        minimum_actual = min(minimum_actual, actual_event, key=lambda item: item[0])
        minimum_command = min(
            minimum_command, command_event, key=lambda item: item[0]
        )
        margin, step, phase, state = actual_event
        chain = state["upper_action_chain"]
        per_joint[name] = {
            "minimum_command_soft_margin_rad": command_event[0],
            "minimum_actual_soft_margin_rad": margin,
            "actual_soft_limit_violation_samples": sum(
                float(sample["upper_action_chain"]["actual_soft_margin_rad"][index])
                < 0.0
                for _, _, sample in states
            ),
            "minimum_actual_event": {
                "control_step": step,
                "sample_phase": phase,
                "motion_frame": state.get("motion_frame"),
                "tail_steps": state.get("tail_steps"),
                "safe_reference_position_rad": chain[
                    "safe_reference_position_rad"
                ][index],
                "frozen_actor_contribution_rad": chain[
                    "frozen_actor_contribution_rad"
                ][index],
                "coordinator_contribution_rad": chain[
                    "coordinator_contribution_rad"
                ][index],
                "target_adapter_contribution_rad": chain[
                    "target_adapter_contribution_rad"
                ][index],
                "safety_override_rad": chain["safety_override_rad"][index],
                "processed_command_position_rad": chain[
                    "processed_command_position_rad"
                ][index],
                "processed_command_velocity_radps": chain[
                    "processed_command_velocity_radps"
                ][index],
                "actual_position_rad": chain["actual_position_rad"][index],
                "actual_velocity_radps": chain["actual_velocity_radps"][index],
            },
            "maximum_absolute_contribution_rad": {
                contribution: max(
                    abs(float(sample["upper_action_chain"][contribution][index]))
                    for _, _, sample in states
                )
                for contribution in (
                    "frozen_actor_contribution_rad",
                    "coordinator_contribution_rad",
                    "target_adapter_contribution_rad",
                    "safety_override_rad",
                )
            },
        }

    reference_minimum = _minimum(
        states, "minimum_reference_soft_joint_margin_rad"
    )
    trial = report["trials"][0]
    if reference_minimum["value"] < 0.0:
        responsibility = "A_REFERENCE_UNSAFE"
    elif minimum_command[0] < 0.0:
        responsibility = "B_COMMAND_UNSAFE"
    elif minimum_actual[0] < 0.0:
        responsibility = "C_DYNAMIC_OVERSHOOT_OR_TRACKING"
    else:
        responsibility = "PASS_ALL_POSITION_LIMIT_LAYERS"

    layers = report.get("goal_state_layers") or {}
    return {
        "source_report": str(path.resolve()),
        "execution_mode": report.get("p4c_upper_execution_mode", "unknown"),
        "control_steps": report.get("control_steps"),
        "physical_termination_count": report.get("physical_termination_count"),
        "hit": {
            "control_step": trial["control_step"],
            "position_error_m": trial["position_error_m"],
            "normal_error_deg": trial["normal_error_deg"],
            "velocity_error_mps": trial["velocity_error_mps"],
        },
        "task_space_error_decomposition": layers.get("error_decomposition"),
        "minimum_reference_soft_margin": reference_minimum,
        "minimum_upper_command_soft_margin_rad": minimum_command[0],
        "minimum_upper_actual_soft_margin_rad": minimum_actual[0],
        "minimum_robot_actual_soft_margin": _minimum(
            states, "minimum_actual_soft_joint_margin_rad"
        ),
        "minimum_robot_actual_hard_margin": _minimum(
            states, "minimum_actual_hard_joint_margin_rad"
        ),
        "responsibility_class": responsibility,
        "per_upper_joint": per_joint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-reference-policy", type=Path, required=True)
    parser.add_argument("--repaired-reference-policy", type=Path, required=True)
    parser.add_argument("--repaired-reference-only", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {
        "old_reference_plus_policy": _summarize_one(args.old_reference_policy),
        "repaired_reference_plus_policy": _summarize_one(
            args.repaired_reference_policy
        ),
        "repaired_reference_only": _summarize_one(args.repaired_reference_only),
    }
    reference_only = reports["repaired_reference_only"]
    output = {
        "schema_version": "p4c_action_chain_responsibility/v1",
        "training_started": False,
        "ppo_allowed": False,
        "paired_reports": reports,
        "conclusion": {
            "responsibility_class": reference_only["responsibility_class"],
            "safe_reference_is_kinematically_valid": (
                reference_only["minimum_reference_soft_margin"]["value"] >= 0.0
            ),
            "processed_upper_command_is_soft_limit_safe": (
                reference_only["minimum_upper_command_soft_margin_rad"] >= 0.0
            ),
            "actual_upper_execution_is_soft_limit_safe": (
                reference_only["minimum_upper_actual_soft_margin_rad"] >= 0.0
            ),
            "reference_only_improves_position_over_repaired_policy": (
                reference_only["hit"]["position_error_m"]
                < reports["repaired_reference_plus_policy"]["hit"][
                    "position_error_m"
                ]
            ),
            "next_required_gate": (
                "qualify direct reference tracking with dynamic limit braking, "
                "phase/velocity alignment, and positive actual soft-limit margins"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output["conclusion"], indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
