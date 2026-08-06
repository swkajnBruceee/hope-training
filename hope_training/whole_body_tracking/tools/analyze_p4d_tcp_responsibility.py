#!/usr/bin/env python3
"""Rank P4D hit-frame upper-joint tracking errors by TCP error contribution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _vec(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must be a three-vector")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} contains a non-finite value")
    return result


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(item * item for item in vector))


def _sub(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _hit_state(report: dict[str, Any]) -> dict[str, Any]:
    hit_step = int(report["all_hit_control_step"])
    for row in report.get("trace", []):
        if int(row.get("control_step", -1)) == hit_step and isinstance(row.get("post_step_state"), dict):
            return row["post_step_state"]
    raise ValueError(f"missing post-step trace state at hit step {hit_step}")


def analyze(report: dict[str, Any], source: Path) -> dict[str, Any]:
    state = _hit_state(report)
    racket, chain = state.get("racket_state"), state.get("upper_action_chain")
    if not isinstance(racket, dict) or not isinstance(chain, dict):
        raise ValueError("trace lacks racket_state or upper_action_chain")
    target = _vec(racket.get("target_position_w_m"), "target_position_w_m")
    actual = _vec(racket.get("actual_position_w_m"), "actual_position_w_m")
    error = _sub(target, actual)
    error_norm = _norm(error)
    error_dir = [item / error_norm for item in error] if error_norm else [0.0, 0.0, 0.0]
    names = chain.get("joint_names", [])
    reference = chain.get("safe_reference_position_rad", [])
    actual_q = chain.get("actual_position_rad", [])
    jacobian = chain.get("tcp_linear_jacobian_xyz_m_per_rad", [])
    safe_contributions = chain.get("linearized_safe_reference_minus_actual_tcp_xyz_m", [])
    command_contributions = chain.get("linearized_processed_command_minus_actual_tcp_xyz_m", [])
    if len({len(names), len(reference), len(actual_q), len(jacobian), len(safe_contributions), len(command_contributions)}) != 1:
        raise ValueError("inconsistent upper action-chain lengths")
    processed_command = chain.get("processed_command_position_rad", [])
    if len(processed_command) != len(names):
        raise ValueError("processed_command_position_rad has an inconsistent length")
    total = [0.0, 0.0, 0.0]
    joints = []
    for name, ref, command, q, jac, safe_contribution, command_contribution in zip(
        names, reference, processed_command, actual_q, jacobian, safe_contributions, command_contributions
    ):
        jac_vec = _vec(jac, f"jacobian[{name}]")
        safe_delta = _vec(safe_contribution, f"safe contribution[{name}]")
        delta = _vec(command_contribution, f"command contribution[{name}]")
        length = _norm(delta)
        along = sum(a * b for a, b in zip(delta, error_dir))
        total = [a + b for a, b in zip(total, delta)]
        joints.append(
            {
                "joint": str(name),
                "safe_reference_minus_actual_rad": float(ref) - float(q),
                "processed_command_minus_actual_rad": float(command) - float(q),
                "safe_reference_minus_processed_command_rad": float(ref) - float(command),
                "tcp_linear_jacobian_xyz_m_per_rad": jac_vec,
                "linearized_safe_reference_minus_actual_tcp_xyz_m": safe_delta,
                "linearized_processed_command_minus_actual_tcp_xyz_m": delta,
                "linearized_processed_command_minus_actual_tcp_norm_m": length,
                "linearized_processed_command_toward_target_error_m": along,
                "linearized_processed_command_alignment_with_target_error": along / length if length else 0.0,
                "role": "reduces_target_error" if along > 0.0 else "opposes_target_error",
            }
        )
    joints.sort(
        key=lambda item: abs(float(item["linearized_processed_command_toward_target_error_m"])),
        reverse=True,
    )
    unexplained = _sub(error, total)
    return {
        "schema_version": "p4d_tcp_responsibility/v1",
        "source_report": str(source.resolve()),
        "motion_id": int(report["motion_id"]),
        "tagged_hit_control_step": int(report["all_hit_control_step"]),
        "physx_mapping": {
            "jacobian_shape": chain.get("physx_jacobian_shape"),
            "body_row": chain.get("physx_jacobian_body_row"),
            "joint_column_offset": chain.get("physx_jacobian_joint_column_offset"),
            "upper_articulation_joint_ids": chain.get("articulation_upper_joint_ids"),
            "upper_physx_column_ids": chain.get("physx_upper_jacobian_column_ids"),
        },
        "target_minus_actual_tcp_xyz_m": error,
        "target_minus_actual_tcp_norm_m": error_norm,
        "linearized_processed_command_minus_actual_tcp_xyz_m": total,
        "linearized_processed_command_minus_actual_tcp_norm_m": _norm(total),
        "target_error_not_explained_by_command_tracking_linearization_xyz_m": unexplained,
        "target_error_not_explained_by_command_tracking_linearization_norm_m": _norm(unexplained),
        "racket_velocity": {
            key: racket.get(key)
            for key in ("velocity_error_mps", "actual_speed_mps", "target_speed_mps", "velocity_direction_error_deg")
        },
        "joint_responsibility_ranked_by_absolute_target_projection": joints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.report.expanduser().resolve()
    result = analyze(json.loads(source.read_text(encoding="utf-8")), source)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
