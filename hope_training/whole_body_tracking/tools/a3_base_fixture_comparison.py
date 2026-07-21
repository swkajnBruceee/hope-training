#!/usr/bin/env python3
"""Pure comparison logic for paired A3 Isaac/MuJoCo fixture evidence."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


ALLOWED_DIFFERENCE_LABELS = {
    "expected_integrator_difference",
    "expected_contact_model_difference",
    "expected_geometry_difference",
    "expected_actuator_difference",
    "unexplained",
}


def compare_pair(
    *,
    isaac_result: Mapping[str, Any],
    isaac_evidence: Mapping[str, np.ndarray],
    mujoco_result: Mapping[str, Any],
    mujoco_evidence: Mapping[str, np.ndarray],
    trace_metadata: Mapping[str, Any],
    difference_labels: list[str],
    rationale: str,
) -> dict[str, Any]:
    if not difference_labels or not set(difference_labels) <= ALLOWED_DIFFERENCE_LABELS:
        raise ValueError("one or more valid difference labels are required")
    if not rationale.strip():
        raise ValueError("comparison classification requires a rationale")
    identity_fields = ("case_id", "trace_sha256", "matrix_sha256")
    for field in identity_fields:
        if isaac_result.get(field) != mujoco_result.get(field):
            raise ValueError(f"paired result {field} mismatch")
    if isaac_result["trace_sha256"] != trace_metadata.get("trace_sha256"):
        raise ValueError("paired results do not match trace metadata")
    if isaac_result["runner_facts"].get("selected_joint_name") != mujoco_result[
        "runner_facts"
    ].get("selected_joint_name"):
        raise ValueError("paired selected joint mismatch")
    if isaac_result["runner_facts"].get("ground_contact_enabled") is not False or mujoco_result[
        "runner_facts"
    ].get("ground_contact_enabled") is not False:
        raise ValueError("v3 fixture comparison requires contact-disabled evidence")

    common_arrays = (
        "time_s",
        "joint_q_rad",
        "joint_target_rad",
        "joint_dq_radps",
        "joint_torque_nm",
        "selected_joint_saturated",
    )
    for name in common_arrays:
        if name not in isaac_evidence or name not in mujoco_evidence:
            raise ValueError(f"paired evidence missing {name}")
    isaac_time = np.asarray(isaac_evidence["time_s"], dtype=np.float64)
    mujoco_time = np.asarray(mujoco_evidence["time_s"], dtype=np.float64)
    if not np.array_equal(isaac_time, mujoco_time):
        raise ValueError("paired evidence timestamps differ")
    if not np.array_equal(
        np.asarray(isaac_evidence["joint_target_rad"]),
        np.asarray(mujoco_evidence["joint_target_rad"]),
    ):
        raise ValueError("paired evidence targets differ")

    window = trace_metadata["metric_window"]
    baseline_indices = np.flatnonzero(isaac_time <= float(window["baseline_end_s"]))
    active = (isaac_time > float(window["active_start_s"])) & (
        isaac_time <= float(window["active_end_s"])
    )
    if baseline_indices.size == 0 or not np.any(active):
        raise ValueError("paired evidence lacks comparison windows")
    isaac_q = np.asarray(isaac_evidence["joint_q_rad"], dtype=np.float64)
    mujoco_q = np.asarray(mujoco_evidence["joint_q_rad"], dtype=np.float64)
    isaac_delta = isaac_q - isaac_q[baseline_indices[-1]]
    mujoco_delta = mujoco_q - mujoco_q[baseline_indices[-1]]
    command_delta = float(isaac_result["metrics"]["commanded_joint_delta_rad"])
    if not math.isfinite(command_delta) or command_delta == 0.0:
        raise ValueError("paired command delta is invalid")
    delta_error = isaac_delta[active] - mujoco_delta[active]
    torque_error = np.asarray(
        isaac_evidence["joint_torque_nm"], dtype=np.float64
    )[active] - np.asarray(mujoco_evidence["joint_torque_nm"], dtype=np.float64)[active]
    isaac_metrics = isaac_result["metrics"]
    mujoco_metrics = mujoco_result["metrics"]

    def relative_difference(first: float, second: float) -> float:
        denominator = max(abs(first), abs(second), 1.0e-12)
        return abs(first - second) / denominator

    isaac_steady = float(isaac_metrics["end_window_joint_delta_rad"])
    mujoco_steady = float(mujoco_metrics["end_window_joint_delta_rad"])
    command_sign = 1 if command_delta > 0.0 else -1
    comparison = {
        "case_id": isaac_result["case_id"],
        "trace_sha256": isaac_result["trace_sha256"],
        "matrix_sha256": isaac_result["matrix_sha256"],
        "selected_joint_name": isaac_result["runner_facts"]["selected_joint_name"],
        "transport_mode": trace_metadata["transport_mode"],
        "identity_and_time_alignment_pass": True,
        "both_safety_envelope_passed": bool(
            isaac_result["case_validation"]["safety_envelope_passed"]
            and mujoco_result["case_validation"]["safety_envelope_passed"]
        ),
        "response_direction_consistent": bool(
            (1 if isaac_steady > 0.0 else -1 if isaac_steady < 0.0 else 0) == command_sign
            and (1 if mujoco_steady > 0.0 else -1 if mujoco_steady < 0.0 else 0)
            == command_sign
        ),
        "active_delta_trajectory_rmse_rad": float(
            np.sqrt(np.mean(delta_error * delta_error))
        ),
        "active_delta_trajectory_nrmse_by_command": float(
            np.sqrt(np.mean(delta_error * delta_error)) / abs(command_delta)
        ),
        "active_torque_trajectory_rmse_nm": float(
            np.sqrt(np.mean(torque_error * torque_error))
        ),
        "end_window_response_ratio_abs_difference": abs(
            float(isaac_metrics["end_window_response_ratio"])
            - float(mujoco_metrics["end_window_response_ratio"])
        ),
        "selected_effort_rms_relative_difference": relative_difference(
            float(isaac_metrics["selected_joint_effort_rms_nm"]),
            float(mujoco_metrics["selected_joint_effort_rms_nm"]),
        ),
        "selected_peak_torque_relative_difference": relative_difference(
            float(isaac_metrics["selected_joint_peak_torque_nm"]),
            float(mujoco_metrics["selected_joint_peak_torque_nm"]),
        ),
        "saturation_duration_abs_difference_s": abs(
            float(isaac_metrics["selected_joint_saturation_duration_s"])
            - float(mujoco_metrics["selected_joint_saturation_duration_s"])
        ),
        "difference_labels": sorted(set(difference_labels)),
        "classification_rationale": rationale.strip(),
        "classification_frozen": False,
        "legacy_pre_window_baseline_only": True,
        "unexplained_blocks_stand": "unexplained" in difference_labels,
        "automatic_promotion": False,
    }
    return comparison
