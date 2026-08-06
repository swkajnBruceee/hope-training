#!/usr/bin/env python3
"""Compare paired step cases after subtracting same-workpoint zero baselines."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from a3_base_fixture_comparison import ALLOWED_DIFFERENCE_LABELS


CLASSIFICATION_COLORS = {"green", "yellow", "red"}


def _slope(time_s: np.ndarray, values: np.ndarray) -> float:
    centered = time_s - float(np.mean(time_s))
    denominator = float(np.dot(centered, centered))
    if time_s.size < 2 or denominator <= 0.0:
        return 0.0
    return float(np.dot(centered, values - float(np.mean(values))) / denominator)


def _relative_difference(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1.0e-12)


def _symmetric_difference(first: float, second: float) -> float:
    denominator = 0.5 * (abs(first) + abs(second))
    return abs(first - second) / denominator if denominator > 1.0e-12 else 0.0


def compare_step_with_zero_baselines(
    *,
    isaac_step_result: Mapping[str, Any],
    isaac_step_evidence: Mapping[str, np.ndarray],
    isaac_zero_result: Mapping[str, Any],
    isaac_zero_evidence: Mapping[str, np.ndarray],
    mujoco_step_result: Mapping[str, Any],
    mujoco_step_evidence: Mapping[str, np.ndarray],
    mujoco_zero_result: Mapping[str, Any],
    mujoco_zero_evidence: Mapping[str, np.ndarray],
    step_trace_metadata: Mapping[str, Any],
    zero_trace_metadata: Mapping[str, Any],
    classification_color: str,
    difference_labels: list[str],
    rationale: str,
) -> dict[str, Any]:
    if classification_color not in CLASSIFICATION_COLORS:
        raise ValueError("classification color must be green, yellow, or red")
    if not difference_labels or not set(difference_labels) <= ALLOWED_DIFFERENCE_LABELS:
        raise ValueError("one or more valid difference labels are required")
    if not rationale.strip():
        raise ValueError("comparison classification requires a rationale")
    results = (
        isaac_step_result,
        isaac_zero_result,
        mujoco_step_result,
        mujoco_zero_result,
    )
    matrix_hashes = {result.get("matrix_sha256") for result in results}
    if len(matrix_hashes) != 1:
        raise ValueError("step/zero results use different matrices")
    selected_names = {
        result.get("runner_facts", {}).get("selected_joint_name") for result in results
    }
    if len(selected_names) != 1 or None in selected_names:
        raise ValueError("step/zero selected joint mismatch")
    if isaac_step_result.get("trace_sha256") != step_trace_metadata.get("trace_sha256"):
        raise ValueError("Isaac step result/metadata mismatch")
    if mujoco_step_result.get("trace_sha256") != step_trace_metadata.get("trace_sha256"):
        raise ValueError("MuJoCo step result/metadata mismatch")
    if isaac_zero_result.get("trace_sha256") != zero_trace_metadata.get("trace_sha256"):
        raise ValueError("Isaac zero result/metadata mismatch")
    if mujoco_zero_result.get("trace_sha256") != zero_trace_metadata.get("trace_sha256"):
        raise ValueError("MuJoCo zero result/metadata mismatch")
    if any(result["runner_facts"].get("ground_contact_enabled") is not False for result in results):
        raise ValueError("zero-baseline comparison requires fixture v3 without contact")
    if isaac_zero_result.get("case_validation", {}).get("category") != "joint_zero_baseline":
        raise ValueError("Isaac zero result category mismatch")
    if mujoco_zero_result.get("case_validation", {}).get("category") != "joint_zero_baseline":
        raise ValueError("MuJoCo zero result category mismatch")

    evidence_sets = (
        isaac_step_evidence,
        isaac_zero_evidence,
        mujoco_step_evidence,
        mujoco_zero_evidence,
    )
    time_arrays = [np.asarray(evidence["time_s"], dtype=np.float64) for evidence in evidence_sets]
    if any(not np.array_equal(time_arrays[0], other) for other in time_arrays[1:]):
        raise ValueError("step/zero evidence timestamps differ")
    time_s = time_arrays[0]
    window = step_trace_metadata["metric_window"]
    active = (time_s > float(window["active_start_s"])) & (
        time_s <= float(window["active_end_s"])
    )
    end_window_s = float(window.get("end_window_s", 0.1))
    tail = (time_s > max(float(window["active_start_s"]), float(window["active_end_s"]) - end_window_s)) & (
        time_s <= float(window["active_end_s"])
    )
    if not np.any(active) or np.count_nonzero(tail) < 2:
        raise ValueError("step/zero evidence lacks active or end window")

    isaac_response = np.asarray(isaac_step_evidence["joint_q_rad"], dtype=np.float64) - np.asarray(
        isaac_zero_evidence["joint_q_rad"], dtype=np.float64
    )
    mujoco_response = np.asarray(mujoco_step_evidence["joint_q_rad"], dtype=np.float64) - np.asarray(
        mujoco_zero_evidence["joint_q_rad"], dtype=np.float64
    )
    command_delta = float(isaac_step_result["metrics"]["commanded_joint_delta_rad"])
    other_delta = float(mujoco_step_result["metrics"]["commanded_joint_delta_rad"])
    if command_delta == 0.0 or not math.isclose(command_delta, other_delta, abs_tol=1.0e-12):
        raise ValueError("paired step command deltas differ or are zero")
    isaac_end = float(np.mean(isaac_response[tail]))
    mujoco_end = float(np.mean(mujoco_response[tail]))
    isaac_gain = isaac_end / command_delta
    mujoco_gain = mujoco_end / command_delta
    command_sign = 1.0 if command_delta > 0.0 else -1.0
    isaac_active_diff = np.diff(isaac_response[active])
    mujoco_active_diff = np.diff(mujoco_response[active])
    response_error = isaac_response[active] - mujoco_response[active]
    isaac_torque_rms = float(isaac_step_result["metrics"]["selected_joint_effort_rms_nm"])
    mujoco_torque_rms = float(mujoco_step_result["metrics"]["selected_joint_effort_rms_nm"])

    return {
        "step_case_id": isaac_step_result["case_id"],
        "zero_case_id": isaac_zero_result["case_id"],
        "step_trace_sha256": step_trace_metadata["trace_sha256"],
        "zero_trace_sha256": zero_trace_metadata["trace_sha256"],
        "matrix_sha256": isaac_step_result["matrix_sha256"],
        "selected_joint_name": next(iter(selected_names)),
        "commanded_joint_delta_rad": command_delta,
        "identity_and_time_alignment_pass": True,
        "all_safety_envelopes_passed": all(
            bool(result["case_validation"]["safety_envelope_passed"]) for result in results
        ),
        "baseline_corrected_response_direction_consistent": bool(
            command_sign * isaac_end > 0.0 and command_sign * mujoco_end > 0.0
        ),
        "isaac_baseline_corrected_end_response_rad": isaac_end,
        "mujoco_baseline_corrected_end_response_rad": mujoco_end,
        "isaac_baseline_corrected_gain": isaac_gain,
        "mujoco_baseline_corrected_gain": mujoco_gain,
        "gain_symmetric_difference": _symmetric_difference(isaac_gain, mujoco_gain),
        "active_response_trajectory_rmse_rad": float(
            np.sqrt(np.mean(response_error * response_error))
        ),
        "active_response_trajectory_nrmse_by_command": float(
            np.sqrt(np.mean(response_error * response_error)) / abs(command_delta)
        ),
        "isaac_response_end_window_slope_radps": _slope(time_s[tail], isaac_response[tail]),
        "mujoco_response_end_window_slope_radps": _slope(time_s[tail], mujoco_response[tail]),
        "isaac_zero_end_window_slope_radps": float(
            isaac_zero_result["metrics"]["end_window_slope_radps"]
        ),
        "mujoco_zero_end_window_slope_radps": float(
            mujoco_zero_result["metrics"]["end_window_slope_radps"]
        ),
        "isaac_active_monotonic_fraction": float(
            np.mean(command_sign * isaac_active_diff >= 0.0) if isaac_active_diff.size else 1.0
        ),
        "mujoco_active_monotonic_fraction": float(
            np.mean(command_sign * mujoco_active_diff >= 0.0) if mujoco_active_diff.size else 1.0
        ),
        "selected_effort_rms_relative_difference": _relative_difference(
            isaac_torque_rms, mujoco_torque_rms
        ),
        "classification_color": classification_color,
        "difference_labels": sorted(set(difference_labels)),
        "classification_rationale": rationale.strip(),
        "classification_frozen": False,
        "unexplained_blocks_stand": "unexplained" in difference_labels,
        "automatic_promotion": False,
    }
