#!/usr/bin/env python3
"""Build and validate deterministic A3 Base Phase 0 calibration artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _case(
    case_id: str,
    stage: str,
    category: str,
    case_family: str,
    repeat_index: int,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "stage": stage,
        "category": category,
        "case_family": case_family,
        "repeat_index": repeat_index,
        "inputs": inputs,
    }


def build_matrix(contracts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    base = contracts["base_policy_contract.json"]
    strike = contracts["strike_policy_contract.json"]
    composer = contracts["command_composer_contract.json"]
    action = contracts["action_schema.json"]
    calibration = contracts["calibration_contract.json"]
    response_protocol = calibration["response_metric_protocol"]
    repeats = int(calibration["repeats_per_case"])
    cases: list[dict[str, Any]] = []

    basis_inputs = (
        ("vx_pos", [0.10, 0.0, 0.0, 1.0684, 0.0]),
        ("vx_neg", [-0.10, 0.0, 0.0, 1.0684, 0.0]),
        ("vy_pos", [0.0, 0.10, 0.0, 1.0684, 0.0]),
        ("vy_neg", [0.0, -0.10, 0.0, 1.0684, 0.0]),
        ("yaw_pos", [0.0, 0.0, 0.20, 1.0684, 0.0]),
        ("yaw_neg", [0.0, 0.0, -0.20, 1.0684, 0.0]),
    )
    for label, command in basis_inputs:
        for repeat in range(repeats):
            cases.append(
                _case(
                    f"basis__{label}__r{repeat + 1:02d}",
                    "command_basis",
                    "command_basis",
                    "command_basis",
                    repeat,
                    {
                        "base_command": command,
                        "duration_s": 2.0,
                        "intervention_active": False,
                    },
                )
            )

    action_names = list(action["action_joint_names"])
    step_levels = (
        (0.10, "base_action_step_low_amplitude"),
        (0.25, "base_action_step_medium_amplitude"),
    )
    nominal_strike = [0.0, 0.0, 0.3, -0.12, 0.0, 0.8, 0.0, 0.0, 0.0]
    for joint_name in action_names:
        for repeat in range(repeats):
            cases.append(
                _case(
                    f"zero__{joint_name}__r{repeat + 1:02d}",
                    "joint_zero_baseline",
                    "joint_zero_baseline",
                    "action_zero",
                    repeat,
                    {
                        "selected_joint_name": joint_name,
                        "base_action": [0.0] * len(action_names),
                        "strike_q_reference": nominal_strike,
                        "plant_constraint": "single_joint_fixture_v1",
                        "pre_hold_s": response_protocol["pre_hold_s"],
                        "step_hold_s": response_protocol["active_hold_s"],
                        "post_hold_s": response_protocol["post_hold_s"],
                        "end_window_s": response_protocol["end_window_s"],
                        "target_transport": "zero_order_hold",
                    },
                )
            )
    for amplitude, stage in step_levels:
        for joint_index, joint_name in enumerate(action_names):
            for sign_label, signed_amplitude in (("pos", amplitude), ("neg", -amplitude)):
                vector = [0.0] * len(action_names)
                vector[joint_index] = signed_amplitude
                for repeat in range(repeats):
                    cases.append(
                        _case(
                            f"step__a{amplitude:.2f}__{joint_name}__{sign_label}__r{repeat + 1:02d}",
                            stage,
                            "base_action_step",
                            "action_low" if amplitude == 0.10 else "action_medium",
                            repeat,
                            {
                                "selected_joint_name": joint_name,
                                "base_action": vector,
                                "strike_q_reference": nominal_strike,
                                "plant_constraint": "single_joint_fixture_v1",
                                "pre_hold_s": response_protocol["pre_hold_s"],
                                "step_hold_s": response_protocol["active_hold_s"],
                                "post_hold_s": response_protocol["post_hold_s"],
                                "end_window_s": response_protocol["end_window_s"],
                                "target_transport": "zero_order_hold",
                            },
                        )
                    )

    for strike_pitch in (-0.10, 0.0, 0.10):
        zero_strike_reference = [
            0.0, strike_pitch, 0.3, -0.12, 0.0, 0.8, 0.0, 0.0, 0.0
        ]
        for repeat in range(repeats):
            cases.append(
                _case(
                    f"waist_pitch_zero__strike_{strike_pitch:+.2f}__r{repeat + 1:02d}",
                    "waist_pitch_working_point_zero_baseline",
                    "joint_zero_baseline",
                    "working_point_zero",
                    repeat,
                    {
                        "selected_joint_name": "waist_pitch_joint",
                        "base_action": [0.0] * len(action_names),
                        "strike_q_reference": zero_strike_reference,
                        "plant_constraint": "single_joint_fixture_v1",
                        "pre_hold_s": response_protocol["pre_hold_s"],
                        "step_hold_s": response_protocol["active_hold_s"],
                        "post_hold_s": response_protocol["post_hold_s"],
                        "end_window_s": response_protocol["end_window_s"],
                        "target_transport": "zero_order_hold",
                    },
                )
            )
        for residual_action in (-0.25, 0.25):
            vector = [0.0] * len(action_names)
            vector[-1] = residual_action
            strike_reference = [
                0.0, strike_pitch, 0.3, -0.12, 0.0, 0.8, 0.0, 0.0, 0.0
            ]
            for repeat in range(repeats):
                cases.append(
                    _case(
                        (
                            f"waist_pitch__strike_{strike_pitch:+.2f}__"
                            f"base_{residual_action:+.2f}__r{repeat + 1:02d}"
                        ),
                        "waist_pitch_residual_low_amplitude",
                        "waist_pitch_residual",
                        "waist_composition" if strike_pitch == 0.0 else "working_point",
                        repeat,
                        {
                            "selected_joint_name": "waist_pitch_joint",
                            "base_action": vector,
                            "strike_q_reference": strike_reference,
                            "plant_constraint": "single_joint_fixture_v1",
                            "pre_hold_s": response_protocol["pre_hold_s"],
                            "step_hold_s": response_protocol["active_hold_s"],
                            "post_hold_s": response_protocol["post_hold_s"],
                            "end_window_s": response_protocol["end_window_s"],
                            "target_transport": "zero_order_hold",
                        },
                    )
                )

    transport_joints = (
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    for physics_rate in calibration["candidate_physics_rates_hz"]:
        for mode in ("zero_order_hold", "linear_substep_interpolation"):
            for joint_name in transport_joints:
                vector = [0.0] * len(action_names)
                vector[action_names.index(joint_name)] = 0.25
                for repeat in range(repeats):
                    cases.append(
                        _case(
                            (
                                f"transport__{int(physics_rate)}hz__{mode}__"
                                f"{joint_name}__r{repeat + 1:02d}"
                            ),
                            "target_transport",
                            "target_transport",
                            "transport",
                            repeat,
                            {
                                "selected_joint_name": joint_name,
                                "physics_rate_hz": float(physics_rate),
                                "policy_rate_hz": float(calibration["policy_rate_hz"]),
                                "target_transport": mode,
                                "plant_constraint": "single_joint_fixture_v1",
                                "base_action": vector,
                                "strike_q_reference": nominal_strike,
                                "step_hold_s": 0.4,
                                "end_window_s": 0.1,
                                "waveform": "one_policy_tick_ramp_then_hold_0.4s",
                            },
                        )
                    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "calibration_contract_id": calibration["calibration_contract_id"],
        "base_policy_contract_id": base["base_policy_contract_id"],
        "strike_policy_contract_id": strike["strike_policy_contract_id"],
        "command_composer_contract_id": composer["command_composer_contract_id"],
        "joint_order_sha256": composer["joint_order_sha256"],
        "action_joint_order_sha256": composer["base_action_joint_order_sha256"],
        "contract_payload_sha256": {
            filename: canonical_sha256(contracts[filename])
            for filename in (
                "base_policy_contract.json",
                "strike_policy_contract.json",
                "command_composer_contract.json",
                "action_schema.json",
                "calibration_contract.json",
                "command_trace_schema.json",
            )
        },
        "case_count": len(cases),
        "cases": cases,
    }
    payload["matrix_sha256"] = canonical_sha256(payload)
    return payload


def validate_matrix(
    matrix: Mapping[str, Any], contracts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    calibration = contracts["calibration_contract.json"]
    composer = contracts["command_composer_contract.json"]
    action = contracts["action_schema.json"]
    expected_hash = matrix.get("matrix_sha256")
    without_hash = dict(matrix)
    without_hash.pop("matrix_sha256", None)
    if expected_hash != canonical_sha256(without_hash):
        raise ValueError("calibration matrix hash mismatch")
    cases = matrix.get("cases")
    if not isinstance(cases, list) or matrix.get("case_count") != len(cases):
        raise ValueError("calibration case_count mismatch")
    case_ids = [case.get("case_id") for case in cases]
    if len(set(case_ids)) != len(case_ids) or not all(
        isinstance(case_id, str) and case_id for case_id in case_ids
    ):
        raise ValueError("calibration case IDs must be unique non-empty strings")
    categories = {case.get("category") for case in cases}
    if categories != set(calibration["required_categories"]):
        raise ValueError("calibration categories mismatch")
    allowed_families = {
        "command_basis",
        "action_zero",
        "action_low",
        "action_medium",
        "waist_composition",
        "working_point",
        "working_point_zero",
        "transport",
    }
    families = {case.get("case_family") for case in cases}
    if families != allowed_families:
        raise ValueError("calibration case families mismatch")
    if matrix.get("command_composer_contract_id") != composer.get(
        "command_composer_contract_id"
    ):
        raise ValueError("calibration Composer ID mismatch")
    if matrix.get("action_joint_order_sha256") != action.get(
        "action_joint_order_sha256"
    ):
        raise ValueError("calibration action order mismatch")
    expected_contract_hashes = {
        filename: canonical_sha256(contracts[filename])
        for filename in (
            "base_policy_contract.json",
            "strike_policy_contract.json",
            "command_composer_contract.json",
            "action_schema.json",
            "calibration_contract.json",
            "command_trace_schema.json",
        )
    }
    if matrix.get("contract_payload_sha256") != expected_contract_hashes:
        raise ValueError("calibration contract payload hash mismatch")
    repeats = int(calibration["repeats_per_case"])
    payload_protocol = calibration["command_payload_protocol"]
    native_runners = (
        calibration["native_mujoco_runner"],
        calibration["native_isaac_runner"],
    )
    grouped: dict[str, set[int]] = {}
    for case in cases:
        base_id = str(case["case_id"]).rsplit("__r", 1)[0]
        grouped.setdefault(base_id, set()).add(int(case["repeat_index"]))
        if case["category"] in payload_protocol[
            "direct_robotio_replay_supported_categories"
        ]:
            inputs = case.get("inputs", {})
            if inputs.get("target_transport") != payload_protocol[
                "direct_robotio_replay_supported_transport"
            ]:
                raise ValueError("direct replay case transport mismatch")
            if float(inputs.get("pre_hold_s", -1.0)) != float(
                payload_protocol["pre_hold_s"]
            ) or float(inputs.get("post_hold_s", -1.0)) != float(
                payload_protocol["post_hold_s"]
            ):
                raise ValueError("direct replay case hold timing mismatch")
        for native_runner in native_runners:
            if case["category"] in native_runner["approved_categories"] and case.get(
                "inputs", {}
            ).get("plant_constraint") != native_runner["plant_constraint"]:
                raise ValueError("native runner plant constraint mismatch")
    if any(indices != set(range(repeats)) for indices in grouped.values()):
        raise ValueError("calibration repeat coverage mismatch")
    return {
        "case_count": len(cases),
        "logical_case_count": len(grouped),
        "categories": sorted(categories),
        "matrix_sha256": expected_hash,
    }


def validate_result_artifact(
    artifact: Mapping[str, Any],
    matrix: Mapping[str, Any],
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validate_matrix(matrix, contracts)
    if artifact.get("matrix_sha256") != matrix.get("matrix_sha256"):
        raise ValueError("result artifact matrix hash mismatch")
    results = artifact.get("results")
    if not isinstance(results, list):
        raise ValueError("calibration results must be a list")
    expected_ids = {case["case_id"] for case in matrix["cases"]}
    actual_ids = [result.get("case_id") for result in results]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError("calibration result coverage mismatch")
    category_by_id = {
        case["case_id"]: case["category"] for case in matrix["cases"]
    }
    violations: list[str] = []
    for result in results:
        summary = validate_case_result(
            result, category_by_id[result["case_id"]], contracts
        )
        violations.extend(summary["violations"])
    return {
        "result_count": len(results),
        "safety_envelope_passed": not violations,
        "violations": violations,
        "automatic_promotion": False,
    }


def validate_case_result(
    result: Mapping[str, Any],
    category: str,
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one pilot result without implying complete matrix coverage."""

    calibration = contracts["calibration_contract.json"]
    category_metrics = calibration["required_category_metrics"]
    if category not in category_metrics:
        raise ValueError(f"unknown calibration result category: {category}")
    case_id = result.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case result requires a non-empty case_id")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"missing metrics for {case_id}")
    common = set(calibration["required_common_result_metrics"])
    required = common | set(category_metrics[category])
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"missing metrics for {case_id}: {missing}")
    numeric_keys = common - {"safety_stop"}
    if not all(math.isfinite(float(metrics[key])) for key in numeric_keys):
        raise ValueError(f"non-finite common metric for {case_id}")
    if category == "command_basis":
        displacement = metrics["displacement_heading_xyz_m"]
        if (
            not isinstance(displacement, list)
            or len(displacement) != 3
            or not all(math.isfinite(float(value)) for value in displacement)
        ):
            raise ValueError(f"invalid command-basis displacement for {case_id}")
        if not math.isfinite(float(metrics["yaw_delta_rad"])):
            raise ValueError(f"invalid command-basis yaw delta for {case_id}")
        if int(metrics["observed_command_axis_sign"]) not in {-1, 0, 1}:
            raise ValueError(f"invalid observed command sign for {case_id}")
    elif category in {
        "joint_zero_baseline",
        "base_action_step",
        "waist_pitch_residual",
    }:
        nonnumeric = {
            "composer_residual_clip_hit",
            "constraint_reaction_available",
            "target_band_reached_and_held",
        }
        if not all(
            math.isfinite(float(metrics[key]))
            for key in set(category_metrics[category]) - nonnumeric
        ):
            raise ValueError(f"non-finite step metric for {case_id}")
    else:
        if metrics["transport_mode"] not in {
            "zero_order_hold",
            "linear_substep_interpolation",
        }:
            raise ValueError(f"invalid transport mode for {case_id}")
        if not all(
            math.isfinite(float(metrics[key]))
            for key in set(category_metrics[category])
            - {"transport_mode", "constraint_reaction_available"}
        ):
            raise ValueError(f"non-finite transport metric for {case_id}")
    envelope = calibration["safety_envelope"]
    violations: list[str] = []
    if int(metrics["nonfinite_count"]) > envelope["max_nonfinite_count"]:
        violations.append(f"{case_id}: nonfinite")
    if bool(metrics["safety_stop"]) and not envelope["allow_safety_stop"]:
        violations.append(f"{case_id}: safety_stop")
    if int(metrics["forbidden_contact_count"]) > envelope[
        "max_forbidden_contact_count"
    ]:
        violations.append(f"{case_id}: forbidden_contact")
    if int(metrics["joint_limit_hit_count"]) > envelope["max_joint_limit_hit_count"]:
        violations.append(f"{case_id}: joint_limit")
    if float(metrics["max_tilt_deg"]) > envelope["max_tilt_deg"]:
        violations.append(f"{case_id}: tilt")
    if float(metrics["min_pelvis_height_m"]) < envelope["min_pelvis_height_m"]:
        violations.append(f"{case_id}: height")
    return {
        "case_id": case_id,
        "category": category,
        "safety_envelope_passed": not violations,
        "violations": violations,
        "matrix_coverage_complete": False,
        "automatic_promotion": False,
    }
