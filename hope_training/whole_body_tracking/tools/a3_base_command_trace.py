#!/usr/bin/env python3
"""Build immutable causal command traces shared by Isaac and MuJoCo runners."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

import a3_base_calibration as calibration
import a3_base_contract as contract


ARRAY_ORDER = (
    "physics_step_start_time_s",
    "policy_sample_time_s",
    "command_publish_time_s",
    "first_effective_physics_step_time_s",
    "state_sample_time_s",
    "metric_timestamp_s",
    "base_action",
    "strike_q_reference",
    "base_residual_rad",
    "composed_policy_target_rad",
    "composed_target_rad",
    "joint_names",
    "selected_joint_name",
    "transport_mode",
    "physics_dt_s",
    "policy_dt_s",
)


def trace_sha256(trace: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ARRAY_ORDER:
        if name not in trace:
            raise ValueError(f"command trace missing {name}")
        array = np.asarray(trace[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii") + b"\0")
        if array.dtype.kind in {"U", "S"}:
            for value in array.reshape(-1).tolist():
                digest.update(str(value).encode("utf-8") + b"\0")
        else:
            numeric = np.asarray(array, dtype=array.dtype.newbyteorder("<"), order="C")
            digest.update(numeric.tobytes(order="C"))
    return digest.hexdigest()


def _logical_case_payload(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id_without_repeat": str(case["case_id"]).rsplit("__r", 1)[0],
        "stage": case["stage"],
        "category": case["category"],
        "case_family": case["case_family"],
        "inputs": case["inputs"],
    }


def build_trace(
    case: Mapping[str, Any],
    contracts: Mapping[str, Mapping[str, Any]],
    physics_rate_hz: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    category = str(case.get("category"))
    if category not in {
        "joint_zero_baseline",
        "base_action_step",
        "waist_pitch_residual",
        "target_transport",
    }:
        raise ValueError(f"shared joint trace does not support {category}")
    inputs = case["inputs"]
    calibration_contract = contracts["calibration_contract.json"]
    response_protocol = calibration_contract["response_metric_protocol"]
    composer = contracts["command_composer_contract.json"]
    action_schema = contracts["action_schema.json"]
    policy_rate_hz = float(calibration_contract["policy_rate_hz"])
    requested_rate = float(
        physics_rate_hz
        if physics_rate_hz is not None
        else inputs.get("physics_rate_hz", 200.0)
    )
    substeps = requested_rate / policy_rate_hz
    if not math.isfinite(substeps) or substeps < 1.0 or not float(substeps).is_integer():
        raise ValueError("physics rate must be an integer multiple of policy rate")
    substeps_per_command = int(substeps)
    physics_dt = 1.0 / requested_rate
    policy_dt = 1.0 / policy_rate_hz
    transport = str(inputs.get("target_transport", "zero_order_hold"))
    if transport not in {"zero_order_hold", "linear_substep_interpolation"}:
        raise ValueError(f"unsupported transport mode: {transport}")

    action = np.asarray(inputs["base_action"], dtype=np.float64)
    if action.shape != (14,) or not np.all(np.isfinite(action)):
        raise ValueError("trace Base action must be finite [14]")
    selected = np.flatnonzero(np.abs(action) > 0.0)
    if category == "joint_zero_baseline":
        if selected.size != 0:
            raise ValueError("zero-baseline trace requires an all-zero Base action")
        selected_joint = str(inputs.get("selected_joint_name", ""))
        if selected_joint not in action_schema["action_joint_names"]:
            raise ValueError("zero-baseline trace requires a valid selected joint")
    else:
        if selected.size != 1:
            raise ValueError("step trace requires exactly one selected Base action")
        selected_joint = action_schema["action_joint_names"][int(selected[0])]
        declared_selected = inputs.get("selected_joint_name")
        if declared_selected is not None and declared_selected != selected_joint:
            raise ValueError("declared selected joint differs from non-zero Base action")
    strike = np.asarray(inputs["strike_q_reference"], dtype=np.float64)
    if strike.shape != (9,) or not np.all(np.isfinite(strike)):
        raise ValueError("trace Strike reference must be finite [9]")
    baseline_action = np.zeros(14, dtype=np.float64)
    baseline_cmd = contract.compose_command(composer, baseline_action, strike)
    excited_cmd = contract.compose_command(composer, action, strike)
    baseline_target = np.asarray(baseline_cmd["q_des"], dtype=np.float64)
    excited_target = np.asarray(excited_cmd["q_des"], dtype=np.float64)

    protocol = calibration_contract["command_payload_protocol"]
    pre_s = float(inputs.get("pre_hold_s", protocol["pre_hold_s"]))
    hold_s = float(inputs.get("step_hold_s", 0.4))
    post_s = float(inputs.get("post_hold_s", protocol["post_hold_s"]))
    for label, duration in (("pre", pre_s), ("hold", hold_s), ("post", post_s)):
        if duration <= 0.0 or not math.isclose(
            duration * policy_rate_hz, round(duration * policy_rate_hz), abs_tol=1.0e-12
        ):
            raise ValueError(f"{label} duration must align to policy ticks")
    total_s = pre_s + hold_s + post_s
    step_count = int(round(total_s * requested_rate))
    start = np.arange(step_count, dtype=np.float64) * physics_dt
    policy_tick = np.floor(np.arange(step_count) / substeps_per_command).astype(np.int64)
    policy_time = policy_tick.astype(np.float64) * policy_dt
    active = (policy_time >= pre_s) & (policy_time < pre_s + hold_s)
    policy_action = np.where(active[:, None], action[None, :], baseline_action[None, :])
    policy_target = np.where(
        active[:, None], excited_target[None, :], baseline_target[None, :]
    )
    previous_target = np.empty_like(policy_target)
    previous_target[0] = policy_target[0]
    for index in range(1, step_count):
        if policy_tick[index] != policy_tick[index - 1]:
            previous_target[index] = policy_target[index - 1]
        else:
            previous_target[index] = previous_target[index - 1]
    if transport == "zero_order_hold":
        transported = policy_target.copy()
    else:
        substep_index = np.arange(step_count) % substeps_per_command
        alpha = (substep_index.astype(np.float64) + 1.0) / substeps_per_command
        transported = previous_target + alpha[:, None] * (policy_target - previous_target)
    residual_scale = float(composer["base_action_scale_rad"][-1])
    residual_limit = float(composer["waist_pitch_residual_limit_rad"])
    residual = np.clip(policy_action[:, -1] * residual_scale, -residual_limit, residual_limit)
    trace = {
        "physics_step_start_time_s": start,
        "policy_sample_time_s": policy_time.copy(),
        "command_publish_time_s": policy_time.copy(),
        "first_effective_physics_step_time_s": policy_time.copy(),
        "state_sample_time_s": start + physics_dt,
        "metric_timestamp_s": start + physics_dt,
        "base_action": policy_action,
        "strike_q_reference": np.tile(strike, (step_count, 1)),
        "base_residual_rad": residual,
        "composed_policy_target_rad": policy_target,
        "composed_target_rad": transported,
        "joint_names": np.asarray(composer["backend_joint_names"]),
        "selected_joint_name": np.asarray([selected_joint]),
        "transport_mode": np.asarray([transport]),
        "physics_dt_s": np.asarray([physics_dt], dtype=np.float64),
        "policy_dt_s": np.asarray([policy_dt], dtype=np.float64),
    }
    digest = trace_sha256(trace)
    metadata = {
        "schema_version": 1,
        "command_trace_schema_id": contracts["command_trace_schema.json"][
            "command_trace_schema_id"
        ],
        "case_id": case["case_id"],
        "stage": case["stage"],
        "category": category,
        "case_family": case["case_family"],
        "logical_case_definition_sha256": calibration.canonical_sha256(
            _logical_case_payload(case)
        ),
        "trace_sha256": digest,
        "physics_rate_hz": requested_rate,
        "policy_rate_hz": policy_rate_hz,
        "substeps_per_policy_command": substeps_per_command,
        "transport_mode": transport,
        "selected_joint_name": selected_joint,
        "plant_constraint": inputs["plant_constraint"],
        "metric_window": {
            "baseline_end_s": pre_s,
            "active_start_s": pre_s,
            "active_end_s": pre_s + hold_s,
            "end_window_s": float(inputs.get("end_window_s", min(0.1, hold_s))),
            "settling_min_tolerance_rad": float(
                response_protocol["settling_min_tolerance_rad"]
            ),
            "settling_relative_tolerance": float(
                response_protocol["settling_relative_tolerance"]
            ),
        },
        "composer_residual_clip_hit": bool(
            abs(float(action[-1]) * residual_scale) > residual_limit
        ),
        "causal_interpolation": True,
        "future_policy_target_accessed": False,
        "runner_mutation_allowed": False,
        "hardware_execution_approved": False,
    }
    return trace, metadata


def validate_trace(
    trace: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(trace) != ARRAY_ORDER:
        raise ValueError("command trace array order/schema mismatch")
    digest = trace_sha256(trace)
    if metadata.get("trace_sha256") != digest:
        raise ValueError("command trace hash mismatch")
    if metadata.get("command_trace_schema_id") != contracts[
        "command_trace_schema.json"
    ]["command_trace_schema_id"]:
        raise ValueError("command trace schema ID mismatch")
    count = len(trace["physics_step_start_time_s"])
    expected_shapes = {
        "base_action": (count, 14),
        "strike_q_reference": (count, 9),
        "composed_policy_target_rad": (count, 31),
        "composed_target_rad": (count, 31),
    }
    for name, shape in expected_shapes.items():
        if np.asarray(trace[name]).shape != shape:
            raise ValueError(f"invalid command trace shape for {name}")
    physics_dt = float(trace["physics_dt_s"][0])
    policy_dt = float(trace["policy_dt_s"][0])
    if not math.isclose(
        policy_dt / physics_dt,
        int(round(policy_dt / physics_dt)),
        abs_tol=1.0e-12,
    ):
        raise ValueError("trace policy/physics ratio is not integral")
    if not np.array_equal(trace["state_sample_time_s"], trace["metric_timestamp_s"]):
        raise ValueError("metric timestamp must equal post-step state sample time")
    metric_window = metadata.get("metric_window", {})
    active_duration = float(metric_window.get("active_end_s", 0.0)) - float(
        metric_window.get("active_start_s", 0.0)
    )
    if not 0.0 < float(metric_window.get("end_window_s", 0.0)) <= active_duration:
        raise ValueError("trace end window must fit inside active window")
    if np.any(trace["command_publish_time_s"] > trace["physics_step_start_time_s"] + 1.0e-12):
        raise ValueError("trace uses a future command")
    return {
        "trace_sha256": digest,
        "step_count": count,
        "physics_rate_hz": 1.0 / physics_dt,
        "policy_rate_hz": 1.0 / policy_dt,
        "transport_mode": str(trace["transport_mode"][0]),
    }


def case_instance_sha256(
    *,
    trace_metadata: Mapping[str, Any],
    model_sha256: str,
    fixture_contract: Mapping[str, Any],
    initial_q_rad: list[float],
    kp: list[float],
    kd: list[float],
) -> str:
    return calibration.canonical_sha256(
        {
            "trace_sha256": trace_metadata["trace_sha256"],
            "logical_case_definition_sha256": trace_metadata[
                "logical_case_definition_sha256"
            ],
            "model_sha256": model_sha256,
            "fixture_contract_sha256": calibration.canonical_sha256(
                fixture_contract
            ),
            "initial_q_rad": initial_q_rad,
            "kp": kp,
            "kd": kd,
            "physics_rate_hz": trace_metadata["physics_rate_hz"],
            "policy_rate_hz": trace_metadata["policy_rate_hz"],
            "transport_mode": trace_metadata["transport_mode"],
        }
    )
