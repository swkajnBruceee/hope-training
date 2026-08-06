#!/usr/bin/env python3
"""Dependency-free A3 Base/Strike/Composer contract helpers.

This module intentionally avoids Isaac Lab, NumPy, and the deployment SDK so
the Phase 0 contract can be validated on a plain Python installation and in CI.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


CONTRACT_FILENAMES = (
    "base_policy_contract.json",
    "strike_policy_contract.json",
    "command_composer_contract.json",
    "actor_observation_schema.json",
    "critic_observation_schema.json",
    "action_schema.json",
    "calibration_contract.json",
    "stand_fixture_gate_v1.json",
    "command_trace_schema.json",
    "golden_composer_vectors.json",
)


def ordered_name_sha256(names: Sequence[str]) -> str:
    """Hash an ordered name list using an unambiguous NUL separator."""

    digest = hashlib.sha256()
    for name in names:
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError(f"invalid ordered name: {name!r}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_contracts(contract_dir: Path) -> dict[str, dict[str, Any]]:
    contract_dir = contract_dir.expanduser().resolve()
    result: dict[str, dict[str, Any]] = {}
    for filename in CONTRACT_FILENAMES:
        path = contract_dir / filename
        if not path.is_file():
            raise ValueError(f"missing contract file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"contract root must be an object: {path}")
        result[filename] = payload
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_vector(payload: Mapping[str, Any], key: str, size: int) -> list[float]:
    values = payload.get(key)
    _require(isinstance(values, list), f"{key} must be a list")
    _require(len(values) == size, f"{key} must contain {size} values, got {len(values)}")
    output = [float(value) for value in values]
    _require(all(math.isfinite(value) for value in output), f"{key} contains non-finite values")
    return output


def _field_dimension(fields: Any, label: str) -> int:
    _require(isinstance(fields, list), f"{label} must be a list")
    total = 0
    seen: set[str] = set()
    for field in fields:
        _require(isinstance(field, dict), f"{label} entries must be objects")
        name = field.get("name")
        dimension = field.get("dimension")
        _require(isinstance(name, str) and name, f"{label} field has invalid name")
        _require(name not in seen, f"duplicate {label} field: {name}")
        _require(isinstance(dimension, int) and dimension > 0, f"invalid dimension for {name}")
        seen.add(name)
        total += dimension
    return total


def validate_contracts(contracts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Validate cross-file structure and return the frozen interface summary."""

    base = contracts["base_policy_contract.json"]
    strike = contracts["strike_policy_contract.json"]
    composer = contracts["command_composer_contract.json"]
    actor = contracts["actor_observation_schema.json"]
    critic = contracts["critic_observation_schema.json"]
    action = contracts["action_schema.json"]
    calibration = contracts["calibration_contract.json"]
    trace_schema = contracts["command_trace_schema.json"]
    golden = contracts["golden_composer_vectors.json"]

    backend_names = composer.get("backend_joint_names")
    policy_names = composer.get("policy_joint_names")
    base_action_names = composer.get("base_action_joint_names")
    strike_names = composer.get("strike_reference_joint_names")
    for names, size, label in (
        (backend_names, 31, "backend_joint_names"),
        (policy_names, 29, "policy_joint_names"),
        (base_action_names, 14, "base_action_joint_names"),
        (strike_names, 9, "strike_reference_joint_names"),
    ):
        _require(isinstance(names, list), f"{label} must be a list")
        _require(len(names) == size, f"{label} must contain {size} entries")
        _require(len(set(names)) == size, f"{label} contains duplicate entries")

    expected_hashes = {
        "joint_order_sha256": ordered_name_sha256(backend_names),
        "policy_joint_order_sha256": ordered_name_sha256(policy_names),
        "base_action_joint_order_sha256": ordered_name_sha256(base_action_names),
        "strike_reference_joint_order_sha256": ordered_name_sha256(strike_names),
    }
    for key, expected in expected_hashes.items():
        _require(composer.get(key) == expected, f"composer {key} mismatch")

    _require(base.get("joint_order_sha256") == expected_hashes["joint_order_sha256"], "base backend hash mismatch")
    _require(
        base.get("policy_joint_order_sha256") == expected_hashes["policy_joint_order_sha256"],
        "base policy hash mismatch",
    )
    _require(
        base.get("action_joint_order_sha256") == expected_hashes["base_action_joint_order_sha256"],
        "base action hash mismatch",
    )
    _require(strike.get("reference_joint_names") == strike_names, "Strike and Composer reference orders differ")
    _require(
        strike.get("reference_joint_order_sha256") == expected_hashes["strike_reference_joint_order_sha256"],
        "strike hash mismatch",
    )
    _require(action.get("action_joint_names") == base_action_names, "Action and Composer orders differ")
    _require(
        action.get("action_joint_order_sha256") == expected_hashes["base_action_joint_order_sha256"],
        "action hash mismatch",
    )

    expected_policy_names = [
        name for name in backend_names if name not in {"head_yaw_joint", "head_pitch_joint"}
    ]
    _require(policy_names == expected_policy_names, "29-DOF policy view must be backend order without head")
    expected_strike_names = ["waist_yaw_joint", "waist_pitch_joint"] + backend_names[12:19]
    _require(strike_names == expected_strike_names, "Strike v2 order mismatch")
    expected_action_names = backend_names[19:31] + ["waist_roll_joint", "waist_pitch_joint"]
    _require(base_action_names == expected_action_names, "Base action order mismatch")

    mirror = action.get("locomotion_only_mirror")
    _require(isinstance(mirror, dict), "locomotion-only action mirror is required")
    mirror_indices = mirror.get("source_index_for_output")
    mirror_signs = mirror.get("sign_for_output")
    _require(isinstance(mirror_indices, list) and len(mirror_indices) == 14, "mirror index size mismatch")
    _require(isinstance(mirror_signs, list) and len(mirror_signs) == 14, "mirror sign size mismatch")
    _require(sorted(mirror_indices) == list(range(14)), "mirror indices must be a permutation")
    for output_index, source_index in enumerate(mirror_indices):
        _require(mirror_indices[source_index] == output_index, "mirror index map is not an involution")
        _require(
            float(mirror_signs[output_index]) * float(mirror_signs[source_index]) == 1.0,
            "mirror signs are not an involution",
        )
    command_mirror = base.get("locomotion_only_command_mirror_sign")
    _require(command_mirror == [1.0, -1.0, -1.0, 1.0, 1.0], "command mirror sign mismatch")

    ownership = composer.get("ownership_by_backend_joint")
    _require(isinstance(ownership, list) and len(ownership) == 31, "ownership must cover all 31 joints")
    allowed_owners = {
        "base",
        "strike",
        "strike_plus_bounded_base_residual",
        "head_baseline",
        "left_arm_baseline",
    }
    _require(set(ownership) <= allowed_owners, "unknown Composer owner")
    _require(ownership[1] == "base", "waist roll must be Base-owned")
    _require(ownership[2] == "strike_plus_bounded_base_residual", "waist pitch ownership mismatch")

    for key in ("nominal_q_rad", "kp", "kd", "joint_lower_rad", "joint_upper_rad"):
        _finite_vector(composer, key, 31)
    scales = _finite_vector(composer, "base_action_scale_rad", 14)
    expected_scales = [float(value) for value in action.get("candidate_action_scale_rad", [])]
    _require(scales == expected_scales, "action scale files differ")
    lower = _finite_vector(composer, "joint_lower_rad", 31)
    upper = _finite_vector(composer, "joint_upper_rad", 31)
    nominal = _finite_vector(composer, "nominal_q_rad", 31)
    _require(all(lo < hi for lo, hi in zip(lower, upper)), "invalid joint limit interval")
    _require(all(lo <= q <= hi for q, lo, hi in zip(nominal, lower, upper)), "nominal pose exceeds joint limits")

    per_frame = _field_dimension(actor.get("history_fields"), "actor history")
    current = _field_dimension(actor.get("current_fields"), "actor current")
    history_length = actor.get("history_length")
    _require(isinstance(history_length, int) and history_length > 0, "invalid history length")
    _require(actor.get("per_history_frame_dimension") == per_frame, "actor per-frame dimension mismatch")
    _require(actor.get("history_dimension") == per_frame * history_length, "actor history dimension mismatch")
    _require(actor.get("current_dimension") == current, "actor current dimension mismatch")
    actor_total = per_frame * history_length + current
    _require(actor.get("total_dimension") == actor_total, "actor total dimension mismatch")
    privileged = _field_dimension(critic.get("privileged_fields"), "critic privileged")
    _require(critic.get("actor_observation_dimension") == actor_total, "critic actor dimension mismatch")
    _require(critic.get("privileged_dimension") == privileged, "critic privileged dimension mismatch")
    _require(critic.get("total_dimension") == actor_total + privileged, "critic total dimension mismatch")

    forbidden = set(actor.get("explicitly_forbidden_actor_fields", []))
    actor_names = {field["name"] for field in actor["history_fields"] + actor["current_fields"]}
    _require(not (forbidden & actor_names), "privileged field leaked into Actor observation")

    policy_rate = base.get("policy_rate_hz")
    _require(policy_rate == actor.get("policy_rate_hz") == composer.get("policy_rate_hz"), "policy rates differ")
    _require(
        base.get("future_reference_offsets_s") == strike.get("future_reference_offsets_s"),
        "future horizons differ",
    )
    _require(
        golden.get("command_composer_contract_id") == composer.get("command_composer_contract_id"),
        "golden contract ID mismatch",
    )
    required_ids = calibration.get("required_contract_ids")
    _require(isinstance(required_ids, dict), "calibration required_contract_ids must be an object")
    _require(
        required_ids.get("base_policy_contract_id") == base.get("base_policy_contract_id"),
        "calibration Base ID mismatch",
    )
    _require(
        required_ids.get("strike_policy_contract_id") == strike.get("strike_policy_contract_id"),
        "calibration Strike ID mismatch",
    )
    _require(
        required_ids.get("command_composer_contract_id") == composer.get("command_composer_contract_id"),
        "calibration Composer ID mismatch",
    )
    _require(calibration.get("policy_rate_hz") == policy_rate, "calibration policy rate mismatch")
    _require(
        calibration.get("command_trace_schema_id")
        == trace_schema.get("command_trace_schema_id"),
        "calibration command trace schema mismatch",
    )
    response_protocol = calibration.get("response_metric_protocol")
    _require(isinstance(response_protocol, dict), "response metric protocol is required")
    _require(
        float(response_protocol.get("active_hold_s", 0.0)) == 1.0
        and float(response_protocol.get("end_window_s", 0.0)) == 0.2
        and response_protocol.get("steady_state_term_allowed_only_when_settled") is True,
        "response metric protocol mismatch",
    )
    _require(
        trace_schema.get("time_semantics", {}).get("policy_command_is_causal") is True
        and trace_schema.get("transport_semantics", {}).get(
            "future_policy_target_access_allowed"
        )
        is False
        and trace_schema.get("execution_boundary", {}).get(
            "hardware_execution_approved"
        )
        is False,
        "command trace causal/safety boundary mismatch",
    )
    payload_protocol = calibration.get("command_payload_protocol")
    _require(isinstance(payload_protocol, dict), "calibration command payload protocol is required")
    _require(
        payload_protocol.get("joint_command_rate_hz") == policy_rate,
        "calibration command payload rate mismatch",
    )
    _require(
        set(payload_protocol.get("direct_robotio_replay_supported_categories", []))
        == {"base_action_step", "waist_pitch_residual"},
        "direct RobotIO replay category boundary mismatch",
    )
    _require(
        payload_protocol.get("direct_robotio_replay_supported_transport")
        == "zero_order_hold",
        "direct RobotIO replay transport must remain zero-order hold",
    )
    _require(
        payload_protocol.get("single_publisher_required") is True
        and payload_protocol.get("isolated_simulator_required") is True
        and payload_protocol.get("hardware_execution_approved") is False,
        "calibration execution safety boundary mismatch",
    )
    native_runner = calibration.get("native_mujoco_runner")
    _require(isinstance(native_runner, dict), "native MuJoCo runner contract is required")
    _require(
        native_runner.get("approved_categories")
        == [
            "joint_zero_baseline",
            "base_action_step",
            "waist_pitch_residual",
            "target_transport",
        ],
        "native MuJoCo runner categories mismatch",
    )
    _require(
        native_runner.get("plant_constraint") == "single_joint_fixture_v1"
        and native_runner.get("requires_no_ros_aimrt_or_network_transport") is True
        and native_runner.get("automatic_promotion") is False
        and native_runner.get("hardware_execution_approved") is False,
        "native MuJoCo runner safety boundary mismatch",
    )
    qualification = calibration.get("qualification_status")
    _require(isinstance(qualification, dict), "calibration qualification status is required")
    _require(
        qualification.get("fixture_v3_semantics_frozen") is True
        and qualification.get("fixture_runner_qualified") is True
        and qualification.get("fixture_matrix_approved") is False
        and qualification.get("stand_task_approved") is False
        and qualification.get("locomotion_command_approved") is False
        and qualification.get("deployment_approved") is False,
        "calibration qualification gates mismatch",
    )

    stand_gate = contracts.get("stand_fixture_gate_v1.json")
    _require(isinstance(stand_gate, dict), "Stand fixture gate contract is required")
    _require(
        stand_gate.get("stand_fixture_gate_contract_id") == "a3_stand_fixture_gate_v1"
        and stand_gate.get("decision_status")
        == "approved_for_bounded_stand_smoke_only",
        "Stand fixture gate identity mismatch",
    )
    legacy = stand_gate.get("legacy_evidence_contract")
    _require(isinstance(legacy, dict), "Stand gate legacy evidence reference is required")
    _require(
        legacy.get("calibration_contract_id") == calibration.get("calibration_contract_id")
        and legacy.get("calibration_contract_canonical_sha256")
        == canonical_sha256(calibration)
        and legacy.get("legacy_full_339_case_promotion_rule_satisfied") is False,
        "Stand gate legacy evidence reference mismatch",
    )
    approved_scope = stand_gate.get("approved_fixture_scope")
    _require(
        isinstance(approved_scope, dict)
        and approved_scope.get("logical_case_count") == 89
        and approved_scope.get("repeat_count") == 3
        and approved_scope.get("executed_case_count_per_engine") == 267,
        "Stand fixture approval scope mismatch",
    )
    smoke = stand_gate.get("stand_smoke_execution_contract")
    _require(
        isinstance(smoke, dict)
        and smoke.get("simulation_only") is True
        and smoke.get("target_transport") == "zero_order_hold"
        and smoke.get("normalized_action_clip_abs") == 0.25
        and smoke.get("long_training_allowed") is False
        and smoke.get("hardware_execution_allowed") is False,
        "bounded Stand smoke execution contract mismatch",
    )
    current_qualification = stand_gate.get("qualification_status")
    _require(
        isinstance(current_qualification, dict)
        and current_qualification.get("fixture_runner_qualified") is True
        and current_qualification.get("fixture_matrix_approved") is True
        and current_qualification.get("stand_task_approved") is True
        and current_qualification.get("stand_smoke_approved") is True
        and current_qualification.get("stand_long_training_approved") is False
        and current_qualification.get("locomotion_command_approved") is False
        and current_qualification.get("deployment_approved") is False,
        "current Stand qualification gates mismatch",
    )

    training_approved = bool(base.get("training_approved") and composer.get("training_approved"))
    deployment_approved = bool(
        base.get("deployment_approved")
        and composer.get("deployment_approved")
        and strike.get("integration_approved")
    )
    return {
        "backend_dof": len(backend_names),
        "policy_view_dof": len(policy_names),
        "base_action_dof": len(base_action_names),
        "strike_reference_dof": len(strike_names),
        "actor_observation_dimension": actor_total,
        "critic_observation_dimension": actor_total + privileged,
        "policy_rate_hz": composer["policy_rate_hz"],
        "joint_order_sha256": expected_hashes["joint_order_sha256"],
        **current_qualification,
        "training_approved": training_approved,
        "deployment_approved": deployment_approved,
    }


def compose_command(
    composer: Mapping[str, Any],
    base_action: Sequence[float],
    strike_q_reference: Sequence[float],
) -> dict[str, Any]:
    """Compose Base and Strike inputs into one full 31-DOF command."""

    if len(base_action) != 14:
        raise ValueError(f"base_action must contain 14 values, got {len(base_action)}")
    if len(strike_q_reference) != 9:
        raise ValueError(f"strike_q_reference must contain 9 values, got {len(strike_q_reference)}")
    action_values = [float(value) for value in base_action]
    strike_values = [float(value) for value in strike_q_reference]
    if not all(math.isfinite(value) for value in action_values + strike_values):
        raise ValueError("Composer inputs must be finite")

    names = list(composer["backend_joint_names"])
    action_names = list(composer["base_action_joint_names"])
    strike_names = list(composer["strike_reference_joint_names"])
    nominal = _finite_vector(composer, "nominal_q_rad", 31)
    scales = _finite_vector(composer, "base_action_scale_rad", 14)
    lower = _finite_vector(composer, "joint_lower_rad", 31)
    upper = _finite_vector(composer, "joint_upper_rad", 31)
    q_des = nominal.copy()
    action_clip = _finite_vector(composer, "normalized_action_clip", 2)
    clipped_action = [min(max(value, action_clip[0]), action_clip[1]) for value in action_values]
    index = {name: position for position, name in enumerate(names)}

    for action_index, joint_name in enumerate(action_names[:12]):
        sdk_index = index[joint_name]
        q_des[sdk_index] = nominal[sdk_index] + scales[action_index] * clipped_action[action_index]

    waist_roll_index = index["waist_roll_joint"]
    q_des[waist_roll_index] = nominal[waist_roll_index] + scales[12] * clipped_action[12]

    strike_by_name = dict(zip(strike_names, strike_values))
    for joint_name in strike_names:
        if joint_name != "waist_pitch_joint":
            q_des[index[joint_name]] = strike_by_name[joint_name]

    residual = scales[13] * clipped_action[13]
    residual_limit = float(composer["waist_pitch_residual_limit_rad"])
    residual = min(max(residual, -residual_limit), residual_limit)
    waist_pitch_index = index["waist_pitch_joint"]
    q_des[waist_pitch_index] = strike_by_name["waist_pitch_joint"] + residual

    limit_hit = []
    for joint_index, value in enumerate(q_des):
        clipped = min(max(value, lower[joint_index]), upper[joint_index])
        limit_hit.append(clipped != value)
        q_des[joint_index] = clipped

    return {
        "joint_names": names,
        "q_des": q_des,
        "dq_des": [0.0] * 31,
        "tau_ff": [0.0] * 31,
        "kp": _finite_vector(composer, "kp", 31),
        "kd": _finite_vector(composer, "kd", 31),
        "debug": {
            "clipped_base_action": clipped_action,
            "waist_pitch_residual_rad": residual,
            "joint_limit_hit": limit_hit,
        },
    }


def validate_golden_vectors(contracts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    composer = contracts["command_composer_contract.json"]
    golden = contracts["golden_composer_vectors.json"]
    tolerance = float(golden.get("absolute_tolerance", 0.0))
    _require(math.isfinite(tolerance) and tolerance >= 0.0, "invalid golden tolerance")
    names: list[str] = []
    for case in golden.get("cases", []):
        _require(isinstance(case, dict), "golden case must be an object")
        name = case.get("name")
        _require(isinstance(name, str) and name and name not in names, "invalid or duplicate golden case name")
        command = compose_command(composer, case.get("base_action", []), case.get("strike_q_reference", []))
        expected = case.get("expected_q_des")
        _require(
            isinstance(expected, list) and len(expected) == 31,
            f"golden {name} expected_q_des must contain 31 values",
        )
        errors = [abs(actual - float(target)) for actual, target in zip(command["q_des"], expected)]
        _require(max(errors, default=0.0) <= tolerance, f"golden {name} mismatch: max error {max(errors)}")
        names.append(name)
    _require(bool(names), "at least one golden Composer case is required")
    return names


def _joint_semantics_from_urdf(path: Path, backend_names: Sequence[str]) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    wanted = set(backend_names)
    joints: list[dict[str, Any]] = []
    for element in root.findall(".//joint"):
        name = element.get("name")
        if name not in wanted:
            continue
        axis_element = element.find("axis")
        limit_element = element.find("limit")
        _require(axis_element is not None and limit_element is not None, f"URDF joint {name} lacks axis/limit")
        joints.append({
            "name": name,
            "axis": [float(value) for value in axis_element.get("xyz", "").split()],
            "lower": float(limit_element.get("lower", "nan")),
            "upper": float(limit_element.get("upper", "nan")),
            "effort": float(limit_element.get("effort", "nan")),
            "velocity": float(limit_element.get("velocity", "nan")),
        })
    return joints


def _joint_semantics_from_mujoco(path: Path, backend_names: Sequence[str]) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    wanted = set(backend_names)
    joints: list[dict[str, Any]] = []
    for element in root.findall(".//joint"):
        name = element.get("name")
        if name not in wanted:
            continue
        limits = [float(value) for value in element.get("range", "").split()]
        effort = [float(value) for value in element.get("actuatorfrcrange", "").split()]
        joints.append({
            "name": name,
            "axis": [float(value) for value in element.get("axis", "").split()],
            "lower": limits[0],
            "upper": limits[1],
            "effort": max(abs(effort[0]), abs(effort[1])),
        })
    return joints


def _xyz(raw: str | None) -> list[float]:
    values = [float(value) for value in (raw or "0 0 0").split()]
    _require(len(values) == 3, f"expected XYZ/RPY triplet, got {raw!r}")
    return values


def _rpy_matrix(rpy: Sequence[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [sum(left[row][inner] * right[inner][column] for inner in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3)]


def _inertia_invariants(matrix: Sequence[Sequence[float]]) -> list[float]:
    ixx, ixy, ixz = matrix[0]
    _, iyy, iyz = matrix[1]
    _, _, izz = matrix[2]
    trace = ixx + iyy + izz
    second = ixx * iyy + ixx * izz + iyy * izz - ixy * ixy - ixz * ixz - iyz * iyz
    determinant = (
        ixx * (iyy * izz - iyz * iyz)
        - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz)
    )
    return [trace, second, determinant]


def _urdf_merged_body_inertials(path: Path, backend_names: Sequence[str]) -> dict[str, dict[str, Any]]:
    root = ET.parse(path).getroot()
    link_elements = {element.get("name"): element for element in root.findall("link")}
    fixed_children: dict[str, list[tuple[str, list[float], list[list[float]]]]] = {}
    active_body_names = ["pelvis_link"]
    for joint in root.findall("joint"):
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        _require(parent_element is not None and child_element is not None, "URDF joint lacks parent/child")
        parent = str(parent_element.get("link"))
        child = str(child_element.get("link"))
        if joint.get("name") in backend_names:
            active_body_names.append(child)
        if joint.get("type") == "fixed":
            origin = joint.find("origin")
            translation = _xyz(origin.get("xyz") if origin is not None else None)
            rotation = _rpy_matrix(_xyz(origin.get("rpy") if origin is not None else None))
            fixed_children.setdefault(parent, []).append((child, translation, rotation))

    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def link_inertial(link_name: str, translation: Sequence[float], rotation: Sequence[Sequence[float]]) -> dict[str, Any] | None:
        link = link_elements[link_name]
        inertial = link.find("inertial")
        if inertial is None:
            return None
        mass_element = inertial.find("mass")
        inertia_element = inertial.find("inertia")
        _require(mass_element is not None and inertia_element is not None, f"incomplete inertial for {link_name}")
        mass = float(mass_element.get("value", "nan"))
        origin = inertial.find("origin")
        local_com = _xyz(origin.get("xyz") if origin is not None else None)
        inertial_rotation = _rpy_matrix(_xyz(origin.get("rpy") if origin is not None else None))
        com = [translation[i] + _matvec(rotation, local_com)[i] for i in range(3)]
        matrix = [
            [float(inertia_element.get("ixx", "nan")), float(inertia_element.get("ixy", "nan")), float(inertia_element.get("ixz", "nan"))],
            [float(inertia_element.get("ixy", "nan")), float(inertia_element.get("iyy", "nan")), float(inertia_element.get("iyz", "nan"))],
            [float(inertia_element.get("ixz", "nan")), float(inertia_element.get("iyz", "nan")), float(inertia_element.get("izz", "nan"))],
        ]
        link_to_inertia = _matmul(rotation, inertial_rotation)
        rotated = _matmul(_matmul(link_to_inertia, matrix), _transpose(link_to_inertia))
        return {"mass": mass, "com": com, "inertia_at_com": rotated}

    def collect_fixed(
        link_name: str,
        translation: Sequence[float],
        rotation: Sequence[Sequence[float]],
        output: list[dict[str, Any]],
    ) -> None:
        item = link_inertial(link_name, translation, rotation)
        if item is not None:
            output.append(item)
        for child, child_translation, child_rotation in fixed_children.get(link_name, []):
            rotated_translation = _matvec(rotation, child_translation)
            next_translation = [translation[i] + rotated_translation[i] for i in range(3)]
            next_rotation = _matmul(rotation, child_rotation)
            collect_fixed(child, next_translation, next_rotation, output)

    result: dict[str, dict[str, Any]] = {}
    for body_name in active_body_names:
        parts: list[dict[str, Any]] = []
        collect_fixed(body_name, [0.0, 0.0, 0.0], identity, parts)
        mass = sum(part["mass"] for part in parts)
        _require(mass > 0.0, f"non-positive merged body mass for {body_name}")
        com = [sum(part["mass"] * part["com"][axis] for part in parts) / mass for axis in range(3)]
        inertia = [[0.0] * 3 for _ in range(3)]
        for part in parts:
            offset = [part["com"][axis] - com[axis] for axis in range(3)]
            squared = sum(value * value for value in offset)
            for row in range(3):
                for column in range(3):
                    parallel_axis = part["mass"] * (
                        (squared if row == column else 0.0) - offset[row] * offset[column]
                    )
                    inertia[row][column] += part["inertia_at_com"][row][column] + parallel_axis
        result[body_name] = {
            "mass": mass,
            "com": com,
            "inertia_invariants": _inertia_invariants(inertia),
        }
    return result


def _mujoco_body_inertials(path: Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(path).getroot()
    result: dict[str, dict[str, Any]] = {}
    for body in root.findall(".//body"):
        inertial = body.find("inertial")
        if inertial is None:
            continue
        if inertial.get("fullinertia"):
            ixx, iyy, izz, ixy, ixz, iyz = [float(value) for value in inertial.get("fullinertia", "").split()]
            matrix = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
        else:
            diagonal = _xyz(inertial.get("diaginertia"))
            matrix = [[diagonal[0], 0.0, 0.0], [0.0, diagonal[1], 0.0], [0.0, 0.0, diagonal[2]]]
        result[str(body.get("name"))] = {
            "mass": float(inertial.get("mass", "nan")),
            "com": _xyz(inertial.get("pos")),
            "inertia_invariants": _inertia_invariants(matrix),
        }
    return result


def _physical_model_audit(urdf_path: Path, mujoco_path: Path, backend_names: Sequence[str]) -> dict[str, Any]:
    urdf = _urdf_merged_body_inertials(urdf_path, backend_names)
    mujoco = _mujoco_body_inertials(mujoco_path)
    _require(set(urdf) <= set(mujoco), "MuJoCo is missing one or more URDF active bodies")
    rows = []
    for name, expected in urdf.items():
        actual = mujoco[name]
        com_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(expected["com"], actual["com"])))
        inertia_relative_errors = []
        for expected_value, actual_value in zip(expected["inertia_invariants"], actual["inertia_invariants"]):
            denominator = max(abs(expected_value), 1e-12)
            inertia_relative_errors.append(abs(actual_value - expected_value) / denominator)
        rows.append({
            "body": name,
            "urdf_merged_mass_kg": expected["mass"],
            "mujoco_mass_kg": actual["mass"],
            "mass_error_kg": actual["mass"] - expected["mass"],
            "com_error_m": com_error,
            "max_inertia_invariant_relative_error": max(inertia_relative_errors),
        })
    return {
        "active_body_count": len(rows),
        "urdf_total_mass_kg": sum(item["mass"] for item in urdf.values()),
        "mujoco_total_mass_kg": sum(mujoco[name]["mass"] for name in urdf),
        "max_abs_body_mass_error_kg": max(abs(row["mass_error_kg"]) for row in rows),
        "max_body_com_error_m": max(row["com_error_m"] for row in rows),
        "max_inertia_invariant_relative_error": max(row["max_inertia_invariant_relative_error"] for row in rows),
        "mismatches": [
            row
            for row in rows
            if abs(row["mass_error_kg"]) > 5e-5
            or row["com_error_m"] > 5e-4
            or row["max_inertia_invariant_relative_error"] > 5e-3
        ],
    }


def _foot_collision_audit(urdf_path: Path, mujoco_path: Path) -> dict[str, Any]:
    urdf_root = ET.parse(urdf_path).getroot()
    mujoco_root = ET.parse(mujoco_path).getroot()
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        body_name = f"{side}_ankle_roll_Link"
        urdf_link = next(link for link in urdf_root.findall("link") if link.get("name") == body_name)
        urdf_meshes = [
            mesh.get("filename")
            for mesh in urdf_link.findall("collision/geometry/mesh")
            if mesh.get("filename")
        ]
        mujoco_body = next(body for body in mujoco_root.findall(".//body") if body.get("name") == body_name)
        mujoco_meshes = [
            geom.get("mesh")
            for geom in mujoco_body.findall("geom")
            if geom.get("class") == "collision" and geom.get("mesh")
        ]
        result[side] = {
            "isaac_urdf_collision_meshes": urdf_meshes,
            "mujoco_collision_meshes": mujoco_meshes,
            "same_collision_asset_name": urdf_meshes == mujoco_meshes,
        }
    result["requires_contact_calibration"] = any(
        not result[side]["same_collision_asset_name"] for side in ("left", "right")
    )
    return result


def validate_source_assets(composer: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Check source hashes plus URDF/MuJoCo order, axes, limits, and effort."""

    repo_root = repo_root.expanduser().resolve()
    sources = composer.get("source_assets")
    _require(isinstance(sources, dict), "source_assets must be an object")
    resolved: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for label, item in sources.items():
        _require(isinstance(item, dict), f"invalid source asset entry: {label}")
        path = repo_root / str(item.get("path", ""))
        _require(path.is_file(), f"source asset missing: {path}")
        actual = file_sha256(path)
        _require(actual == item.get("sha256"), f"source asset hash mismatch: {label}")
        resolved[label] = path
        hashes[label] = actual

    backend_names = list(composer["backend_joint_names"])
    prepared = _joint_semantics_from_urdf(resolved["prepared_isaac_urdf"], backend_names)
    source = _joint_semantics_from_urdf(resolved["agibot_source_urdf"], backend_names)
    mujoco = _joint_semantics_from_mujoco(resolved["official_mujoco_xml"], backend_names)
    for label, joints in (("prepared URDF", prepared), ("source URDF", source), ("MuJoCo", mujoco)):
        _require([joint["name"] for joint in joints] == backend_names, f"{label} joint order mismatch")

    contract_lower = _finite_vector(composer, "joint_lower_rad", 31)
    contract_upper = _finite_vector(composer, "joint_upper_rad", 31)
    for index, name in enumerate(backend_names):
        for urdf_joint in (prepared[index], source[index]):
            _require(urdf_joint["axis"] == source[index]["axis"], f"URDF axis mismatch for {name}")
            _require(
                abs(urdf_joint["lower"] - contract_lower[index]) <= 1e-12,
                f"URDF lower limit mismatch for {name}",
            )
            _require(
                abs(urdf_joint["upper"] - contract_upper[index]) <= 1e-12,
                f"URDF upper limit mismatch for {name}",
            )
        _require(mujoco[index]["axis"] == source[index]["axis"], f"MuJoCo axis mismatch for {name}")
        _require(
            abs(mujoco[index]["lower"] - contract_lower[index]) <= 5e-6,
            f"MuJoCo lower limit mismatch for {name}",
        )
        _require(
            abs(mujoco[index]["upper"] - contract_upper[index]) <= 5e-6,
            f"MuJoCo upper limit mismatch for {name}",
        )
        _require(abs(mujoco[index]["effort"] - source[index]["effort"]) <= 1e-9, f"effort limit mismatch for {name}")

    option = ET.parse(resolved["official_mujoco_xml"]).getroot().find("option")
    _require(option is not None, "MuJoCo option element missing")
    timestep = float(option.get("timestep", "nan"))
    _require(math.isfinite(timestep) and timestep > 0.0, "invalid MuJoCo timestep")
    return {
        "source_hashes": hashes,
        "joint_semantics_checked": len(backend_names),
        "official_mujoco_physics_hz": 1.0 / timestep,
        "physical_model_audit": _physical_model_audit(
            resolved["agibot_source_urdf"], resolved["official_mujoco_xml"], backend_names
        ),
        "foot_collision_audit": _foot_collision_audit(
            resolved["prepared_isaac_urdf"], resolved["official_mujoco_xml"]
        ),
    }


def contract_dir_from_script() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts" / "a3_base_locomotion_v1"
