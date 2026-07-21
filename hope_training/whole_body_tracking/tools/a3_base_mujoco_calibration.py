#!/usr/bin/env python3
"""Consume immutable A3 Base traces in isolated native-substep MuJoCo."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import a3_base_command_trace as command_trace
import a3_base_contract as contract
import a3_base_fixture_metrics as fixture_metrics


def _load_mujoco(extra_python_path: str | None = None):
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError:
        if not extra_python_path:
            raise RuntimeError(
                "MuJoCo Python bindings are unavailable; install mujoco==3.1.6 "
                "or pass --mujoco-python-path"
            ) from None
        import sys

        sys.path.append(extra_python_path)
        import mujoco  # type: ignore
    if mujoco.__version__ != "3.1.6":
        raise RuntimeError(f"expected mujoco 3.1.6, got {mujoco.__version__}")
    return mujoco


def _tilt_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))


def run_shared_trace(
    case: Mapping[str, Any],
    shared_trace: Mapping[str, np.ndarray],
    trace_metadata: Mapping[str, Any],
    contracts: Mapping[str, Mapping[str, Any]],
    model_path: str,
    mujoco_python_path: str | None = None,
    passive_friction_scale: float = 1.0,
    passive_damping_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run a read-only shared trace; target generation is forbidden here."""

    command_trace.validate_trace(shared_trace, trace_metadata, contracts)
    if trace_metadata.get("case_id") != case.get("case_id"):
        raise ValueError("trace/case ID mismatch")
    category = str(trace_metadata["category"])
    if category not in {
        "joint_zero_baseline",
        "base_action_step",
        "waist_pitch_residual",
        "target_transport",
    }:
        raise ValueError(f"native MuJoCo runner does not support {category}")
    calibration = contracts["calibration_contract.json"]
    composer = contracts["command_composer_contract.json"]
    runner_contract = calibration["native_mujoco_runner"]
    if trace_metadata.get("plant_constraint") != runner_contract["plant_constraint"]:
        raise ValueError("trace fixture contract mismatch")

    mujoco = _load_mujoco(mujoco_python_path)
    model = mujoco.MjModel.from_xml_path(model_path)
    if not math.isfinite(passive_friction_scale) or passive_friction_scale < 0.0:
        raise ValueError("passive friction scale must be finite and non-negative")
    if not math.isfinite(passive_damping_scale) or passive_damping_scale < 0.0:
        raise ValueError("passive damping scale must be finite and non-negative")
    model.dof_frictionloss[:] *= passive_friction_scale
    model.dof_damping[:] *= passive_damping_scale
    data = mujoco.MjData(model)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    physics_dt = float(shared_trace["physics_dt_s"][0])
    physics_hz = 1.0 / physics_dt
    model.opt.timestep = physics_dt
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_id < 0:
        raise RuntimeError("official model does not contain stand keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)
    fixture_qpos = data.qpos[:7].copy()

    names = list(composer["backend_joint_names"])
    if shared_trace["joint_names"].tolist() != names:
        raise ValueError("trace/Composer joint order mismatch")
    joint_ids: list[int] = []
    qpos_adr: list[int] = []
    dof_adr: list[int] = []
    actuator_ids: list[int] = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor"
        )
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(f"official model lacks joint/motor for {name}")
        joint_ids.append(joint_id)
        qpos_adr.append(int(model.jnt_qposadr[joint_id]))
        dof_adr.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(actuator_id)
    if actuator_ids != list(range(31)):
        raise RuntimeError("official actuator order differs from Composer order")

    selected_name = str(shared_trace["selected_joint_name"][0])
    selected_index = names.index(selected_name)
    initial_q = data.qpos[qpos_adr].astype(np.float64).tolist()
    baseline = np.asarray(shared_trace["composed_policy_target_rad"][0])
    policy_targets = np.asarray(shared_trace["composed_policy_target_rad"])
    target_changes = np.flatnonzero(
        np.abs(policy_targets[:, selected_index] - baseline[selected_index]) > 0.0
    )
    if category == "joint_zero_baseline":
        if target_changes.size != 0:
            raise ValueError("zero-baseline trace changes its selected target")
        excited_value = float(baseline[selected_index])
        command_delta = 0.0
    else:
        if target_changes.size == 0:
            raise ValueError("trace never excites its selected joint")
        excited_value = float(policy_targets[int(target_changes[0]), selected_index])
        command_delta = excited_value - float(baseline[selected_index])
        if command_delta == 0.0:
            raise ValueError("selected trace command delta is zero")

    kp = np.asarray(composer["kp"], dtype=np.float64)
    kd = np.asarray(composer["kd"], dtype=np.float64)
    ctrl_min = model.actuator_ctrlrange[:, 0].copy()
    ctrl_max = model.actuator_ctrlrange[:, 1].copy()
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    allowed_feet = {
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "left_ankle_roll_collision"
        ),
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "right_ankle_roll_collision"
        ),
    }
    envelope = calibration["safety_envelope"]

    time_rows: list[float] = []
    q_rows: list[float] = []
    target_rows: list[float] = []
    dq_rows: list[float] = []
    torque_rows: list[float] = []
    saturation_rows: list[bool] = []
    height_rows: list[float] = []
    tilt_rows: list[float] = []
    forbidden_count = 0
    forbidden_pairs: set[str] = set()
    joint_limit_count = 0
    nonfinite_count = 0
    safety_stop = False
    max_abs_velocity = 0.0
    max_abs_torque = 0.0
    targets = np.asarray(shared_trace["composed_target_rad"])

    for step, target in enumerate(targets):
        q = data.qpos[qpos_adr].copy()
        dq = data.qvel[dof_adr].copy()
        torque = np.clip(kp * (target - q) - kd * dq, ctrl_min, ctrl_max)
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
        # State restoration is explicit and therefore exposes no physically
        # meaningful fixture reaction. Only the selected joint remains dynamic.
        data.qpos[:7] = fixture_qpos
        data.qvel[:6] = 0.0
        for fixture_index in range(31):
            if fixture_index == selected_index:
                continue
            data.qpos[qpos_adr[fixture_index]] = baseline[fixture_index]
            data.qvel[dof_adr[fixture_index]] = 0.0
        mujoco.mj_forward(model, data)

        q_after = data.qpos[qpos_adr].copy()
        dq_after = data.qvel[dof_adr].copy()
        height = float(data.qpos[2])
        tilt = _tilt_deg(data.xmat[pelvis_body].reshape(3, 3))
        forbidden_this_step = 0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if floor_geom in pair and len(pair & allowed_feet) == 1:
                continue
            forbidden_this_step += 1
            first = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
            )
            second = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
            )
            forbidden_pairs.add(f"{first or contact.geom1}|{second or contact.geom2}")
        forbidden_count += forbidden_this_step
        limit_hit = any(
            bool(model.jnt_limited[joint_id])
            and not (
                float(model.jnt_range[joint_id, 0]) - 1.0e-6
                <= float(q_after[index])
                <= float(model.jnt_range[joint_id, 1]) + 1.0e-6
            )
            for index, joint_id in enumerate(joint_ids)
        )
        joint_limit_count += int(limit_hit)
        finite = np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        nonfinite_count += int(not finite)
        max_abs_velocity = max(max_abs_velocity, float(np.max(np.abs(dq_after))))
        max_abs_torque = max(max_abs_torque, float(np.max(np.abs(torque))))
        selected_torque = float(torque[selected_index])

        time_rows.append(float(shared_trace["metric_timestamp_s"][step]))
        q_rows.append(float(q_after[selected_index]))
        target_rows.append(float(target[selected_index]))
        dq_rows.append(float(dq_after[selected_index]))
        torque_rows.append(selected_torque)
        saturation_rows.append(
            math.isclose(selected_torque, float(ctrl_min[selected_index]), abs_tol=1.0e-9)
            or math.isclose(selected_torque, float(ctrl_max[selected_index]), abs_tol=1.0e-9)
        )
        height_rows.append(height)
        tilt_rows.append(tilt)
        if (
            not finite
            or forbidden_this_step > 0
            or limit_hit
            or tilt > float(envelope["max_tilt_deg"])
            or height < float(envelope["min_pelvis_height_m"])
        ):
            safety_stop = True
            break

    evidence = {
        "time_s": np.asarray(time_rows, dtype=np.float64),
        "joint_q_rad": np.asarray(q_rows, dtype=np.float64),
        "joint_target_rad": np.asarray(target_rows, dtype=np.float64),
        "joint_dq_radps": np.asarray(dq_rows, dtype=np.float64),
        "joint_torque_nm": np.asarray(torque_rows, dtype=np.float64),
        "selected_joint_saturated": np.asarray(saturation_rows, dtype=np.bool_),
        "pelvis_height_m": np.asarray(height_rows, dtype=np.float64),
        "pelvis_tilt_deg": np.asarray(tilt_rows, dtype=np.float64),
    }
    selected_metrics = fixture_metrics.summarize_response(
        category=category,
        evidence=evidence,
        trace_metadata=trace_metadata,
        command_delta=command_delta,
        excited_value=excited_value,
        physics_dt=physics_dt,
        constraint_reaction_available=False,
    )
    metrics: dict[str, Any] = {
        "nonfinite_count": int(nonfinite_count),
        "safety_stop": bool(safety_stop),
        "forbidden_contact_count": int(forbidden_count),
        "joint_limit_hit_count": int(joint_limit_count),
        "max_tilt_deg": float(np.max(evidence["pelvis_tilt_deg"])),
        "min_pelvis_height_m": float(np.min(evidence["pelvis_height_m"])),
        "max_abs_joint_velocity_radps": max_abs_velocity,
        "max_abs_torque_nm": max_abs_torque,
        **selected_metrics,
    }
    model_hash = contract.file_sha256(Path(model_path))
    fixture_instance_contract = dict(runner_contract)
    fixture_instance_contract["diagnostic_model_overrides"] = {
        "passive_friction_scale": passive_friction_scale,
        "passive_damping_scale": passive_damping_scale,
    }
    instance_hash = command_trace.case_instance_sha256(
        trace_metadata=trace_metadata,
        model_sha256=model_hash,
        fixture_contract=fixture_instance_contract,
        initial_q_rad=initial_q,
        kp=kp.tolist(),
        kd=kd.tolist(),
    )
    result = {
        "case_id": case["case_id"],
        "case_instance_sha256": instance_hash,
        "trace_sha256": trace_metadata["trace_sha256"],
        "model_sha256": model_hash,
        "metrics": metrics,
        "runner_facts": {
            "runner": (
                runner_contract["runner_id"]
                if passive_friction_scale == 1.0 and passive_damping_scale == 1.0
                else runner_contract["runner_id"] + "_diagnostic_ablation"
            ),
            "mujoco_version": mujoco.__version__,
            "physics_rate_hz": physics_hz,
            "policy_rate_hz": 1.0 / float(shared_trace["policy_dt_s"][0]),
            "selected_joint_name": selected_name,
            "plant_constraint": runner_contract["plant_constraint"],
            "fixture_is_free_base_evidence": False,
            "ground_contact_enabled": False,
            "self_collision_enabled": False,
            "passive_friction_scale": passive_friction_scale,
            "passive_damping_scale": passive_damping_scale,
            "diagnostic_model_override": bool(
                passive_friction_scale != 1.0 or passive_damping_scale != 1.0
            ),
            "constraint_reaction_available": False,
            "forbidden_contact_pairs": sorted(forbidden_pairs),
            "completed_steps": len(time_rows),
            "planned_steps": len(targets),
            "trace_was_mutated": False,
            "no_ros_aimrt_or_network_transport": True,
        },
    }
    return result, evidence


def run_case(
    case: Mapping[str, Any],
    contracts: Mapping[str, Mapping[str, Any]],
    model_path: str,
    mujoco_python_path: str | None = None,
    passive_friction_scale: float = 1.0,
    passive_damping_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compatibility wrapper: generate once, then call the read-only runner."""

    rate = float(case["inputs"].get("physics_rate_hz", 1000.0))
    shared_trace, metadata = command_trace.build_trace(case, contracts, rate)
    return run_shared_trace(
        case,
        shared_trace,
        metadata,
        contracts,
        model_path,
        mujoco_python_path,
        passive_friction_scale,
        passive_damping_scale,
    )
