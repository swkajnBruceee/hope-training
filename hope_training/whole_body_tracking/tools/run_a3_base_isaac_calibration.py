#!/usr/bin/env python3
"""Run one or more immutable A3 Base traces in isolated Isaac Sim/Isaac Lab."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--contract-dir", type=Path, required=True)
parser.add_argument("--matrix", type=Path, required=True)
parser.add_argument("--trace", type=Path)
parser.add_argument("--trace-metadata", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument(
    "--batch-manifest",
    type=Path,
    help="JSON list of pre-generated trace/trace_metadata/output paths.",
)
parser.add_argument("--batch-summary", type=Path)
parser.add_argument(
    "--batch-output-dir",
    type=Path,
    help="Override only batch result destinations; trace inputs remain read-only.",
)
parser.add_argument(
    "--diagnostic-joint-friction-coefficient",
    type=float,
    help=(
        "Diagnostic-only unitless PhysX joint friction coefficient applied to "
        "the selected joint. It is not a MuJoCo frictionloss value."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
single_paths = (args_cli.trace, args_cli.trace_metadata, args_cli.output)
if args_cli.batch_manifest is None and not all(single_paths):
    parser.error("single mode requires --trace, --trace-metadata, and --output")
if args_cli.batch_manifest is not None and (any(single_paths) or args_cli.batch_summary is None):
    parser.error("batch mode requires only --batch-manifest and --batch-summary")
if args_cli.batch_manifest is None and args_cli.batch_output_dir is not None:
    parser.error("--batch-output-dir is valid only in batch mode")
if args_cli.diagnostic_joint_friction_coefficient is not None and (
    not math.isfinite(args_cli.diagnostic_joint_friction_coefficient)
    or args_cli.diagnostic_joint_friction_coefficient < 0.0
):
    parser.error("diagnostic joint friction coefficient must be finite and non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

import a3_base_calibration as calibration
import a3_base_command_trace as command_trace
import a3_base_contract as contract
import a3_base_fixture_metrics as fixture_metrics
from training.robots.agibot_a3 import A3_FEET_BODIES, AGIBOT_A3_CFG, AGIBOT_A3_URDF_PATH


_ROBOT_CFG = AGIBOT_A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_ROBOT_CFG.spawn.fix_base = False


@configclass
class FixtureSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _ROBOT_CFG
    contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
    )


def _tilt_deg(root_quat_wxyz: torch.Tensor) -> float:
    x = float(root_quat_wxyz[0, 1])
    y = float(root_quat_wxyz[0, 2])
    r_zz = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(float(np.clip(r_zz, -1.0, 1.0))))


def _load_common():
    contracts = contract.load_contracts(args_cli.contract_dir.expanduser().resolve())
    contract.validate_contracts(contracts)
    matrix = json.loads(args_cli.matrix.expanduser().resolve().read_text(encoding="utf-8"))
    calibration.validate_matrix(matrix, contracts)
    return contracts, matrix


def _load_case(trace_path: Path, metadata_path: Path, contracts, matrix):
    metadata = json.loads(
        metadata_path.expanduser().resolve().read_text(encoding="utf-8")
    )
    matches = [case for case in matrix["cases"] if case["case_id"] == metadata.get("case_id")]
    if len(matches) != 1:
        raise ValueError("trace metadata does not select exactly one matrix case")
    with np.load(trace_path.expanduser().resolve(), allow_pickle=False) as archive:
        shared_trace = {name: archive[name] for name in command_trace.ARRAY_ORDER}
    command_trace.validate_trace(shared_trace, metadata, contracts)
    return metadata, matches[0], shared_trace


def _load_jobs(contracts, matrix):
    if args_cli.batch_manifest is None:
        metadata, case, shared_trace = _load_case(
            args_cli.trace, args_cli.trace_metadata, contracts, matrix
        )
        return [(metadata, case, shared_trace, args_cli.output.expanduser().resolve())]
    manifest_path = args_cli.batch_manifest.expanduser().resolve()
    if args_cli.batch_output_dir is not None:
        output_dir = args_cli.batch_output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"batch output directory must be fresh: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("batch manifest must be a non-empty JSON list")
    jobs = []
    output_paths: set[Path] = set()
    for index, item in enumerate(manifest):
        if not isinstance(item, dict) or set(item) != {"trace", "trace_metadata", "output"}:
            raise ValueError(f"invalid batch manifest item {index}")
        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return (manifest_path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        trace_path = resolve(item["trace"])
        metadata_path = resolve(item["trace_metadata"])
        output_path = resolve(item["output"])
        if args_cli.batch_output_dir is not None:
            output_path = (
                args_cli.batch_output_dir.expanduser().resolve() / output_path.name
            )
        if output_path in output_paths:
            raise ValueError("batch manifest output paths must be unique")
        output_paths.add(output_path)
        metadata, case, shared_trace = _load_case(
            trace_path, metadata_path, contracts, matrix
        )
        jobs.append((metadata, case, shared_trace, output_path))
    rates = {float(job[2]["physics_dt_s"][0]) for job in jobs}
    if len(rates) != 1:
        raise ValueError("one Isaac batch may contain only one physics dt")
    return jobs


def _run_fixture(contracts, metadata, case, shared_trace, sim, scene):
    calibration_contract = contracts["calibration_contract.json"]
    runner_contract = calibration_contract["native_isaac_runner"]
    category = str(metadata["category"])
    if category not in runner_contract["approved_categories"]:
        raise ValueError(f"Isaac fixture runner does not support {category}")
    if metadata.get("plant_constraint") != runner_contract["plant_constraint"]:
        raise ValueError("trace fixture contract mismatch")
    physics_dt = float(shared_trace["physics_dt_s"][0])
    if not math.isclose(sim.get_physics_dt(), physics_dt, abs_tol=1.0e-12):
        raise ValueError("Isaac physics dt differs from shared trace")

    robot: Articulation = scene["robot"]
    contacts: ContactSensor = scene["contacts"]
    backend_names = list(contracts["command_composer_contract.json"]["backend_joint_names"])
    if shared_trace["joint_names"].tolist() != backend_names:
        raise ValueError("trace/Composer joint order mismatch")
    isaac_names = list(robot.data.joint_names)
    if set(isaac_names) != set(backend_names) or len(isaac_names) != len(backend_names):
        raise ValueError("Isaac articulation joint set differs from Composer contract")
    backend_to_isaac = torch.tensor(
        [isaac_names.index(name) for name in backend_names], dtype=torch.long, device=sim.device
    )
    selected_name = str(shared_trace["selected_joint_name"][0])
    selected_backend = backend_names.index(selected_name)
    selected_isaac = isaac_names.index(selected_name)
    locked_isaac = torch.tensor(
        [index for index in range(robot.num_joints) if index != selected_isaac],
        dtype=torch.long,
        device=sim.device,
    )

    # Every job starts from the imported/configured coefficients.  This avoids
    # diagnostic state leaking between selected joints in batch mode.
    robot.write_joint_friction_coefficient_to_sim(
        robot.data.default_joint_friction_coeff
    )
    imported_selected_friction = float(
        robot.data.default_joint_friction_coeff[0, selected_isaac]
    )
    if args_cli.diagnostic_joint_friction_coefficient is not None:
        robot.write_joint_friction_coefficient_to_sim(
            float(args_cli.diagnostic_joint_friction_coefficient),
            joint_ids=[selected_isaac],
        )
    active_selected_friction = float(
        robot.data.joint_friction_coeff[0, selected_isaac]
    )

    initial_position = robot.data.default_joint_pos.clone()
    initial_velocity = torch.zeros_like(robot.data.default_joint_vel)
    fixture_root_state = robot.data.default_root_state.clone()
    initial_q_backend = initial_position[0, backend_to_isaac].detach().cpu().numpy().astype(float).tolist()
    robot.write_root_state_to_sim(fixture_root_state)
    robot.write_joint_state_to_sim(initial_position, initial_velocity)
    robot.reset()
    contacts.reset()
    baseline_backend = torch.as_tensor(
        np.asarray(shared_trace["composed_policy_target_rad"])[0],
        dtype=initial_position.dtype,
        device=sim.device,
    )
    baseline_isaac = initial_position.clone()
    baseline_isaac[0, backend_to_isaac] = baseline_backend
    locked_position = baseline_isaac[:, locked_isaac]
    locked_velocity = torch.zeros_like(locked_position)

    policy_targets = np.asarray(shared_trace["composed_policy_target_rad"])
    baseline_selected = float(policy_targets[0, selected_backend])
    changed = np.flatnonzero(np.abs(policy_targets[:, selected_backend] - baseline_selected) > 0.0)
    if category == "joint_zero_baseline":
        if changed.size != 0:
            raise ValueError("zero-baseline trace changes its selected target")
        excited_value = baseline_selected
        command_delta = 0.0
    else:
        if changed.size == 0:
            raise ValueError("trace never excites its selected joint")
        excited_value = float(policy_targets[int(changed[0]), selected_backend])
        command_delta = excited_value - baseline_selected
        if command_delta == 0.0:
            raise ValueError("selected trace command delta is zero")

    foot_names = set(A3_FEET_BODIES)
    forbidden_sensor_indices = [
        index for index, name in enumerate(contacts.body_names) if name not in foot_names
    ]
    force_threshold_n = 1.0e-4
    envelope = calibration_contract["safety_envelope"]
    original_trace_hash = command_trace.trace_sha256(shared_trace)
    effort_limit = float(robot.data.joint_effort_limits[0, selected_isaac])

    rows: dict[str, list] = {
        "time_s": [],
        "joint_q_rad": [],
        "joint_target_rad": [],
        "joint_dq_radps": [],
        "joint_torque_nm": [],
        "selected_joint_saturated": [],
        "pelvis_height_m": [],
        "pelvis_tilt_deg": [],
        "max_locked_joint_error_rad": [],
        "link_incoming_joint_force": [],
    }
    forbidden_count = 0
    forbidden_bodies: set[str] = set()
    joint_limit_count = 0
    nonfinite_count = 0
    safety_stop = False
    max_abs_velocity = 0.0
    max_abs_torque = 0.0
    max_pre_restore_root_excursion = 0.0
    max_post_restore_root_error = 0.0

    targets = np.asarray(shared_trace["composed_target_rad"])
    for step, target_backend_np in enumerate(targets):
        target_backend = torch.as_tensor(
            target_backend_np, dtype=initial_position.dtype, device=sim.device
        )
        target_isaac = baseline_isaac.clone()
        target_isaac[0, backend_to_isaac] = target_backend
        robot.set_joint_position_target(target_isaac)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(physics_dt)

        applied_torque = robot.data.applied_torque.clone()
        contact_forces = contacts.data.net_forces_w.clone()
        incoming_force = robot.root_physx_view.get_link_incoming_joint_force().clone()
        max_pre_restore_root_excursion = max(
            max_pre_restore_root_excursion,
            float(
                torch.linalg.vector_norm(
                    robot.data.root_pos_w - fixture_root_state[:, :3], dim=-1
                ).max()
            ),
        )
        robot.write_root_state_to_sim(fixture_root_state)
        robot.write_joint_state_to_sim(
            locked_position, locked_velocity, joint_ids=locked_isaac
        )

        q = robot.data.joint_pos.clone()
        dq = robot.data.joint_vel.clone()
        root_quat = robot.data.root_quat_w.clone()
        height = float(robot.data.root_pos_w[0, 2])
        tilt = _tilt_deg(root_quat)
        max_post_restore_root_error = max(
            max_post_restore_root_error,
            float(
                torch.linalg.vector_norm(
                    robot.data.root_pos_w - fixture_root_state[:, :3], dim=-1
                ).max()
            ),
        )
        locked_error = float(
            torch.max(torch.abs(q[:, locked_isaac] - locked_position))
        )
        nonfinite = not bool(
            torch.all(torch.isfinite(q))
            and torch.all(torch.isfinite(dq))
            and torch.all(torch.isfinite(applied_torque))
            and torch.all(torch.isfinite(incoming_force))
        )
        nonfinite_count += int(nonfinite)

        forbidden_this_step = 0
        if forbidden_sensor_indices:
            magnitudes = torch.linalg.vector_norm(
                contact_forces[:, forbidden_sensor_indices], dim=-1
            )[0]
            hit_indices = torch.nonzero(magnitudes > force_threshold_n).flatten().tolist()
            forbidden_this_step = len(hit_indices)
            forbidden_bodies.update(
                contacts.body_names[forbidden_sensor_indices[index]] for index in hit_indices
            )
        forbidden_count += forbidden_this_step

        lower = robot.data.joint_pos_limits[0, :, 0]
        upper = robot.data.joint_pos_limits[0, :, 1]
        limit_hit = bool(torch.any((q[0] < lower - 1.0e-6) | (q[0] > upper + 1.0e-6)))
        joint_limit_count += int(limit_hit)
        max_abs_velocity = max(max_abs_velocity, float(torch.max(torch.abs(dq))))
        max_abs_torque = max(max_abs_torque, float(torch.max(torch.abs(applied_torque))))
        selected_torque = float(applied_torque[0, selected_isaac])

        rows["time_s"].append(float(shared_trace["metric_timestamp_s"][step]))
        rows["joint_q_rad"].append(float(q[0, selected_isaac]))
        rows["joint_target_rad"].append(float(target_backend_np[selected_backend]))
        rows["joint_dq_radps"].append(float(dq[0, selected_isaac]))
        rows["joint_torque_nm"].append(selected_torque)
        rows["selected_joint_saturated"].append(abs(selected_torque) >= effort_limit - 1.0e-6)
        rows["pelvis_height_m"].append(height)
        rows["pelvis_tilt_deg"].append(tilt)
        rows["max_locked_joint_error_rad"].append(locked_error)
        rows["link_incoming_joint_force"].append(incoming_force[0].detach().cpu().numpy())

        if (
            nonfinite
            or forbidden_this_step > 0
            or limit_hit
            or tilt > float(envelope["max_tilt_deg"])
            or height < float(envelope["min_pelvis_height_m"])
        ):
            safety_stop = True
            break

    evidence = {
        name: np.asarray(values, dtype=np.bool_ if name == "selected_joint_saturated" else np.float64)
        for name, values in rows.items()
    }
    selected_metrics = fixture_metrics.summarize_response(
        category=category,
        evidence=evidence,
        trace_metadata=metadata,
        command_delta=command_delta,
        excited_value=excited_value,
        physics_dt=physics_dt,
        constraint_reaction_available=False,
    )
    metrics = {
        "nonfinite_count": nonfinite_count,
        "safety_stop": safety_stop,
        "forbidden_contact_count": forbidden_count,
        "joint_limit_hit_count": joint_limit_count,
        "max_tilt_deg": float(np.max(evidence["pelvis_tilt_deg"])),
        "min_pelvis_height_m": float(np.min(evidence["pelvis_height_m"])),
        "max_abs_joint_velocity_radps": max_abs_velocity,
        "max_abs_torque_nm": max_abs_torque,
        **selected_metrics,
    }
    model_hash = contract.file_sha256(Path(AGIBOT_A3_URDF_PATH))
    composer = contracts["command_composer_contract.json"]
    fixture_instance_contract = dict(runner_contract)
    fixture_instance_contract["diagnostic_model_overrides"] = {
        "selected_joint_friction_coefficient": (
            args_cli.diagnostic_joint_friction_coefficient
        )
    }
    instance_hash = command_trace.case_instance_sha256(
        trace_metadata=metadata,
        model_sha256=model_hash,
        fixture_contract=fixture_instance_contract,
        initial_q_rad=initial_q_backend,
        kp=list(composer["kp"]),
        kd=list(composer["kd"]),
    )
    result = {
        "case_id": case["case_id"],
        "case_instance_sha256": instance_hash,
        "trace_sha256": metadata["trace_sha256"],
        "model_sha256": model_hash,
        "metrics": metrics,
        "runner_facts": {
            "runner": (
                runner_contract["runner_id"]
                if args_cli.diagnostic_joint_friction_coefficient is None
                else runner_contract["runner_id"] + "_diagnostic_ablation"
            ),
            "isaac_sim_version": importlib.metadata.version("isaacsim"),
            "isaac_lab_version": importlib.metadata.version("isaaclab"),
            "device": str(sim.device),
            "physics_rate_hz": 1.0 / physics_dt,
            "policy_rate_hz": 1.0 / float(shared_trace["policy_dt_s"][0]),
            "selected_joint_name": selected_name,
            "joint_friction_semantics": (
                "unitless_physx_coefficient_bounded_by_transmitted_spatial_force"
            ),
            "imported_selected_joint_friction_coefficient": imported_selected_friction,
            "active_selected_joint_friction_coefficient": active_selected_friction,
            "diagnostic_model_override": (
                args_cli.diagnostic_joint_friction_coefficient is not None
            ),
            "plant_constraint": runner_contract["plant_constraint"],
            "fixture_semantics": runner_contract["fixture_semantics"],
            "fixture_is_free_base_evidence": False,
            "ground_contact_enabled": False,
            "self_collision_enabled": False,
            "root_is_fixed": bool(robot.is_fixed_base),
            "root_state_restored_each_substep": True,
            "max_pre_restore_root_excursion_m": max_pre_restore_root_excursion,
            "max_post_restore_root_error_m": max_post_restore_root_error,
            "max_locked_joint_error_rad": float(
                np.max(evidence["max_locked_joint_error_rad"])
            ),
            "constraint_reaction_available": False,
            "raw_link_incoming_joint_force_recorded": True,
            "raw_link_incoming_joint_force_semantics_verified": False,
            "forbidden_contact_bodies": sorted(forbidden_bodies),
            "forbidden_contact_count_semantics": (
                "sum_nonfoot_bodies_with_net_contact_force_gt_1e-4N_per_step"
            ),
            "completed_steps": len(evidence["time_s"]),
            "planned_steps": len(targets),
            "trace_was_mutated": command_trace.trace_sha256(shared_trace) != original_trace_hash,
            "no_ros_aimrt_or_network_transport": True,
        },
    }
    return result, evidence


def _write_case_output(result, evidence, case, output, contracts, matrix, runner_contract):
    case_validation = calibration.validate_case_result(result, case["category"], contracts)
    result["matrix_sha256"] = matrix["matrix_sha256"]
    result["runner_source_sha256"] = contract.file_sha256(Path(__file__).resolve())
    result["contract_status"] = runner_contract["result_status"]
    result["case_validation"] = case_validation
    if output.suffix != ".json":
        raise ValueError("case output must end in .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    evidence_path = output.with_suffix(".trace.npz")
    np.savez(evidence_path, **evidence)
    return {
        "output": str(output),
        "trace": str(evidence_path),
        "case_id": result["case_id"],
        "metrics": result["metrics"],
        "runner_facts": result["runner_facts"],
        "case_validation": case_validation,
    }


def main() -> None:
    contracts, matrix = _load_common()
    jobs = _load_jobs(contracts, matrix)
    runner_contract = contracts["calibration_contract.json"]["native_isaac_runner"]
    if importlib.metadata.version("isaacsim") != runner_contract["required_isaac_sim_version"]:
        raise RuntimeError("Isaac Sim version differs from fixture contract")
    if importlib.metadata.version("isaaclab") != runner_contract["required_isaac_lab_version"]:
        raise RuntimeError("Isaac Lab version differs from fixture contract")
    expected_model_hash = contracts["command_composer_contract.json"]["source_assets"][
        "prepared_isaac_urdf"
    ]["sha256"]
    if contract.file_sha256(Path(AGIBOT_A3_URDF_PATH)) != expected_model_hash:
        raise ValueError("prepared Isaac URDF hash mismatch")

    physics_dt = float(jobs[0][2]["physics_dt_s"][0])
    sim = SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, device=args_cli.device))
    scene = InteractiveScene(FixtureSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    summaries = []
    for metadata, case, shared_trace, output in jobs:
        result, evidence = _run_fixture(
            contracts, metadata, case, shared_trace, sim, scene
        )
        summaries.append(
            _write_case_output(
                result, evidence, case, output, contracts, matrix, runner_contract
            )
        )
    payload = {
        "runner": runner_contract["runner_id"],
        "matrix_sha256": matrix["matrix_sha256"],
        "batch_mode": args_cli.batch_manifest is not None,
        "case_count": len(summaries),
        "diagnostic_model_overrides": {
            "selected_joint_friction_coefficient": (
                args_cli.diagnostic_joint_friction_coefficient
            )
        },
        "cases": summaries,
        "automatic_promotion": False,
    }
    if args_cli.batch_manifest is not None:
        summary_path = args_cli.batch_summary.expanduser().resolve()
        if summary_path.suffix != ".json":
            raise ValueError("--batch-summary must end in .json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    printed_payload = payload
    if args_cli.batch_manifest is not None:
        passed = sum(
            bool(item["case_validation"]["safety_envelope_passed"])
            for item in summaries
        )
        printed_payload = {
            "runner": payload["runner"],
            "matrix_sha256": payload["matrix_sha256"],
            "batch_mode": True,
            "batch_summary": str(args_cli.batch_summary.expanduser().resolve()),
            "case_count": len(summaries),
            "safety_envelope_pass_count": int(passed),
            "safety_envelope_fail_count": len(summaries) - int(passed),
            "case_ids": [item["case_id"] for item in summaries],
            "automatic_promotion": False,
        }
    print(json.dumps(printed_payload, indent=2), flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # Isaac Sim 4.5 can block indefinitely inside close() in this headless
        # environment. Arm a bounded teardown only after every artifact and
        # stream has been flushed, then still give Kit a normal close attempt.
        teardown = threading.Timer(5.0, lambda: os._exit(exit_code))
        teardown.daemon = True
        teardown.start()
        simulation_app.close()
    # If close() returns normally, do not keep non-daemon Kit threads alive.
    os._exit(exit_code)
