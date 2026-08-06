#!/usr/bin/env python3
"""Calibrate zero-residual Recovery-A disturbance profiles before PPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--runtime-smoke",
    action="store_true",
    help="Allow a short code-path smoke; never marks calibration_measured.",
)
parser.add_argument(
    "--trace",
    type=Path,
    required=True,
    help="Versioned NPZ produced by build_a3_base_recovery_disturbance_trace.py.",
)
parser.add_argument(
    "--runtime-contract",
    type=Path,
    help="Frozen source/asset/trace hash manifest; required for formal evidence.",
)
parser.add_argument(
    "--profiles",
    type=str,
    default="all",
    help="Comma-separated profile names, or 'all'.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 32 or args_cli.num_envs % 16:
    parser.error("--num-envs must be a multiple of 16 and at least 32")
if args_cli.runtime_smoke:
    if args_cli.steps < 1 or args_cli.steps >= 500:
        parser.error("runtime smoke requires 1 <= --steps < 500")
elif args_cli.steps != 500:
    parser.error("formal recovery calibration requires exactly 500 policy steps")
if not args_cli.runtime_smoke and args_cli.runtime_contract is None:
    parser.error("formal recovery calibration requires --runtime-contract")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils

import training.tasks.base_locomotion.config.a3  # noqa: F401
from training.robots.agibot_a3 import A3_BASE_ACTION_JOINTS
from training.tasks.base_locomotion.base_env_cfg import A3_NOMINAL_BODY_HEIGHT_M


PROFILES = (
    ("recovery_a_clean", 0, 0.0, 0.0),
    ("recovery_a_candidate", 1, 0.035, 0.20),
    ("recovery_a_medium", 2, 0.050, 0.30),
    ("recovery_a_upper_probe", 3, 0.075, 0.45),
)
SETTLED_ENVELOPES = (
    ("strict", 0.01, 0.05),
    ("practical", 0.02, 0.10),
)
SETTLED_CONSECUTIVE_STEPS = 10
KEY_JOINT_NAMES = tuple(
    name
    for name in A3_BASE_ACTION_JOINTS
    if "ankle" in name or "waist" in name
)
TRACE = None
TRACE_SHA256 = None


def _tilt(projected_gravity: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.clamp(-projected_gravity[:, 2], min=-1.0, max=1.0))


def _stats(values: torch.Tensor) -> dict:
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    values = values.float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.max().item()),
    }


def _fraction(values: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    return float(values.float().mean().item())


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _combined_quadrant(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    pose_quadrant = (values_a[:, 0] < 0).astype(np.int64) * 2 + (values_a[:, 1] < 0)
    velocity_quadrant = (values_b[:, 0] < 0).astype(np.int64) * 2 + (values_b[:, 1] < 0)
    return pose_quadrant * 4 + velocity_quadrant


def _take_stratified(indices: np.ndarray, category: np.ndarray, count: int) -> np.ndarray:
    """Select deterministic rows while spreading all 16 sign combinations."""
    selected = []
    per_category = count // 16
    remainder = count % 16
    for category_id in range(16):
        candidates = indices[category == category_id]
        take = per_category + (1 if category_id < remainder else 0)
        if candidates.size < take:
            raise RuntimeError(
                f"Trace category {category_id} has {candidates.size} rows, needs {take}"
            )
        selected.extend(candidates[:take].tolist())
    return np.asarray(selected, dtype=np.int64)


def _profile_trace(profile_id: int, count: int) -> dict[str, np.ndarray]:
    if TRACE is None:
        raise RuntimeError("Recovery trace was not loaded")
    profile_indices = np.flatnonzero(TRACE["profile_id"] == profile_id)
    if profile_id == 0:
        selected = profile_indices[:count]
        if selected.size != count:
            raise RuntimeError(f"Clean trace has {selected.size} rows, needs {count}")
        return {
            "trace_index": TRACE["trace_index"][selected],
            "disturbed": np.zeros(count, dtype=np.bool_),
            "roll_pitch_rad": TRACE["roll_pitch_rad"][selected],
            "angular_velocity_rad_s": TRACE["angular_velocity_rad_s"][selected],
        }
    categories = _combined_quadrant(
        TRACE["roll_pitch_rad"][profile_indices],
        TRACE["angular_velocity_rad_s"][profile_indices],
    )
    selected = _take_stratified(profile_indices, categories, count)
    return {
        "trace_index": TRACE["trace_index"][selected],
        "disturbed": np.ones(count, dtype=np.bool_),
        "roll_pitch_rad": TRACE["roll_pitch_rad"][selected],
        "angular_velocity_rad_s": TRACE["angular_velocity_rad_s"][selected],
    }


def _root_roll_pitch_abs(projected_gravity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pitch = torch.abs(torch.asin(torch.clamp(projected_gravity[:, 0], -1.0, 1.0)))
    roll = torch.abs(torch.atan2(-projected_gravity[:, 1], -projected_gravity[:, 2]))
    return roll, pitch


def _run_profile(
    profile_name: str,
    profile_id: int,
    pose_abs_rad: float,
    angular_abs_rad_s: float,
) -> dict:
    profile_started_at = time.monotonic()
    heartbeat_path = args_cli.output.with_suffix(args_cli.output.suffix + ".heartbeat.json")
    cfg = gym.spec("A3BaseStandRecoveryA-v0").kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    cfg.sim.device = args_cli.device
    cfg.events.reset_all.params["roll_pitch_range_rad"] = (-pose_abs_rad, pose_abs_rad)
    cfg.events.reset_all.params["angular_velocity_range_rad_s"] = (
        -angular_abs_rad_s,
        angular_abs_rad_s,
    )
    env = gym.make("A3BaseStandRecoveryA-v0", cfg=cfg)
    try:
        env.reset(seed=args_cli.seed)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        trace = _profile_trace(profile_id, args_cli.num_envs)
        env_ids = torch.arange(args_cli.num_envs, device=unwrapped.device)
        pose_samples = torch.as_tensor(
            trace["roll_pitch_rad"], device=unwrapped.device, dtype=torch.float32
        )
        velocity_samples = torch.as_tensor(
            trace["angular_velocity_rad_s"], device=unwrapped.device, dtype=torch.float32
        )
        disturbed = torch.as_tensor(
            trace["disturbed"], device=unwrapped.device, dtype=torch.bool
        )
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += unwrapped.scene.env_origins
        orientation_delta = math_utils.quat_from_euler_xyz(
            pose_samples[:, 0],
            pose_samples[:, 1],
            torch.zeros(args_cli.num_envs, device=unwrapped.device),
        )
        root_state[:, 3:7] = math_utils.quat_mul(root_state[:, 3:7], orientation_delta)
        root_state[:, 10:12] += velocity_samples
        robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
        unwrapped.recovery_disturbed_mask[:] = disturbed
        unwrapped.recovery_initial_roll_pitch_rad[:] = pose_samples
        unwrapped.recovery_initial_angular_velocity_rad_s[:] = velocity_samples
        clean = ~disturbed
        initial_pose = pose_samples.clone()
        initial_ang_vel = velocity_samples.clone()
        active = torch.ones(args_cli.num_envs, dtype=torch.bool, device=unwrapped.device)
        timeout = torch.zeros_like(active)
        max_tilt = _tilt(robot.data.projected_gravity_b).clone()
        max_ang_vel = torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=-1)
        initial_root_xy = robot.data.root_pos_w[:, :2].clone()
        max_root_xy_drift = torch.zeros(args_cli.num_envs, device=unwrapped.device)
        key_joint_ids, resolved_key_joint_names = robot.find_joints(
            list(KEY_JOINT_NAMES), preserve_order=True
        )
        if resolved_key_joint_names != list(KEY_JOINT_NAMES):
            raise RuntimeError(
                f"Recovery key-joint resolution mismatch: {resolved_key_joint_names}"
            )
        term_counts = {name: 0 for name in unwrapped.termination_manager.active_terms}
        settled_run = {
            name: torch.zeros(args_cli.num_envs, dtype=torch.long, device=unwrapped.device)
            for name, _tilt_limit, _ang_limit in SETTLED_ENVELOPES
        }
        settled_step = {
            name: torch.full(
                (args_cli.num_envs,), -1, dtype=torch.long, device=unwrapped.device
            )
            for name, _tilt_limit, _ang_limit in SETTLED_ENVELOPES
        }
        clean_tail = {
            "abs_pelvis_roll_rad": [],
            "abs_pelvis_pitch_rad": [],
            "abs_root_angular_velocity_x_rad_s": [],
            "abs_root_angular_velocity_y_rad_s": [],
            "abs_root_linear_velocity_x_m_s": [],
            "abs_root_linear_velocity_y_m_s": [],
            "abs_base_height_error_m": [],
        }
        for joint_name in KEY_JOINT_NAMES:
            clean_tail[f"abs_joint_velocity_rad_s/{joint_name}"] = []
        trajectory = {name: [] for name in clean_tail}
        trajectory_active = []
        zero = torch.zeros((args_cli.num_envs, 14), device=unwrapped.device)
        finite = True
        completed_policy_steps = 0

        for step in range(args_cli.steps):
            tilt = _tilt(robot.data.projected_gravity_b)
            roll_abs, pitch_abs = _root_roll_pitch_abs(robot.data.projected_gravity_b)
            angular_speed = torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=-1)
            key_vel = torch.abs(robot.data.joint_vel[:, key_joint_ids])
            current_channels = {
                "abs_pelvis_roll_rad": roll_abs,
                "abs_pelvis_pitch_rad": pitch_abs,
                "abs_root_angular_velocity_x_rad_s": torch.abs(
                    robot.data.root_ang_vel_b[:, 0]
                ),
                "abs_root_angular_velocity_y_rad_s": torch.abs(
                    robot.data.root_ang_vel_b[:, 1]
                ),
                "abs_root_linear_velocity_x_m_s": torch.abs(robot.data.root_lin_vel_b[:, 0]),
                "abs_root_linear_velocity_y_m_s": torch.abs(robot.data.root_lin_vel_b[:, 1]),
                "abs_base_height_error_m": torch.abs(
                    robot.data.root_pos_w[:, 2] - A3_NOMINAL_BODY_HEIGHT_M
                ),
            }
            for joint_index, joint_name in enumerate(KEY_JOINT_NAMES):
                current_channels[f"abs_joint_velocity_rad_s/{joint_name}"] = key_vel[
                    :, joint_index
                ]
            for name, values in current_channels.items():
                trajectory[name].append(values.clone())
            trajectory_active.append(active.clone())
            max_tilt = torch.where(active, torch.maximum(max_tilt, tilt), max_tilt)
            max_ang_vel = torch.where(active, torch.maximum(max_ang_vel, angular_speed), max_ang_vel)
            root_xy_drift = torch.linalg.vector_norm(
                robot.data.root_pos_w[:, :2] - initial_root_xy, dim=-1
            )
            max_root_xy_drift = torch.where(
                active, torch.maximum(max_root_xy_drift, root_xy_drift), max_root_xy_drift
            )
            for name, tilt_limit, ang_limit in SETTLED_ENVELOPES:
                inside = (tilt <= tilt_limit) & (angular_speed <= ang_limit) & active
                settled_run[name] = torch.where(
                    inside, settled_run[name] + 1, torch.zeros_like(settled_run[name])
                )
                newly_settled = (
                    (settled_step[name] < 0)
                    & (settled_run[name] >= SETTLED_CONSECUTIVE_STEPS)
                )
                settled_step[name][newly_settled] = (
                    step - SETTLED_CONSECUTIVE_STEPS + 1
                )
            if step >= max(0, args_cli.steps - 100) and clean.any():
                clean_tail["abs_pelvis_roll_rad"].append(roll_abs[clean].clone())
                clean_tail["abs_pelvis_pitch_rad"].append(pitch_abs[clean].clone())
                clean_tail["abs_root_angular_velocity_x_rad_s"].append(
                    torch.abs(robot.data.root_ang_vel_b[clean, 0]).clone()
                )
                clean_tail["abs_root_angular_velocity_y_rad_s"].append(
                    torch.abs(robot.data.root_ang_vel_b[clean, 1]).clone()
                )
                clean_tail["abs_root_linear_velocity_x_m_s"].append(
                    torch.abs(robot.data.root_lin_vel_b[clean, 0]).clone()
                )
                clean_tail["abs_root_linear_velocity_y_m_s"].append(
                    torch.abs(robot.data.root_lin_vel_b[clean, 1]).clone()
                )
                clean_tail["abs_base_height_error_m"].append(
                    torch.abs(
                        robot.data.root_pos_w[clean, 2] - A3_NOMINAL_BODY_HEIGHT_M
                    ).clone()
                )
                for joint_index, joint_name in enumerate(KEY_JOINT_NAMES):
                    clean_tail[f"abs_joint_velocity_rad_s/{joint_name}"].append(
                        key_vel[clean, joint_index].clone()
                    )

            _obs, _reward, terminated, truncated, _extras = env.step(zero)
            done = (terminated | truncated) & active
            if done.any():
                timeout[done] = truncated[done] & (~terminated[done])
                for name in term_counts:
                    term_counts[name] += int(
                        (unwrapped.termination_manager.get_term(name) & done).sum().item()
                    )
                active[done] = False
            finite = finite and bool(
                torch.isfinite(robot.data.root_state_w).all()
                and torch.isfinite(robot.data.joint_pos).all()
                and torch.isfinite(robot.data.joint_vel).all()
            )
            completed_policy_steps = step + 1
            if completed_policy_steps == 1 or completed_policy_steps % 25 == 0:
                elapsed_s = time.monotonic() - profile_started_at
                heartbeat = {
                    "profile": profile_name,
                    "completed_policy_steps": completed_policy_steps,
                    "required_policy_steps": args_cli.steps,
                    "active_envs": int(active.sum().item()),
                    "finite": finite,
                    "elapsed_s": elapsed_s,
                    "steps_per_s": completed_policy_steps / max(elapsed_s, 1.0e-9),
                    "complete": False,
                }
                _write_json_atomic(heartbeat_path, heartbeat)
                print(
                    f"[{profile_name}] step {completed_policy_steps}/{args_cli.steps} "
                    f"active={heartbeat['active_envs']} finite={finite} "
                    f"steps/s={heartbeat['steps_per_s']:.2f}",
                    flush=True,
                )
            if not active.any():
                break

        elapsed_s = time.monotonic() - profile_started_at
        _write_json_atomic(
            heartbeat_path,
            {
                "profile": profile_name,
                "completed_policy_steps": completed_policy_steps,
                "required_policy_steps": args_cli.steps,
                "active_envs": int(active.sum().item()),
                "finite": finite,
                "elapsed_s": elapsed_s,
                "steps_per_s": completed_policy_steps / max(elapsed_s, 1.0e-9),
                "complete": completed_policy_steps == args_cli.steps,
            },
        )
        trajectory_path = args_cli.output.with_name(
            f"{args_cli.output.stem}.trajectory.npz"
        )
        trajectory_temporary = trajectory_path.with_name(
            f".{trajectory_path.name}.tmp.npz"
        )
        trajectory_payload = {
            name: torch.stack(values).cpu().numpy()
            for name, values in trajectory.items()
        }
        trajectory_payload.update(
            {
                "active": torch.stack(trajectory_active).cpu().numpy(),
                "trace_index": np.asarray(trace["trace_index"], dtype=np.int32),
                "disturbed": np.asarray(trace["disturbed"], dtype=np.bool_),
                "policy_dt_s": np.asarray([float(unwrapped.step_dt)], dtype=np.float32),
            }
        )
        np.savez_compressed(trajectory_temporary, **trajectory_payload)
        trajectory_temporary.replace(trajectory_path)
        trajectory_sha256 = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
        envelopes = {}
        for name, tilt_limit, ang_limit in SETTLED_ENVELOPES:
            recovered = settled_step[name] >= 0
            time_s = settled_step[name].float() * float(unwrapped.step_dt)
            envelopes[name] = {
                "tilt_rad": tilt_limit,
                "angular_velocity_rad_s": ang_limit,
                "consecutive_steps": SETTLED_CONSECUTIVE_STEPS,
                "clean_settled_fraction": _fraction(recovered[clean]),
                "disturbed_settled_fraction": _fraction(recovered[disturbed]),
                "disturbed_settle_time_s": _stats(time_s[disturbed & recovered]),
            }

        return {
            "profile": profile_name,
            "profile_id": profile_id,
            "roll_pitch_abs_rad": pose_abs_rad,
            "angular_velocity_abs_rad_s": angular_abs_rad_s,
            "realized_clean_fraction": float(clean.float().mean().item()),
            "trace_index_sha256": hashlib.sha256(
                np.asarray(trace["trace_index"], dtype=np.int32).tobytes()
            ).hexdigest(),
            "trace_index_min": int(np.min(trace["trace_index"])),
            "trace_index_max": int(np.max(trace["trace_index"])),
            "trace_index_count": int(np.asarray(trace["trace_index"]).size),
            "sample_bounds_passed": bool(
                torch.count_nonzero(initial_pose[clean]) == 0
                and torch.count_nonzero(initial_ang_vel[clean]) == 0
                and torch.max(torch.abs(initial_pose)) <= pose_abs_rad + 1.0e-7
                and torch.max(torch.abs(initial_ang_vel)) <= angular_abs_rad_s + 1.0e-7
            ),
            "clean_timeout_fraction": _fraction(timeout[clean]),
            "disturbed_timeout_fraction": _fraction(timeout[disturbed]),
            "clean_max_tilt_rad": _stats(max_tilt[clean]),
            "disturbed_max_tilt_rad": _stats(max_tilt[disturbed]),
            "clean_max_angular_velocity_rad_s": _stats(max_ang_vel[clean]),
            "disturbed_max_angular_velocity_rad_s": _stats(max_ang_vel[disturbed]),
            "clean_max_root_xy_drift_m": _stats(max_root_xy_drift[clean]),
            "disturbed_max_root_xy_drift_m": _stats(max_root_xy_drift[disturbed]),
            "clean_tail_statistics": {
                name: _stats(torch.cat(values) if values else torch.empty(0))
                for name, values in clean_tail.items()
            },
            "settled_envelopes": envelopes,
            "termination_term_counts": term_counts,
            "trajectory_path": str(trajectory_path.resolve()),
            "trajectory_sha256": trajectory_sha256,
            "trajectory_channels": list(trajectory),
            "completed_policy_steps": completed_policy_steps,
            "wall_time_s": elapsed_s,
            "policy_steps_per_s": completed_policy_steps / max(elapsed_s, 1.0e-9),
            "runtime_integrity_passed": finite and completed_policy_steps == args_cli.steps,
        }
    finally:
        env.close()


def main() -> int:
    global TRACE, TRACE_SHA256
    try:
        trace_path = args_cli.trace.expanduser().resolve()
        TRACE_SHA256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        runtime_contract_path = (
            args_cli.runtime_contract.expanduser().resolve()
            if args_cli.runtime_contract is not None
            else None
        )
        runtime_contract_sha256 = (
            hashlib.sha256(runtime_contract_path.read_bytes()).hexdigest()
            if runtime_contract_path is not None
            else None
        )
        runtime_contract = (
            json.loads(runtime_contract_path.read_text(encoding="utf-8"))
            if runtime_contract_path is not None
            else None
        )
        if runtime_contract is not None:
            contract_trace_sha = runtime_contract.get("trace", {}).get("sha256")
            if contract_trace_sha != TRACE_SHA256:
                raise ValueError(
                    "Runtime contract trace SHA mismatch: "
                    f"{contract_trace_sha} != {TRACE_SHA256}"
                )
        with np.load(trace_path, allow_pickle=False) as payload:
            TRACE = {name: payload[name].copy() for name in payload.files}
        required_trace_fields = {
            "schema_version",
            "trace_index",
            "profile_id",
            "roll_pitch_rad",
            "angular_velocity_rad_s",
        }
        missing_trace_fields = required_trace_fields - TRACE.keys()
        if missing_trace_fields:
            raise ValueError(f"Recovery trace is missing fields: {sorted(missing_trace_fields)}")
        requested_names = {item.strip() for item in args_cli.profiles.split(",") if item.strip()}
        if requested_names == {"all"}:
            selected_profiles = PROFILES
        else:
            known_names = {profile[0] for profile in PROFILES}
            unknown = requested_names - known_names
            if unknown:
                raise ValueError(f"Unknown recovery calibration profiles: {sorted(unknown)}")
            selected_profiles = tuple(
                profile for profile in PROFILES if profile[0] in requested_names
            )
        profiles = [
            _run_profile(name, profile_id, pose_abs, angular_abs)
            for name, profile_id, pose_abs, angular_abs in selected_profiles
        ]
        runtime_passed = all(
            profile["runtime_integrity_passed"] and profile["sample_bounds_passed"]
            for profile in profiles
        )
        result = {
            "schema_version": 1,
            "audit_id": (
                "a3_base_stand_recovery_a_calibration_runtime_smoke_v1"
                if args_cli.runtime_smoke
                else "a3_base_stand_recovery_a_disturbance_calibration_v1"
            ),
            "task": "A3BaseStandRecoveryA-v0",
            "simulation_only": True,
            "controller": "Base14 PD_STAND plant plus exact zero residual",
            "num_envs_per_profile": args_cli.num_envs,
            "policy_steps": args_cli.steps,
            "policy_dt_s": 0.02,
            "runtime_smoke_only": bool(args_cli.runtime_smoke),
            "disturbance_trace_path": str(trace_path),
            "disturbance_trace_sha256": TRACE_SHA256,
            "runtime_contract_path": (
                str(runtime_contract_path) if runtime_contract_path is not None else None
            ),
            "runtime_contract_sha256": runtime_contract_sha256,
            "profiles": profiles,
            "runtime_integrity_passed": runtime_passed,
            "calibration_measured": runtime_passed and not args_cli.runtime_smoke,
            "recovery_training_approved": False,
            "deployment_approved": False,
        }
        _write_json_atomic(args_cli.output, result)
        print(json.dumps(result, indent=2))
        return 0 if runtime_passed else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
