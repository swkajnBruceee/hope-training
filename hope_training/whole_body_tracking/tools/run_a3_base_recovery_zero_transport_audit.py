#!/usr/bin/env python3
"""End-to-end zero-action transport audit for the Recovery-A task."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
import types
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--expected-trace-sha256", required=True)
parser.add_argument("--envelope-decision", type=Path, required=True)
parser.add_argument("--expected-envelope-decision-sha256", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--group-size", type=int, default=16)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=20260719)
parser.add_argument("--trajectory-atol", type=float, default=1.0e-6)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.group_size < 1:
    parser.error("--group-size must be positive")
if args_cli.steps < 1:
    parser.error("--steps must be positive")
if args_cli.trajectory_atol < 0.0:
    parser.error("--trajectory-atol must be non-negative")
for option, value in (
    ("--expected-trace-sha256", args_cli.expected_trace_sha256),
    ("--expected-envelope-decision-sha256", args_cli.expected_envelope_decision_sha256),
):
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        parser.error(f"{option} must be a 64-character hexadecimal SHA-256")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab.utils.math as math_utils
import training.tasks.base_locomotion.config.a3  # noqa: F401
from training.robots.agibot_a3 import A3_BASE_ACTION_JOINTS
from training.tasks.base_locomotion.config.a3.agents.ppo import (
    A3BaseStandRecoveryAPPORunnerCfg,
)
from training.utils.a3_base_actor_init import initialize_zero_residual_actor_mean
from training.utils.my_on_policy_runner import MyOnPolicyRunner


GROUP_NAMES = ("passive_zero", "deterministic_actor_mean", "forced_zero_after_actor")
CORE_NAMES = (
    "root_height_m",
    "root_lin_vel_x_m_s",
    "root_lin_vel_y_m_s",
    "root_lin_vel_z_m_s",
    "root_ang_vel_x_rad_s",
    "root_ang_vel_y_rad_s",
    "root_ang_vel_z_rad_s",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: np.ndarray) -> dict[str, float]:
    values64 = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values64.mean()),
        "std": float(values64.std()),
        "rms": float(np.sqrt(np.mean(np.square(values64)))),
        "min": float(values64.min()),
        "max": float(values64.max()),
        "max_abs": float(np.abs(values64).max()),
    }


def _per_joint_summary(values: np.ndarray, joint_names: list[str]) -> dict[str, dict[str, float]]:
    if values.shape[-1] != len(joint_names):
        raise ValueError(f"Expected final dimension {len(joint_names)}, got {values.shape}")
    return {
        name: _summary(values[..., index])
        for index, name in enumerate(joint_names)
    }


def _tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(value.detach().to(device="cpu", dtype=torch.float32).numpy())


def _array_hash(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _paired_max_differences(value: np.ndarray) -> dict[str, float]:
    return {
        "passive_zero_vs_deterministic_actor_mean": float(
            np.max(np.abs(value[:, 0] - value[:, 1]))
        ),
        "passive_zero_vs_forced_zero_after_actor": float(
            np.max(np.abs(value[:, 0] - value[:, 2]))
        ),
        "deterministic_actor_mean_vs_forced_zero_after_actor": float(
            np.max(np.abs(value[:, 1] - value[:, 2]))
        ),
    }


def _manager_buffer(manager, name: str) -> torch.Tensor | None:
    try:
        value = getattr(manager, name)
    except (AttributeError, RuntimeError):
        return None
    return value if isinstance(value, torch.Tensor) else None


def _buffer_report(value: torch.Tensor | None) -> dict:
    if value is None:
        return {"available": False, "exact_zero": None, "max_abs": None}
    return {
        "available": True,
        "exact_zero": bool(torch.count_nonzero(value) == 0),
        "max_abs": float(torch.abs(value).max().item()),
    }


def _validate_inputs() -> tuple[Path, str, Path, str, dict, dict[str, np.ndarray]]:
    trace_path = args_cli.trace.expanduser().resolve()
    decision_path = args_cli.envelope_decision.expanduser().resolve()
    trace_sha256 = _sha256(trace_path)
    decision_sha256 = _sha256(decision_path)
    if trace_sha256.lower() != args_cli.expected_trace_sha256.lower():
        raise ValueError(
            f"Trace SHA-256 mismatch: expected={args_cli.expected_trace_sha256}, actual={trace_sha256}"
        )
    if decision_sha256.lower() != args_cli.expected_envelope_decision_sha256.lower():
        raise ValueError(
            "Envelope decision SHA-256 mismatch: "
            f"expected={args_cli.expected_envelope_decision_sha256}, actual={decision_sha256}"
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    approved = decision.get("approved_envelope", {})
    if (
        decision.get("recovery_envelope_approved") is not True
        or approved.get("name") != "B_core_only"
        or decision.get("authorizes_ppo") is not False
    ):
        raise ValueError("Envelope decision is not the approved B_core_only no-PPO decision")
    with np.load(trace_path, allow_pickle=False) as payload:
        trace = {name: payload[name].copy() for name in payload.files}
    required = {
        "trace_index",
        "profile_id",
        "roll_pitch_rad",
        "angular_velocity_rad_s",
    }
    if not required.issubset(trace):
        raise ValueError(f"Trace is missing fields: {sorted(required - trace.keys())}")
    return trace_path, trace_sha256, decision_path, decision_sha256, decision, trace


def _select_clean_trace(trace: dict[str, np.ndarray]) -> np.ndarray:
    clean = np.flatnonzero(trace["profile_id"] == 0)
    if clean.size < args_cli.group_size:
        raise ValueError(
            f"Clean trace has {clean.size} rows, requires {args_cli.group_size}"
        )
    selected = clean[: args_cli.group_size]
    if not np.all(trace["roll_pitch_rad"][selected] == 0.0):
        raise ValueError("Selected clean trace contains nonzero roll/pitch")
    if not np.all(trace["angular_velocity_rad_s"][selected] == 0.0):
        raise ValueError("Selected clean trace contains nonzero angular velocity")
    return selected


def _write_identical_trace_state(
    unwrapped, robot, trace: dict[str, np.ndarray], selected: np.ndarray
) -> None:
    pose_one = trace["roll_pitch_rad"][selected]
    velocity_one = trace["angular_velocity_rad_s"][selected]
    pose = torch.as_tensor(
        np.concatenate((pose_one, pose_one, pose_one)),
        device=unwrapped.device,
        dtype=torch.float32,
    )
    velocity = torch.as_tensor(
        np.concatenate((velocity_one, velocity_one, velocity_one)),
        device=unwrapped.device,
        dtype=torch.float32,
    )
    total_envs = 3 * args_cli.group_size
    env_ids = torch.arange(total_envs, device=unwrapped.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += unwrapped.scene.env_origins
    delta = math_utils.quat_from_euler_xyz(
        pose[:, 0], pose[:, 1], torch.zeros(total_envs, device=unwrapped.device)
    )
    root_state[:, 3:7] = math_utils.quat_mul(root_state[:, 3:7], delta)
    root_state[:, 10:12] += velocity
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
        env_ids=env_ids,
    )
    disturbed = torch.linalg.vector_norm(pose, dim=-1) > 0.0
    unwrapped.recovery_disturbed_mask[:] = disturbed
    unwrapped.recovery_initial_roll_pitch_rad[:] = pose
    unwrapped.recovery_initial_angular_velocity_rad_s[:] = velocity


def _grouped(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(3, args_cli.group_size, *value.shape[1:])


def _root_local_state(unwrapped, robot) -> torch.Tensor:
    root = robot.data.root_state_w.clone()
    root[:, :3] -= unwrapped.scene.env_origins
    return root


def _core_state(robot) -> torch.Tensor:
    return torch.cat(
        (
            robot.data.root_pos_w[:, 2:3],
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
        ),
        dim=-1,
    )


def _install_zoh_probe(action_term) -> tuple[dict, object]:
    probe = {"apply_count": 0, "targets": [], "error": None}
    original = action_term.apply_actions

    def wrapped(self):
        probe["apply_count"] += 1
        probe["targets"].append(self.full_joint_targets.detach().clone())
        return original()

    try:
        action_term.apply_actions = types.MethodType(wrapped, action_term)
    except BaseException as error:
        probe["error"] = f"{type(error).__name__}: {error}"
    return probe, original


def _restore_zoh_probe(action_term, original) -> None:
    action_term.apply_actions = original


def main() -> int:
    gym_env = None
    try:
        (
            trace_path,
            trace_sha256,
            decision_path,
            decision_sha256,
            decision,
            trace,
        ) = _validate_inputs()
        selected = _select_clean_trace(trace)
        selected_trace_index = trace["trace_index"][selected].astype(np.int32)
        selected_trace_hash = hashlib.sha256(selected_trace_index.tobytes()).hexdigest()

        env_cfg = gym.spec("A3BaseStandRecoveryA-v0").kwargs["env_cfg_entry_point"]()
        env_cfg.scene.num_envs = 3 * args_cli.group_size
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device
        gym_env = gym.make("A3BaseStandRecoveryA-v0", cfg=env_cfg)
        vec_env = RslRlVecEnvWrapper(gym_env)
        unwrapped = vec_env.unwrapped
        robot = unwrapped.scene["robot"]
        action_term = unwrapped.action_manager.get_term("base")

        runner_cfg = A3BaseStandRecoveryAPPORunnerCfg()
        runner_cfg.device = args_cli.device
        runner = MyOnPolicyRunner(
            vec_env, runner_cfg.to_dict(), log_dir=None, device=args_cli.device
        )
        initialize_zero_residual_actor_mean(runner, action_dim=14)
        inference_policy = runner.get_inference_policy(device=unwrapped.device)

        obs, _extras = vec_env.reset()
        _write_identical_trace_state(unwrapped, robot, trace, selected)

        base_joint_ids, resolved_names = robot.find_joints(
            list(A3_BASE_ACTION_JOINTS), preserve_order=True
        )
        base_joint_names = list(A3_BASE_ACTION_JOINTS)
        joint_order_passed = bool(
            resolved_names == base_joint_names
            and list(action_term.cfg.base_joint_names) == base_joint_names
            and list(action_term._base_joint_ids) == list(base_joint_ids)
        )
        base_ids_tensor = torch.tensor(
            base_joint_ids, dtype=torch.long, device=unwrapped.device
        )
        non_base_ids = [
            index for index in range(robot.num_joints) if index not in set(base_joint_ids)
        ]
        non_base_ids_tensor = torch.tensor(
            non_base_ids, dtype=torch.long, device=unwrapped.device
        )
        scale = action_term._scale.detach().clone()
        raw_clip = float(action_term.cfg.raw_clip)
        default_full = robot.data.default_joint_pos.detach().clone()
        default_base = default_full[:, base_ids_tensor]

        records: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "actor_mean",
                "manager_input",
                "raw_action",
                "processed_target",
                "scaled_residual",
                "full_joint_target",
                "base_actual_q",
                "tracking_error",
                "applied_torque",
                "root_state",
                "core_state",
            )
        }
        runtime_finite = True
        actor_mean_exact_zero = {
            "passive_zero": None,
            "deterministic_actor_mean": True,
            "forced_zero_after_actor": True,
        }
        raw_zero = True
        residual_zero = True
        target_default = True
        scale_applied_once = True
        non_integrating_passed = True
        default_added_once = True
        non_base_default_passed = True
        zoh_probe = None
        zoh_original = None
        completed_steps = 0

        actor_slice = slice(args_cli.group_size, 3 * args_cli.group_size)
        b_slice = slice(args_cli.group_size, 2 * args_cli.group_size)
        c_slice = slice(2 * args_cli.group_size, 3 * args_cli.group_size)
        for step in range(args_cli.steps):
            with torch.inference_mode():
                actor_mean = inference_policy(obs[actor_slice])
            actor_mean_exact_zero["deterministic_actor_mean"] &= bool(
                torch.count_nonzero(actor_mean[: args_cli.group_size]) == 0
            )
            actor_mean_exact_zero["forced_zero_after_actor"] &= bool(
                torch.count_nonzero(actor_mean[args_cli.group_size :]) == 0
            )
            action = torch.zeros(
                (3 * args_cli.group_size, 14),
                device=unwrapped.device,
                dtype=actor_mean.dtype,
            )
            action[b_slice] = actor_mean[: args_cli.group_size]
            action[c_slice] = 0.0
            actor_record = torch.zeros_like(action)
            actor_record[actor_slice] = actor_mean

            if step == 0:
                zoh_probe, zoh_original = _install_zoh_probe(action_term)
            try:
                obs, _reward, _done, _extras = vec_env.step(action)
            finally:
                if step == 0 and zoh_original is not None:
                    _restore_zoh_probe(action_term, zoh_original)

            manager_input = _manager_buffer(unwrapped.action_manager, "action")
            if manager_input is None:
                raise RuntimeError("ActionManager.action is unavailable")
            raw = action_term.raw_actions.detach()
            processed = action_term.processed_actions.detach()
            full_target = action_term.full_joint_targets.detach()
            residual = processed - default_base
            expected_raw = torch.clamp(manager_input, -raw_clip, raw_clip)
            expected_processed = default_base + expected_raw * scale
            expected_full = default_full.clone()
            expected_full[:, base_ids_tensor] = expected_processed
            if action_term.cfg.clip_to_soft_joint_limits:
                limits = robot.data.soft_joint_pos_limits
                expected_full = torch.clamp(
                    expected_full, min=limits[..., 0], max=limits[..., 1]
                )
                expected_processed = expected_full[:, base_ids_tensor]

            raw_zero &= bool(torch.count_nonzero(raw) == 0)
            residual_zero &= bool(torch.count_nonzero(residual) == 0)
            target_default &= bool(torch.equal(processed, default_base))
            scale_applied_once &= bool(
                torch.allclose(residual, expected_raw * scale, atol=1.0e-7, rtol=0.0)
            )
            non_integrating_passed &= bool(
                torch.allclose(processed, expected_processed, atol=1.0e-7, rtol=0.0)
            )
            default_added_once &= bool(
                torch.allclose(full_target, expected_full, atol=1.0e-7, rtol=0.0)
            )
            if non_base_ids:
                non_base_default_passed &= bool(
                    torch.equal(
                        full_target[:, non_base_ids_tensor],
                        default_full[:, non_base_ids_tensor],
                    )
                )

            base_q = robot.data.joint_pos[:, base_ids_tensor]
            base_torque = robot.data.applied_torque[:, base_ids_tensor]
            current = {
                "actor_mean": actor_record,
                "manager_input": manager_input,
                "raw_action": raw,
                "processed_target": processed,
                "scaled_residual": residual,
                "full_joint_target": full_target,
                "base_actual_q": base_q,
                "tracking_error": processed - base_q,
                "applied_torque": base_torque,
                "root_state": _root_local_state(unwrapped, robot),
                "core_state": _core_state(robot),
            }
            for name, value in current.items():
                records[name].append(_grouped(value).detach().clone())
            runtime_finite &= all(
                bool(torch.isfinite(value).all()) for value in current.values()
            )
            completed_steps = step + 1
            if completed_steps == 1 or completed_steps % 25 == 0:
                print(
                    f"[zero-transport] step {completed_steps}/{args_cli.steps} "
                    f"finite={runtime_finite}",
                    flush=True,
                )

        if zoh_probe is None:
            raise RuntimeError("ZOH probe was not installed")
        zoh_targets = zoh_probe["targets"]
        zoh_identical = bool(
            zoh_targets
            and all(torch.equal(zoh_targets[0], value) for value in zoh_targets[1:])
        )
        zoh_apply_count = int(zoh_probe["apply_count"])
        zoh_passed = bool(
            zoh_probe["error"] is None
            and zoh_apply_count == int(env_cfg.decimation)
            and zoh_identical
        )

        stacked = {
            name: _tensor_numpy(torch.stack(values, dim=0))
            for name, values in records.items()
        }
        group_metrics = {}
        for group_index, group_name in enumerate(GROUP_NAMES):
            group_metrics[group_name] = {
                "actor_called": group_name != "passive_zero",
                "actor_mean_exact_zero": actor_mean_exact_zero[group_name],
                "base14_by_joint": {
                    signal: _per_joint_summary(
                        stacked[signal][:, group_index], base_joint_names
                    )
                    for signal in (
                        "actor_mean",
                        "manager_input",
                        "raw_action",
                        "processed_target",
                        "scaled_residual",
                        "base_actual_q",
                        "tracking_error",
                        "applied_torque",
                    )
                },
                "full_joint_target_by_joint": _per_joint_summary(
                    stacked["full_joint_target"][:, group_index],
                    list(robot.joint_names),
                ),
                "root_state": _summary(stacked["root_state"][:, group_index]),
                "core_by_channel": {
                    name: _summary(stacked["core_state"][:, group_index, ..., index])
                    for index, name in enumerate(CORE_NAMES)
                },
            }

        paired = {}
        hashes = {}
        for name, value in stacked.items():
            paired[name] = _paired_max_differences(value)
            hashes[name] = {
                group_name: _array_hash(value[:, group_index])
                for group_index, group_name in enumerate(GROUP_NAMES)
            }
        effective_equal = all(
            max(paired[name].values()) == 0.0
            for name in (
                "manager_input",
                "raw_action",
                "processed_target",
                "scaled_residual",
                "full_joint_target",
            )
        )
        root_q_within_tolerance = all(
            max(paired[name].values()) <= args_cli.trajectory_atol
            for name in ("root_state", "core_state", "base_actual_q")
        )

        # This probe intentionally runs after all main comparison statistics.
        nonzero_action = torch.zeros(
            (3 * args_cli.group_size, 14), device=unwrapped.device
        )
        nonzero_action[:, 0] = min(0.05, raw_clip * 0.5)
        vec_env.step(nonzero_action)
        reset_before = {
            "action_term_raw": _buffer_report(action_term.raw_actions),
            "action_manager_action": _buffer_report(
                _manager_buffer(unwrapped.action_manager, "action")
            ),
            "action_manager_prev_action": _buffer_report(
                _manager_buffer(unwrapped.action_manager, "prev_action")
            ),
        }
        obs, _extras = vec_env.reset()
        reset_after = {
            "action_term_raw": _buffer_report(action_term.raw_actions),
            "action_manager_action": _buffer_report(
                _manager_buffer(unwrapped.action_manager, "action")
            ),
            "action_manager_prev_action": _buffer_report(
                _manager_buffer(unwrapped.action_manager, "prev_action")
            ),
        }
        zero_action = torch.zeros(
            (3 * args_cli.group_size, 14), device=unwrapped.device
        )
        vec_env.step(zero_action)
        first_zero_raw = action_term.raw_actions.detach()
        first_zero_residual = (
            action_term.processed_actions.detach()
            - robot.data.default_joint_pos[:, base_ids_tensor]
        )
        first_zero_target = action_term.full_joint_targets.detach()
        first_zero_step = {
            "raw_exact_zero": bool(torch.count_nonzero(first_zero_raw) == 0),
            "scaled_residual_exact_zero": bool(
                torch.count_nonzero(first_zero_residual) == 0
            ),
            "target_default": bool(
                torch.equal(first_zero_target, robot.data.default_joint_pos)
            ),
            "runtime_finite": bool(
                torch.isfinite(first_zero_raw).all()
                and torch.isfinite(first_zero_residual).all()
                and torch.isfinite(first_zero_target).all()
            ),
        }
        first_zero_step["passed"] = all(first_zero_step.values())

        runtime_finite = bool(
            runtime_finite
            and completed_steps == args_cli.steps
            and first_zero_step["runtime_finite"]
        )
        actor_chain_zero = bool(
            actor_mean_exact_zero["deterministic_actor_mean"]
            and actor_mean_exact_zero["forced_zero_after_actor"]
        )
        command_transport_verified = bool(
            runtime_finite
            and actor_chain_zero
            and raw_zero
            and residual_zero
            and target_default
            and effective_equal
            and joint_order_passed
            and scale_applied_once
            and non_integrating_passed
            and default_added_once
            and non_base_default_passed
            and zoh_passed
            and first_zero_step["passed"]
        )
        transport_audit_passed = bool(
            command_transport_verified and root_q_within_tolerance
        )
        result = {
            "schema_version": 1,
            "audit_id": "a3_base_recovery_zero_transport_audit_v1",
            "task": "A3BaseStandRecoveryA-v0",
            "diagnostic_only": True,
            "checkpoint_loaded": False,
            "training_run": False,
            "trace_path": str(trace_path),
            "trace_sha256": trace_sha256,
            "trace_sha256_verified": True,
            "selected_clean_trace_index": selected_trace_index.tolist(),
            "selected_clean_trace_index_sha256": selected_trace_hash,
            "same_clean_trace_index_reused_by_all_groups": True,
            "envelope_decision_path": str(decision_path),
            "envelope_decision_sha256": decision_sha256,
            "envelope_decision_sha256_verified": True,
            "approved_envelope_name": decision["approved_envelope"]["name"],
            "envelope_b_approved": True,
            "authorizes_ppo": False,
            "group_size": args_cli.group_size,
            "policy_steps": args_cli.steps,
            "physics_decimation": int(env_cfg.decimation),
            "action_joint_order": base_joint_names,
            "joint_order_passed": joint_order_passed,
            "action_scale_rad": [float(value) for value in scale.flatten().tolist()],
            "raw_clip_abs": raw_clip,
            "default_target_by_joint": {
                name: float(default_base[0, index].item())
                for index, name in enumerate(base_joint_names)
            },
            "actor_mean_exact_zero": actor_mean_exact_zero,
            "raw_zero_to_scaled_residual_zero_to_target_default": {
                "raw_zero": raw_zero,
                "scaled_residual_zero": residual_zero,
                "target_default": target_default,
                "passed": bool(raw_zero and residual_zero and target_default),
            },
            "scale_applied_once": scale_applied_once,
            "non_integrating_passed": non_integrating_passed,
            "default_added_once": default_added_once,
            "non_base_default_passed": non_base_default_passed,
            "zoh_probe": {
                "zoh_apply_count": zoh_apply_count,
                "expected_apply_count": int(env_cfg.decimation),
                "zoh_targets_identical": zoh_identical,
                "instrumentation_error": zoh_probe["error"],
                "passed": zoh_passed,
            },
            "group_metrics": group_metrics,
            "paired_trajectory_max_abs_difference": paired,
            "trajectory_sha256": hashes,
            "effective_action_residual_target_identical": effective_equal,
            "zero_action_command_transport_verified": command_transport_verified,
            "root_q_trajectory_within_tolerance": root_q_within_tolerance,
            "parallel_physics_exact_replication_verified": root_q_within_tolerance,
            "parallel_physics_replication_note": (
                "Command and target transport are identical across all modes. "
                "Exact root/q replay across independent parallel PhysX environments "
                "is tracked separately and is not implied by command equivalence."
            ),
            "trajectory_atol": args_cli.trajectory_atol,
            "reset_residue_probe": {
                "after_nonzero_before_reset": reset_before,
                "immediately_after_reset": reset_after,
                "immediate_buffer_clear_is_informational_only": True,
                "first_zero_step": first_zero_step,
            },
            "completed_policy_steps": completed_steps,
            "runtime_finite": runtime_finite,
            "transport_audit_passed": transport_audit_passed,
            "ppo_approved": False,
            "deployment_approved": False,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args_cli.output.with_name(f".{args_cli.output.name}.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args_cli.output)
        print(json.dumps(result, indent=2))
        return 0 if transport_audit_passed else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if gym_env is not None:
            gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
