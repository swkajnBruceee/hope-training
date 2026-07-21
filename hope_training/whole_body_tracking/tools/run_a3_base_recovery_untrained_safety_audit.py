#!/usr/bin/env python3
"""Paired passive-vs-untrained stochastic Recovery-A safety audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--envelope-decision",
    type=Path,
    required=True,
    help="Approved B-core envelope decision.",
)
parser.add_argument("--profiles", default="clean,candidate,medium")
parser.add_argument("--pairs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument(
    "--stochastic-steps",
    type=int,
    default=None,
    help="Inject random residual only for the first N policy steps, then force zero residual for settling tail.",
)
parser.add_argument("--runtime-smoke", action="store_true")
parser.add_argument("--noise-std", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=20260719)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.pairs < 16 or args_cli.pairs % 16:
    parser.error("--pairs must be a multiple of 16 and at least 16")
if args_cli.noise_std <= 0.0:
    parser.error("--noise-std must be positive")
if args_cli.runtime_smoke:
    if not 1 <= args_cli.steps < 500:
        parser.error("runtime smoke requires 1 <= --steps < 500")
elif args_cli.steps != 500:
    parser.error("formal untrained safety audit requires exactly 500 policy steps")
if args_cli.stochastic_steps is None:
    args_cli.stochastic_steps = args_cli.steps
if not 1 <= args_cli.stochastic_steps <= args_cli.steps:
    parser.error("--stochastic-steps must satisfy 1 <= stochastic_steps <= steps")

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
from tools.analyze_a3_base_recovery_manual_review import episode_events, summarize_events
from training.tasks.base_locomotion.base_env_cfg import A3_NOMINAL_BODY_HEIGHT_M


PROFILE_IDS = {"clean": 0, "candidate": 1, "medium": 2, "upper": 3}
APPROVED_ENVELOPE = None
APPROVED_DWELL_S = None


def _tilt(projected_gravity: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0))


def _root_roll_pitch_abs(
    projected_gravity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pitch = torch.abs(torch.asin(torch.clamp(projected_gravity[:, 0], -1.0, 1.0)))
    roll = torch.abs(
        torch.atan2(-projected_gravity[:, 1], -projected_gravity[:, 2])
    )
    return roll, pitch


def _stats(values: torch.Tensor) -> dict:
    values = values.float()
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def _combined_quadrant(pose: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    pose_q = (pose[:, 0] < 0).astype(np.int64) * 2 + (pose[:, 1] < 0)
    velocity_q = (velocity[:, 0] < 0).astype(np.int64) * 2 + (velocity[:, 1] < 0)
    return pose_q * 4 + velocity_q


def _trace_rows(trace: dict[str, np.ndarray], profile_id: int, count: int) -> np.ndarray:
    indices = np.flatnonzero(trace["profile_id"] == profile_id)
    if profile_id == 0:
        return indices[:count]
    category = _combined_quadrant(
        trace["roll_pitch_rad"][indices], trace["angular_velocity_rad_s"][indices]
    )
    per_category = count // 16
    selected = []
    for category_id in range(16):
        candidates = indices[category == category_id]
        if candidates.size < per_category:
            raise RuntimeError(f"Trace category {category_id} is undersized")
        selected.extend(candidates[:per_category].tolist())
    return np.asarray(selected, dtype=np.int64)


def _run_profile(
    trace: dict[str, np.ndarray], profile_name: str, profile_id: int
) -> dict:
    pair_count = args_cli.pairs
    total_envs = 2 * pair_count
    selected = _trace_rows(trace, profile_id, pair_count)
    pose_np = trace["roll_pitch_rad"][selected]
    velocity_np = trace["angular_velocity_rad_s"][selected]

    cfg = gym.spec("A3BaseStandRecoveryA-v0").kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = total_envs
    cfg.seed = args_cli.seed
    cfg.sim.device = args_cli.device
    env = gym.make("A3BaseStandRecoveryA-v0", cfg=cfg)
    try:
        env.reset(seed=args_cli.seed)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        pose = torch.as_tensor(
            np.concatenate((pose_np, pose_np)), device=unwrapped.device, dtype=torch.float32
        )
        velocity = torch.as_tensor(
            np.concatenate((velocity_np, velocity_np)),
            device=unwrapped.device,
            dtype=torch.float32,
        )
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += unwrapped.scene.env_origins
        delta = math_utils.quat_from_euler_xyz(
            pose[:, 0], pose[:, 1], torch.zeros(total_envs, device=unwrapped.device)
        )
        root_state[:, 3:7] = math_utils.quat_mul(root_state[:, 3:7], delta)
        root_state[:, 10:12] += velocity
        env_ids = torch.arange(total_envs, device=unwrapped.device)
        robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
        disturbed = torch.linalg.vector_norm(pose, dim=-1) > 0.0
        unwrapped.recovery_disturbed_mask[:] = disturbed
        unwrapped.recovery_initial_roll_pitch_rad[:] = pose
        unwrapped.recovery_initial_angular_velocity_rad_s[:] = velocity

        passive_slice = slice(0, pair_count)
        stochastic_slice = slice(pair_count, total_envs)
        initial_xy = robot.data.root_pos_w[:, :2].clone()
        active = torch.ones(total_envs, dtype=torch.bool, device=unwrapped.device)
        timeout = torch.zeros_like(active)
        max_tilt = _tilt(robot.data.projected_gravity_b).clone()
        max_ang_vel = torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=-1)
        max_xy_drift = torch.zeros(total_envs, device=unwrapped.device)
        torque_square_sum = torch.zeros(total_envs, device=unwrapped.device)
        action_square_sum = torch.zeros(total_envs, device=unwrapped.device)
        sampled_action_rate_square_sum = torch.zeros(total_envs, device=unwrapped.device)
        effective_action_rate_square_sum = torch.zeros(total_envs, device=unwrapped.device)
        previous_sampled_action = torch.zeros((total_envs, 14), device=unwrapped.device)
        previous_effective_action = torch.zeros((total_envs, 14), device=unwrapped.device)
        action_value_count = 0
        sampled_clip_count = 0
        effective_clip_count = 0
        sampled_clip_count_by_joint = torch.zeros(14, dtype=torch.long, device=unwrapped.device)
        effective_clip_count_by_joint = torch.zeros(14, dtype=torch.long, device=unwrapped.device)
        termination_by_mode = {
            "passive": {name: 0 for name in unwrapped.termination_manager.active_terms},
            "stochastic": {name: 0 for name in unwrapped.termination_manager.active_terms},
        }
        generator = torch.Generator(device=unwrapped.device)
        generator.manual_seed(args_cli.seed)
        raw_clip = float(unwrapped.action_manager.get_term("base").cfg.raw_clip)
        finite = True
        completed_policy_steps = 0
        core_trajectory = {
            name: []
            for name in (
                "abs_pelvis_roll_rad",
                "abs_pelvis_pitch_rad",
                "abs_root_angular_velocity_x_rad_s",
                "abs_root_angular_velocity_y_rad_s",
                "abs_root_linear_velocity_x_m_s",
                "abs_root_linear_velocity_y_m_s",
                "abs_base_height_error_m",
            )
        }
        active_trajectory = []

        for step in range(args_cli.steps):
            tilt = _tilt(robot.data.projected_gravity_b)
            roll_abs, pitch_abs = _root_roll_pitch_abs(robot.data.projected_gravity_b)
            angular_speed = torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=-1)
            current_core = {
                "abs_pelvis_roll_rad": roll_abs,
                "abs_pelvis_pitch_rad": pitch_abs,
                "abs_root_angular_velocity_x_rad_s": torch.abs(
                    robot.data.root_ang_vel_b[:, 0]
                ),
                "abs_root_angular_velocity_y_rad_s": torch.abs(
                    robot.data.root_ang_vel_b[:, 1]
                ),
                "abs_root_linear_velocity_x_m_s": torch.abs(
                    robot.data.root_lin_vel_b[:, 0]
                ),
                "abs_root_linear_velocity_y_m_s": torch.abs(
                    robot.data.root_lin_vel_b[:, 1]
                ),
                "abs_base_height_error_m": torch.abs(
                    robot.data.root_pos_w[:, 2] - A3_NOMINAL_BODY_HEIGHT_M
                ),
            }
            for name, values in current_core.items():
                core_trajectory[name].append(values.clone())
            active_trajectory.append(active.clone())
            xy_drift = torch.linalg.vector_norm(
                robot.data.root_pos_w[:, :2] - initial_xy, dim=-1
            )
            max_tilt = torch.where(active, torch.maximum(max_tilt, tilt), max_tilt)
            max_ang_vel = torch.where(
                active, torch.maximum(max_ang_vel, angular_speed), max_ang_vel
            )
            max_xy_drift = torch.where(
                active, torch.maximum(max_xy_drift, xy_drift), max_xy_drift
            )
            action = torch.zeros((total_envs, 14), device=unwrapped.device)
            if step < args_cli.stochastic_steps:
                stochastic_action = torch.randn(
                    (pair_count, 14), generator=generator, device=unwrapped.device
                ) * args_cli.noise_std
            else:
                stochastic_action = torch.zeros(
                    (pair_count, 14), device=unwrapped.device
                )
            action[stochastic_slice] = stochastic_action
            action_square_sum += torch.sum(torch.square(action), dim=-1)
            sampled_action_rate_square_sum += torch.sum(
                torch.square(action - previous_sampled_action), dim=-1
            )
            previous_sampled_action.copy_(action)
            action_value_count += stochastic_action.numel()
            sampled_clipped = torch.abs(stochastic_action) >= raw_clip
            sampled_clip_count += int(sampled_clipped.sum().item())
            sampled_clip_count_by_joint += sampled_clipped.sum(dim=0)

            _obs, _reward, terminated, truncated, _extras = env.step(action)
            effective = unwrapped.action_manager.get_term("base").raw_actions[stochastic_slice]
            full_effective = torch.zeros_like(action)
            full_effective[stochastic_slice] = effective
            effective_action_rate_square_sum += torch.sum(
                torch.square(full_effective - previous_effective_action), dim=-1
            )
            previous_effective_action.copy_(full_effective)
            effective_clipped = torch.abs(effective) >= raw_clip - 1.0e-6
            effective_clip_count += int(effective_clipped.sum().item())
            effective_clip_count_by_joint += effective_clipped.sum(dim=0)
            torque_square_sum += torch.sum(torch.square(robot.data.applied_torque), dim=-1)
            done = (terminated | truncated) & active
            if done.any():
                timeout[done] = truncated[done] & (~terminated[done])
                for name in unwrapped.termination_manager.active_terms:
                    term = unwrapped.termination_manager.get_term(name) & done
                    termination_by_mode["passive"][name] += int(term[passive_slice].sum().item())
                    termination_by_mode["stochastic"][name] += int(
                        term[stochastic_slice].sum().item()
                    )
                active[done] = False
            finite = finite and bool(
                torch.isfinite(robot.data.root_state_w).all()
                and torch.isfinite(robot.data.joint_pos).all()
                and torch.isfinite(robot.data.joint_vel).all()
                and torch.isfinite(action).all()
            )
            completed_policy_steps = step + 1
            if completed_policy_steps == 1 or completed_policy_steps % 25 == 0:
                print(
                    f"[{profile_name}] step {completed_policy_steps}/{args_cli.steps} "
                    f"active={int(active.sum().item())} finite={finite}",
                    flush=True,
                )
            if not active.any():
                break

        print(f"[{profile_name}] rollout complete; computing paired metrics", flush=True)
        def mode_metrics(mode_slice: slice) -> dict:
            return {
                "timeout_fraction": float(timeout[mode_slice].float().mean().item()),
                "max_tilt_rad": _stats(max_tilt[mode_slice]),
                "max_angular_velocity_rad_s": _stats(max_ang_vel[mode_slice]),
                "max_root_xy_drift_m": _stats(max_xy_drift[mode_slice]),
                "action_rms": float(
                    torch.sqrt(
                        action_square_sum[mode_slice].sum()
                        / (pair_count * args_cli.steps * 14)
                    ).item()
                ),
                "sampled_action_rate_rms_per_policy_step": float(
                    torch.sqrt(
                        sampled_action_rate_square_sum[mode_slice].sum()
                        / (pair_count * args_cli.steps * 14)
                    ).item()
                ),
                "effective_action_rate_rms_per_policy_step": float(
                    torch.sqrt(
                        effective_action_rate_square_sum[mode_slice].sum()
                        / (pair_count * args_cli.steps * 14)
                    ).item()
                ),
                "applied_torque_rms_nm": float(
                    torch.sqrt(
                        torque_square_sum[mode_slice].sum()
                        / (pair_count * args_cli.steps * robot.num_joints)
                    ).item()
                ),
            }

        passive = mode_metrics(passive_slice)
        stochastic = mode_metrics(stochastic_slice)
        print(f"[{profile_name}] scalar metrics complete; copying core trajectory", flush=True)
        trajectory_np = {
            name: torch.stack(values).cpu().numpy()
            for name, values in core_trajectory.items()
        }
        active_np = torch.stack(active_trajectory).cpu().numpy()
        print(f"[{profile_name}] core trajectory copied; evaluating B envelope", flush=True)

        def recovery_metrics(mode_slice: slice, non_timeout_count: int) -> dict:
            if APPROVED_ENVELOPE is None or APPROVED_DWELL_S is None:
                raise RuntimeError("Approved recovery envelope was not loaded")
            events = episode_events(
                {
                    name: values[:, mode_slice]
                    for name, values in trajectory_np.items()
                },
                active_np[:, mode_slice],
                APPROVED_ENVELOPE,
                float(unwrapped.step_dt),
                dwell_s=APPROVED_DWELL_S,
            )
            summary = summarize_events(events, non_timeout_count)
            return {
                "transient_recovery_rate": summary["transient_recovery_rate"],
                "durable_recovery_rate": summary["durable_recovery_rate"],
                "final_1s_stable_rate": summary["final_1s_stable_rate"],
                "recovery_time_s": summary["recovery_time_s"],
                "durable_recovery_time_s": summary["durable_recovery_time_s"],
                "exit_cycle_count": summary["exit_cycle_count"],
            }

        non_timeout_stochastic = sum(
            count
            for name, count in termination_by_mode["stochastic"].items()
            if name != "time_out"
        )
        non_timeout_passive = sum(
            count
            for name, count in termination_by_mode["passive"].items()
            if name != "time_out"
        )
        passive_recovery = recovery_metrics(passive_slice, non_timeout_passive)
        stochastic_recovery = recovery_metrics(
            stochastic_slice, non_timeout_stochastic
        )
        print(f"[{profile_name}] B-envelope paired metrics complete", flush=True)
        passive_p90 = passive_recovery["recovery_time_s"]["p90"]
        stochastic_p90 = stochastic_recovery["recovery_time_s"]["p90"]
        recovery_p90_delta_s = (
            None
            if passive_p90 is None or stochastic_p90 is None
            else stochastic_p90 - passive_p90
        )
        recovery_deltas = {
            "transient_recovery_rate": (
                stochastic_recovery["transient_recovery_rate"]
                - passive_recovery["transient_recovery_rate"]
            ),
            "durable_recovery_rate": (
                stochastic_recovery["durable_recovery_rate"]
                - passive_recovery["durable_recovery_rate"]
            ),
            "final_1s_stable_rate": (
                stochastic_recovery["final_1s_stable_rate"]
                - passive_recovery["final_1s_stable_rate"]
            ),
            "recovery_time_p90_s": recovery_p90_delta_s,
        }
        per_joint_denominator = pair_count * args_cli.steps
        action_joint_names = list(unwrapped.action_manager.get_term("base").cfg.base_joint_names)
        sampled_clip_fraction_by_joint = {
            name: float(sampled_clip_count_by_joint[index].item() / per_joint_denominator)
            for index, name in enumerate(action_joint_names)
        }
        effective_clip_fraction_by_joint = {
            name: float(effective_clip_count_by_joint[index].item() / per_joint_denominator)
            for index, name in enumerate(action_joint_names)
        }
        formal_safe = bool(
            not args_cli.runtime_smoke
            and finite
            and completed_policy_steps == args_cli.steps
            and passive["timeout_fraction"] == 1.0
            and stochastic["timeout_fraction"] >= 0.99
            and non_timeout_stochastic <= non_timeout_passive
            and stochastic["max_tilt_rad"]["p95"]
            <= passive["max_tilt_rad"]["p95"] + 0.05
            and recovery_deltas["transient_recovery_rate"] >= -0.02
            and recovery_deltas["durable_recovery_rate"] >= -0.05
            and recovery_deltas["final_1s_stable_rate"] >= -0.05
            and (
                recovery_p90_delta_s is None
                or recovery_p90_delta_s <= 1.0
            )
            and sampled_clip_count / action_value_count < 0.10
        )
        return {
            "profile": profile_name,
            "profile_id": profile_id,
            "pair_count": pair_count,
            "trace_index_sha256": hashlib.sha256(
                trace["trace_index"][selected].astype(np.int32).tobytes()
            ).hexdigest(),
            "passive": passive,
            "stochastic": stochastic,
            "passive_recovery": passive_recovery,
            "stochastic_recovery": stochastic_recovery,
            "stochastic_minus_passive": recovery_deltas,
            "termination_by_mode": termination_by_mode,
            "sampled_clip_fraction": sampled_clip_count / action_value_count,
            "effective_clip_fraction": effective_clip_count / action_value_count,
            "sampled_clip_fraction_by_joint": sampled_clip_fraction_by_joint,
            "effective_clip_fraction_by_joint": effective_clip_fraction_by_joint,
            "runtime_integrity_passed": (
                finite and completed_policy_steps == args_cli.steps
            ),
            "completed_policy_steps": completed_policy_steps,
            "untrained_stochastic_profile_safe": formal_safe,
        }
    finally:
        print(f"[{profile_name}] closing environment", flush=True)
        env.close()
        print(f"[{profile_name}] environment closed", flush=True)


def main() -> int:
    global APPROVED_ENVELOPE, APPROVED_DWELL_S
    try:
        decision_path = args_cli.envelope_decision.expanduser().resolve()
        decision_sha256 = hashlib.sha256(decision_path.read_bytes()).hexdigest()
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        approved = decision.get("approved_envelope", {})
        if (
            decision.get("recovery_envelope_approved") is not True
            or approved.get("name") != "B_core_only"
            or decision.get("authorizes_untrained_stochastic_policy_safety_audit")
            is not True
            or decision.get("authorizes_ppo") is not False
        ):
            raise ValueError("Envelope decision does not authorize this paired safety audit")
        APPROVED_ENVELOPE = approved["channels"]
        APPROVED_DWELL_S = float(approved["dwell_s"])
        trace_path = args_cli.trace.expanduser().resolve()
        trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        with np.load(trace_path, allow_pickle=False) as payload:
            trace = {name: payload[name].copy() for name in payload.files}
        requested = [item.strip() for item in args_cli.profiles.split(",") if item.strip()]
        unknown = set(requested) - PROFILE_IDS.keys()
        if unknown:
            raise ValueError(f"Unknown profiles: {sorted(unknown)}")
        profiles = [
            _run_profile(trace, profile_name, PROFILE_IDS[profile_name])
            for profile_name in requested
        ]
        runtime_passed = all(
            profile["runtime_integrity_passed"]
            and profile["completed_policy_steps"] == args_cli.steps
            for profile in profiles
        )
        safety_verified = bool(
            not args_cli.runtime_smoke
            and runtime_passed
            and all(profile["untrained_stochastic_profile_safe"] for profile in profiles)
        )
        result = {
            "schema_version": 1,
            "audit_id": "a3_base_recovery_untrained_stochastic_safety_audit_v1",
            "task": "A3BaseStandRecoveryA-v0",
            "simulation_only": True,
            "runtime_smoke_only": bool(args_cli.runtime_smoke),
            "trace_path": str(trace_path),
            "trace_sha256": trace_sha256,
            "envelope_decision_path": str(decision_path),
            "envelope_decision_sha256": decision_sha256,
            "approved_envelope_name": approved["name"],
            "approved_dwell_s": APPROVED_DWELL_S,
            "approved_hysteresis_ratio": float(approved["hysteresis_ratio"]),
            "audit_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "noise_std": args_cli.noise_std,
            "stochastic_steps": args_cli.stochastic_steps,
            "settling_tail_steps": args_cli.steps - args_cli.stochastic_steps,
            "policy_steps": args_cli.steps,
            "profiles": profiles,
            "runtime_integrity_passed": runtime_passed,
            "untrained_stochastic_policy_safety_verified": safety_verified,
            "bounded_recovery_smoke_approved": False,
            "deployment_approved": False,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args_cli.output.with_name(f".{args_cli.output.name}.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args_cli.output)
        print(json.dumps(result, indent=2))
        return 0 if runtime_passed else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
