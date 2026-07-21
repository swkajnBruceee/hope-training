#!/usr/bin/env python3
"""Four-way deterministic causal evaluation for a frozen Recovery-A checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--task", choices=("A3BaseStandRecoveryA-v0", "A3BaseStandRecoveryAV2-v0", "A3BaseStandRecoveryAV2WaistMask-v0", "A3BaseStandRecoveryAV21WaistMask-v0"), default="A3BaseStandRecoveryA-v0")
parser.add_argument("--pairs", type=int, default=16)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--profiles", nargs="+", choices=("clean", "low", "medium"), default=["clean", "low", "medium"])
parser.add_argument("--healthy-tilt-rad", type=float, default=0.05)
parser.add_argument("--healthy-ang-vel-rad-s", type=float, default=0.20)
parser.add_argument("--healthy-dwell-steps", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--override-policy-std",
    type=float,
    default=None,
    help="Override the loaded Gaussian policy std for the stochastic exceed audit; mean network is unchanged.",
)
parser.add_argument("--gated-eval", action="store_true", help="Use smooth-gated and hysteresis hard-zero intervention modes.")
parser.add_argument("--gate-on", type=float, default=1.25)
parser.add_argument("--gate-off", type=float, default=0.90)
parser.add_argument("--gate-exit-dwell", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.pairs < 1 or args_cli.steps < 1 or args_cli.healthy_dwell_steps < 1:
    parser.error("pairs, steps, and healthy-dwell-steps must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import torch
import isaaclab.utils.math as math_utils
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import training.tasks.base_locomotion.config.a3  # noqa: F401
from training.robots.agibot_a3 import A3_BASE_ACTION_JOINTS
from training.utils.ppo_cfg import load_ppo_params, runner_kwargs


PROFILE_IDS = {"clean": 0, "low": 1, "medium": 2}
MODES = ("zero", "full", "waist_pitch_zero", "healthy_then_zero")


def _tilt(projected_gravity: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0))


def _select_rows(trace: dict[str, np.ndarray], profile_id: int, count: int) -> np.ndarray:
    candidates = np.flatnonzero(trace["profile_id"] == profile_id)
    if candidates.size < count:
        raise ValueError(f"profile {profile_id} has {candidates.size} rows, needs {count}")
    # Preserve directional coverage by taking evenly spaced rows from the fixed trace.
    return candidates[np.linspace(0, candidates.size - 1, count, dtype=np.int64)]


def _install_states(env, robot, pose_np: np.ndarray, velocity_np: np.ndarray) -> None:
    count = len(pose_np)
    device = env.device
    pose = torch.as_tensor(pose_np, device=device, dtype=torch.float32)
    velocity = torch.as_tensor(velocity_np, device=device, dtype=torch.float32)
    env_ids = torch.arange(count, device=device)
    root = robot.data.default_root_state.clone()
    root[:, :3] += env.scene.env_origins
    delta = math_utils.quat_from_euler_xyz(
        pose[:, 0], pose[:, 1], torch.zeros(count, device=device)
    )
    root[:, 3:7] = math_utils.quat_mul(root[:, 3:7], delta)
    root[:, 10:12] += velocity
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone(), env_ids=env_ids)
    env.recovery_disturbed_mask[:] = torch.linalg.vector_norm(pose, dim=-1) > 0.0
    env.recovery_initial_roll_pitch_rad[:] = pose
    env.recovery_initial_angular_velocity_rad_s[:] = velocity


def _run_profile(vec_env, env, robot, policy, obs, pose_np, velocity_np, waist_index: int, policy_std: torch.Tensor) -> tuple[dict, torch.Tensor]:
    pair_count = len(pose_np)
    device = env.device
    total = 4 * pair_count
    pose = np.concatenate([pose_np] * 4, axis=0)
    velocity = np.concatenate([velocity_np] * 4, axis=0)
    _install_states(env, robot, pose, velocity)
    obs, _ = vec_env.reset()
    _install_states(env, robot, pose, velocity)
    # Refresh observations after the explicit state write.
    obs = vec_env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    active = torch.ones(total, dtype=torch.bool, device=device)
    failed = torch.zeros(total, dtype=torch.bool, device=device)
    failure_step = torch.full((total,), args_cli.steps, dtype=torch.long, device=device)
    healthy_count = torch.zeros(total, dtype=torch.long, device=device)
    healthy_enter = torch.full((total,), args_cli.steps, dtype=torch.long, device=device)
    max_tilt = torch.zeros(total, device=device)
    min_height = torch.full((total,), float("inf"), device=device)
    action_sq = torch.zeros(total, device=device)
    waist_sum = torch.zeros(total, device=device)
    waist_abs_sum = torch.zeros(total, device=device)
    post_action_sq = torch.zeros(total, device=device)
    post_action_sum_by_joint = torch.zeros((total, 14), device=device)
    post_action_sq_by_joint = torch.zeros((total, 14), device=device)
    post_waist_sq = torch.zeros(total, device=device)
    post_count = torch.zeros(total, device=device)
    termination_counts = [{name: 0 for name in env.termination_manager.active_terms} for _ in MODES]
    last_action = torch.zeros((total, 14), device=device)
    raw_clip = float(env.action_manager.get_term("base").cfg.raw_clip)
    clip_count = torch.zeros(total, device=device)
    execution_clip_count = torch.zeros(total, device=device)
    sampled_exceed_count = torch.zeros(total, device=device)
    sampled_action_sq = torch.zeros(total, device=device)
    gate_value_sum = torch.zeros(total, device=device)
    gate_active_count = torch.zeros(total, device=device)
    gate_latched = torch.zeros(total, dtype=torch.bool, device=device)
    gate_exit_count = torch.zeros(total, dtype=torch.long, device=device)

    for step in range(args_cli.steps):
        with torch.inference_mode():
            actor_action = policy(obs)
        tilt_now = _tilt(robot.data.projected_gravity_b)
        ang_now = torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=-1)
        height_error = torch.abs(robot.data.root_pos_w[:, 2] - robot.data.default_root_state[:, 2])
        health_metric = torch.sqrt(
            (torch.square(tilt_now / max(args_cli.healthy_tilt_rad, 1.0e-6))
             + torch.square(ang_now / max(args_cli.healthy_ang_vel_rad_s, 1.0e-6))
             + torch.square(height_error / 0.02)) / (3.0**0.5))
        gate_latched |= health_metric > args_cli.gate_on
        below_off = health_metric < args_cli.gate_off
        gate_exit_count = torch.where(below_off & gate_latched, gate_exit_count + 1, torch.zeros_like(gate_exit_count))
        gate_latched &= gate_exit_count < args_cli.gate_exit_dwell
        gate_fraction = torch.clamp(
            (health_metric - args_cli.gate_off) / max(args_cli.gate_on - args_cli.gate_off, 1.0e-6), 0.0, 1.0
        )
        gate_fraction = gate_fraction * gate_fraction * (3.0 - 2.0 * gate_fraction)
        gate_fraction = torch.where(gate_latched, gate_fraction, torch.zeros_like(gate_fraction))
        action = actor_action.clone()
        action[:pair_count] = 0.0
        ready = healthy_count >= args_cli.healthy_dwell_steps
        if args_cli.gated_eval:
            action[2 * pair_count : 3 * pair_count] *= gate_fraction[2 * pair_count : 3 * pair_count].unsqueeze(-1)
            action[3 * pair_count : 4 * pair_count] *= gate_latched[3 * pair_count : 4 * pair_count].unsqueeze(-1)
        else:
            action[2 * pair_count : 3 * pair_count, waist_index] = 0.0
            action[3 * pair_count :][ready[3 * pair_count :]] = 0.0
        action[~active] = 0.0
        sampled = action + torch.randn_like(action) * policy_std
        obs, _reward, done, _extras = vec_env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]
        done = done.bool()
        for mode_index in range(4):
            sl = slice(mode_index * pair_count, (mode_index + 1) * pair_count)
            for name in termination_counts[mode_index]:
                termination_counts[mode_index][name] += int(env.termination_manager.get_term(name)[sl].sum().item())
        tilt = _tilt(robot.data.projected_gravity_b)
        ang = torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=-1)
        healthy = (tilt <= args_cli.healthy_tilt_rad) & (ang <= args_cli.healthy_ang_vel_rad_s)
        max_tilt = torch.maximum(max_tilt, tilt)
        min_height = torch.minimum(min_height, robot.data.root_pos_w[:, 2])
        action_sq += torch.sum(torch.square(action), dim=1)
        waist_sum += action[:, waist_index]
        waist_abs_sum += torch.abs(action[:, waist_index])
        clip_count += (torch.abs(action) >= raw_clip - 1.0e-6).sum(dim=1)
        execution_clip_count += (torch.abs(env.action_manager.get_term("base").raw_actions) >= raw_clip - 1.0e-6).sum(dim=1)
        sampled_exceed_count += (torch.abs(sampled) >= raw_clip).sum(dim=1)
        sampled_action_sq += torch.sum(torch.square(sampled), dim=1)
        gate_value_sum += gate_fraction
        gate_active_count += gate_latched
        post = ready
        post_action_sq[post] += torch.sum(torch.square(action[post]), dim=1)
        post_action_sum_by_joint[post] += action[post]
        post_action_sq_by_joint[post] += torch.square(action[post])
        post_waist_sq[post] += torch.square(action[post, waist_index])
        post_count[post] += 1.0
        healthy_count = torch.where(healthy, healthy_count + 1, torch.zeros_like(healthy_count))
        newly = (healthy_count >= args_cli.healthy_dwell_steps) & (healthy_enter == args_cli.steps)
        healthy_enter[newly] = step + 1 - args_cli.healthy_dwell_steps
        time_out = env.termination_manager.get_term("time_out")
        first_done = done & ~time_out & active
        failed[first_done] = True
        failure_step[first_done] = step + 1
        active &= ~done
        last_action = action

    summaries = {}
    for mode_index, mode in enumerate(MODES):
        sl = slice(mode_index * pair_count, (mode_index + 1) * pair_count)
        count = torch.clamp(post_count[sl], min=1.0)
        tail_mean = post_action_sum_by_joint[sl].sum(dim=0) / (post_count[sl].sum() + 1.0e-6)
        tail_second = post_action_sq_by_joint[sl].sum(dim=0) / (post_count[sl].sum() + 1.0e-6)
        tail_std = torch.sqrt(torch.clamp(tail_second - torch.square(tail_mean), min=0.0))
        scale = env.action_manager.get_term("base")._scale.squeeze(0)
        summaries[mode] = {
            "survival_fraction": float((~failed[sl]).float().mean().item()),
            "failure_step_min": int(failure_step[sl].min().item()),
            "healthy_dwell_fraction": float((healthy_enter[sl] < args_cli.steps).float().mean().item()),
            "healthy_enter_step_median": float(torch.median(healthy_enter[sl].float()).item()),
            "max_tilt_p95_rad": float(torch.quantile(max_tilt[sl], 0.95).item()),
            "minimum_height_p05_m": float(torch.quantile(min_height[sl], 0.05).item()),
            "action_rms": float(torch.sqrt(action_sq[sl].mean() / args_cli.steps).item()),
            "sampled_action_rms_estimate": float(torch.sqrt(sampled_action_sq[sl].mean() / args_cli.steps).item()),
            "waist_pitch_action_mean": float(waist_sum[sl].mean().item() / args_cli.steps),
            "waist_pitch_action_mean_abs": float(waist_abs_sum[sl].mean().item() / args_cli.steps),
            "post_healthy_action_rms": float(torch.sqrt((post_action_sq[sl] / count).mean()).item()),
            "post_healthy_waist_pitch_rms": float(torch.sqrt((post_waist_sq[sl] / count).mean()).item()),
            "post_healthy_action_mean_by_joint": [float(v) for v in tail_mean.tolist()],
            "post_healthy_action_std_by_joint": [float(v) for v in tail_std.tolist()],
            "post_healthy_action_rms_by_joint": [float(v) for v in torch.sqrt(torch.clamp(tail_second, min=0.0)).tolist()],
            "post_healthy_physical_residual_rms_by_joint": [float(v) for v in (torch.sqrt(torch.clamp(tail_second, min=0.0)) * scale).tolist()],
            "raw_clip_fraction": float((clip_count[sl].sum() / (pair_count * args_cli.steps * 14)).item()),
            "execution_clip_fraction": float((execution_clip_count[sl].sum() / (pair_count * args_cli.steps * 14)).item()),
            "sampled_raw_exceed_fraction_estimate": float((sampled_exceed_count[sl].sum() / (pair_count * args_cli.steps * 14)).item()),
            "gate_mean": float((gate_value_sum[sl].mean() / args_cli.steps).item()),
            "gate_active_fraction": float((gate_active_count[sl].mean() / args_cli.steps).item()),
            "termination_counts": termination_counts[mode_index],
            "cases": [
                {
                    "initial_roll_pitch_rad": [float(v) for v in pose_np[i].tolist()],
                    "initial_angular_velocity_rad_s": [float(v) for v in velocity_np[i].tolist()],
                    "survival": bool((~failed[sl][i]).item()),
                    "peak_tilt_rad": float(max_tilt[sl][i].item()),
                    "gate_active_fraction": float((gate_active_count[sl][i] / args_cli.steps).item()),
                    "action_rms": float(torch.sqrt(action_sq[sl][i] / args_cli.steps).item()),
                }
                for i in range(pair_count)
            ],
        }
    return summaries, obs


def main() -> int:
    gym_env = None
    try:
        checkpoint = args_cli.checkpoint.expanduser().resolve()
        trace_path = args_cli.trace.expanduser().resolve()
        if not checkpoint.is_file() or not trace_path.is_file():
            raise FileNotFoundError(checkpoint if not checkpoint.is_file() else trace_path)
        trace = dict(np.load(trace_path))
        env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
        env_cfg.scene.num_envs = 4 * args_cli.pairs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        vec_env = RslRlVecEnvWrapper(gym_env)
        env = vec_env.unwrapped
        robot = env.scene["robot"]
        runner_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(load_ppo_params(), "recovery_a_causal_intervention"))
        runner_cfg.device = args_cli.device
        runner = OnPolicyRunner(vec_env, runner_cfg.to_dict(), log_dir=None, device=args_cli.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        std = runner.alg.policy.std.detach().flatten()
        if std.numel() == 1:
            policy_std = std.repeat(14).to(env.device)
        elif std.numel() == 14:
            policy_std = std.to(env.device)
        else:
            raise RuntimeError(f"Unexpected policy std shape: {tuple(std.shape)}")
        if args_cli.override_policy_std is not None:
            if args_cli.override_policy_std <= 0.0:
                raise ValueError("--override-policy-std must be positive")
            policy_std = torch.full_like(policy_std, args_cli.override_policy_std)
        policy_std = policy_std.view(1, 14)
        waist_index = A3_BASE_ACTION_JOINTS.index("waist_pitch_joint")
        result_profiles = {}
        obs = None
        for profile in args_cli.profiles:
            selected = _select_rows(trace, PROFILE_IDS[profile], args_cli.pairs)
            result_profiles[profile], obs = _run_profile(
                vec_env,
                env,
                robot,
                policy,
                obs,
                trace["roll_pitch_rad"][selected],
                trace["angular_velocity_rad_s"][selected],
                waist_index,
                policy_std,
            )
        output = {
            "schema_version": 1,
            "evaluation_id": "recovery_a_causal_intervention_eval_v1",
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "trace": str(trace_path),
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "pairs": args_cli.pairs,
            "steps": args_cli.steps,
            "modes": list(MODES),
            "healthy_definition": {"tilt_rad": args_cli.healthy_tilt_rad, "ang_vel_rad_s": args_cli.healthy_ang_vel_rad_s, "dwell_steps": args_cli.healthy_dwell_steps},
            "policy_std_by_joint": [float(value) for value in policy_std.flatten().tolist()],
            "policy_std_override": args_cli.override_policy_std,
            "gated_eval": args_cli.gated_eval,
            "profiles": result_profiles,
            "simulation_only": True,
            "checkpoint_approved": False,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output, indent=2), flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if gym_env is not None:
            gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
