#!/usr/bin/env python3
"""Deterministically evaluate an A3 Base Stand smoke checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--task",
    choices=(
        "A3BaseStand-v0",
        "A3BaseStandAuthorityCandidate-v0",
        "A3BaseStandClipCandidate-v0",
        "A3BaseStandAuthorityClipCandidate-v0",
        "A3BaseStandPassiveStableCandidate-v0",
        "A3CatchReadyStand-v0",
    ),
    default="A3BaseStand-v0",
)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500, help="Policy steps; 500 equals one 10 s episode.")
parser.add_argument(
    "--raw-action-clip-abs",
    type=float,
    help="Optional diagnostic override in [0.25, 1.0]; omitted preserves the task config.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1 or args_cli.steps < 1:
    parser.error("--num-envs and --steps must be positive")
if args_cli.raw_action_clip_abs is not None and not 0.25 <= args_cli.raw_action_clip_abs <= 1.0:
    parser.error("--raw-action-clip-abs must be inside [0.25, 1.0]")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import training.tasks.base_locomotion.config.a3  # noqa: F401
from training.robots.agibot_a3 import A3_ANCHOR_BODY, A3_BASE_ACTION_JOINTS
from training.utils.ppo_cfg import load_ppo_params, runner_kwargs


def _tilt_rad(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.clamp(-projected_gravity_b[:, 2], min=-1.0, max=1.0))


def main() -> int:
    gym_env = None
    try:
        checkpoint = args_cli.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = 0
        env_cfg.sim.device = args_cli.device
        if args_cli.raw_action_clip_abs is not None:
            env_cfg.actions.base.raw_clip = args_cli.raw_action_clip_abs
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        vec_env = RslRlVecEnvWrapper(gym_env)

        runner_cfg = RslRlOnPolicyRunnerCfg(
            **runner_kwargs(load_ppo_params(), "a3_base_stand_smoke_eval")
        )
        runner_cfg.device = args_cli.device
        runner = OnPolicyRunner(
            vec_env, runner_cfg.to_dict(), log_dir=None, device=args_cli.device
        )
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        obs, _ = vec_env.reset()
        unwrapped = vec_env.unwrapped
        robot = unwrapped.scene["robot"]
        action_term = unwrapped.action_manager.get_term("base")
        raw_clip_abs = float(action_term.cfg.raw_clip)
        torso_ids, torso_names = robot.find_bodies([A3_ANCHOR_BODY], preserve_order=True)
        if torso_names != [A3_ANCHOR_BODY]:
            raise RuntimeError("A3 torso mapping changed")
        torso_id = torso_ids[0]

        ages = torch.zeros(args_cli.num_envs, dtype=torch.long, device=unwrapped.device)
        completed_lengths: list[int] = []
        termination_counts = {
            name: 0 for name in unwrapped.termination_manager.active_terms
        }
        joint_limit_term = unwrapped.termination_manager.get_term_cfg("joint_limit").func
        joint_limit_violation_counts = torch.zeros(
            robot.num_joints, dtype=torch.long, device=unwrapped.device
        )
        joint_limit_max_excess_rad = torch.zeros(
            robot.num_joints, device=unwrapped.device
        )
        nonfinite_steps = 0
        total_action_values = 0
        raw_clip_values = 0
        effective_clip_values = 0
        raw_action_abs_sum = 0.0
        effective_action_abs_sum = 0.0
        raw_action_abs_max = 0.0
        effective_action_abs_max = 0.0
        raw_action_sum_by_joint = torch.zeros(14, device=unwrapped.device)
        raw_action_abs_sum_by_joint = torch.zeros(14, device=unwrapped.device)
        raw_action_clip_count_by_joint = torch.zeros(
            14, dtype=torch.long, device=unwrapped.device
        )
        effective_action_sum_by_joint = torch.zeros(14, device=unwrapped.device)
        effective_action_min_by_joint = torch.full(
            (14,), math.inf, device=unwrapped.device
        )
        effective_action_max_by_joint = torch.full(
            (14,), -math.inf, device=unwrapped.device
        )
        reward_sum = 0.0
        root_height_min = math.inf
        root_tilt_max = 0.0
        torso_tilt_max = 0.0

        for _step in range(args_cli.steps):
            with torch.inference_mode():
                raw_action = policy(obs)
            obs, reward, dones, _extras = vec_env.step(raw_action)
            ages += 1
            done_mask = dones.bool()
            if done_mask.any():
                completed_lengths.extend(ages[done_mask].cpu().tolist())
                ages[done_mask] = 0

            effective_action = action_term.raw_actions
            total_action_values += raw_action.numel()
            raw_clip_values += int((torch.abs(raw_action) >= raw_clip_abs - 1.0e-6).sum().item())
            effective_clip_values += int(
                (torch.abs(effective_action) >= raw_clip_abs - 1.0e-6).sum().item()
            )
            raw_action_abs_sum += float(torch.abs(raw_action).sum().item())
            effective_action_abs_sum += float(torch.abs(effective_action).sum().item())
            raw_action_abs_max = max(raw_action_abs_max, float(torch.abs(raw_action).max().item()))
            effective_action_abs_max = max(
                effective_action_abs_max, float(torch.abs(effective_action).max().item())
            )
            raw_action_sum_by_joint += raw_action.sum(dim=0)
            raw_action_abs_sum_by_joint += torch.abs(raw_action).sum(dim=0)
            raw_action_clip_count_by_joint += (
                torch.abs(raw_action) >= raw_clip_abs - 1.0e-6
            ).sum(dim=0)
            effective_action_sum_by_joint += effective_action.sum(dim=0)
            effective_action_min_by_joint = torch.minimum(
                effective_action_min_by_joint, effective_action.amin(dim=0)
            )
            effective_action_max_by_joint = torch.maximum(
                effective_action_max_by_joint, effective_action.amax(dim=0)
            )
            reward_sum += float(reward.mean().item())
            for name in termination_counts:
                termination_counts[name] += int(
                    unwrapped.termination_manager.get_term(name).sum().item()
                )
            joint_limit_violation_counts += joint_limit_term.violation_mask.sum(dim=0)
            joint_limit_max_excess_rad = torch.maximum(
                joint_limit_max_excess_rad, joint_limit_term.excess_rad.amax(dim=0)
            )

            torso_quat = robot.data.body_quat_w[:, torso_id]
            torso_gravity = math_utils.quat_rotate_inverse(torso_quat, robot.data.GRAVITY_VEC_W)
            root_height_min = min(root_height_min, float(robot.data.root_pos_w[:, 2].min().item()))
            root_tilt_max = max(
                root_tilt_max, float(_tilt_rad(robot.data.projected_gravity_b).max().item())
            )
            torso_tilt_max = max(torso_tilt_max, float(_tilt_rad(torso_gravity).max().item()))
            finite = (
                torch.isfinite(obs).all()
                and torch.isfinite(reward).all()
                and torch.isfinite(raw_action).all()
                and torch.isfinite(robot.data.root_state_w).all()
                and torch.isfinite(robot.data.joint_pos).all()
                and torch.isfinite(robot.data.joint_vel).all()
            )
            if not bool(finite):
                nonfinite_steps += 1

        completed_count = len(completed_lengths)
        timeout_count = termination_counts.get("time_out", 0)
        joint_limit_by_joint = sorted(
            (
                {
                    "joint_name": joint_name,
                    "violation_count": int(count),
                    "max_excess_rad": float(excess),
                }
                for joint_name, count, excess in zip(
                    robot.joint_names,
                    joint_limit_violation_counts.tolist(),
                    joint_limit_max_excess_rad.tolist(),
                )
                if count > 0
            ),
            key=lambda item: item["violation_count"],
            reverse=True,
        )
        samples_per_joint = args_cli.num_envs * args_cli.steps
        action_statistics_by_joint = [
            {
                "joint_name": joint_name,
                "raw_mean": float(raw_action_sum_by_joint[index].item()) / samples_per_joint,
                "raw_mean_abs": float(raw_action_abs_sum_by_joint[index].item()) / samples_per_joint,
                "raw_clip_fraction": float(raw_action_clip_count_by_joint[index].item())
                / samples_per_joint,
                "effective_mean": float(effective_action_sum_by_joint[index].item())
                / samples_per_joint,
                "effective_min": float(effective_action_min_by_joint[index].item()),
                "effective_max": float(effective_action_max_by_joint[index].item()),
            }
            for index, joint_name in enumerate(A3_BASE_ACTION_JOINTS)
        ]
        result = {
            "schema_version": 1,
            "evaluation_id": "a3_base_stand_checkpoint_eval_v1",
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "num_envs": args_cli.num_envs,
            "policy_steps": args_cli.steps,
            "physics_dt_s": float(unwrapped.physics_dt),
            "policy_dt_s": float(unwrapped.step_dt),
            "deterministic_inference": True,
            "raw_action_clip_abs": raw_clip_abs,
            "non_default_clip_is_diagnostic_only": raw_clip_abs != 0.25,
            "completed_episode_count": completed_count,
            "timeout_episode_count": timeout_count,
            "timeout_fraction_of_completed": timeout_count / completed_count if completed_count else None,
            "mean_completed_episode_length_steps": (
                sum(completed_lengths) / completed_count if completed_count else None
            ),
            "min_completed_episode_length_steps": min(completed_lengths) if completed_lengths else None,
            "max_completed_episode_length_steps": max(completed_lengths) if completed_lengths else None,
            "censored_env_count_at_end": int((ages > 0).sum().item()),
            "termination_term_counts": termination_counts,
            "joint_limit_violations_by_joint": joint_limit_by_joint,
            "mean_reward_per_policy_step": reward_sum / args_cli.steps,
            "root_height_min_sampled_m": root_height_min,
            "root_tilt_max_sampled_rad": root_tilt_max,
            "torso_tilt_max_sampled_rad": torso_tilt_max,
            "raw_action_mean_abs": raw_action_abs_sum / total_action_values,
            "raw_action_max_abs": raw_action_abs_max,
            "raw_action_clip_fraction": raw_clip_values / total_action_values,
            "effective_action_mean_abs": effective_action_abs_sum / total_action_values,
            "effective_action_max_abs": effective_action_abs_max,
            "effective_action_clip_fraction": effective_clip_values / total_action_values,
            "action_statistics_by_joint": action_statistics_by_joint,
            "nonfinite_step_count": nonfinite_steps,
            "runtime_integrity_passed": nonfinite_steps == 0,
            "stand_phase1_qualified": False,
            "stand_long_training_approved": False,
            "deployment_approved": False,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["runtime_integrity_passed"] else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if gym_env is not None:
            gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
