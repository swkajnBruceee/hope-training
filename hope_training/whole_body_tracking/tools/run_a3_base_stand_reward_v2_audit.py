#!/usr/bin/env python3
"""Numerically audit Stand reward-v2 survival and termination semantics."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--healthy-steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.healthy_steps < 1:
    parser.error("--healthy-steps must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

import training.tasks.base_locomotion.config.a3  # noqa: F401


def _step_terms(env) -> dict[str, float]:
    manager = env.unwrapped.reward_manager
    return {
        name: float(manager._step_reward[0, index].item() * env.unwrapped.step_dt)
        for index, name in enumerate(manager.active_terms)
    }


def main() -> int:
    env = None
    try:
        cfg = gym.spec("A3BaseStandPassiveStableCandidate-v0").kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = 1
        cfg.seed = 0
        cfg.sim.device = args_cli.device
        env = gym.make("A3BaseStandPassiveStableCandidate-v0", cfg=cfg)
        env.reset(seed=0)
        zero = torch.zeros((1, 14), device=env.unwrapped.device)

        healthy_rewards = []
        healthy_terms = None
        for _ in range(args_cli.healthy_steps):
            _obs, reward, terminated, truncated, _extras = env.step(zero)
            if bool((terminated | truncated).item()):
                raise RuntimeError("Passive-stable candidate reset during healthy reward audit")
            healthy_rewards.append(float(reward[0].item()))
            healthy_terms = _step_terms(env)

        robot = env.unwrapped.scene["robot"]
        forced_pose = robot.data.root_state_w[:, :7].clone()
        forced_pose[:, 2] = 0.70
        robot.write_root_pose_to_sim(forced_pose)
        robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_vel_w))
        _obs, failure_reward, terminated, truncated, _extras = env.step(zero)
        failure_terms = _step_terms(env)
        failure_labels = [
            name
            for name in env.unwrapped.termination_manager.active_terms
            if bool(env.unwrapped.termination_manager.get_term(name)[0])
        ]

        dt = float(env.unwrapped.step_dt)
        mean_healthy = sum(healthy_rewards) / len(healthy_rewards)
        failure_penalty = failure_terms["termination_penalty"]
        equivalent_alive_seconds = abs(failure_penalty) / 1.0
        cutoff_returns = {}
        for seconds in (0.5, 1.0, 1.5, 2.0):
            steps = round(seconds / dt)
            cutoff_returns[f"{seconds:.1f}s"] = steps * mean_healthy + failure_penalty
        monotonic = list(cutoff_returns.values()) == sorted(cutoff_returns.values())

        passed = bool(
            mean_healthy > 0.0
            and bool(terminated.item())
            and not bool(truncated.item())
            and "base_height" in failure_labels
            and abs(failure_penalty + 2.0) <= 1.0e-6
            and equivalent_alive_seconds >= 2.0
            and monotonic
            and all(torch.isfinite(torch.tensor(list(failure_terms.values()))))
        )
        result = {
            "schema_version": 1,
            "audit_id": "a3_base_stand_reward_v2_audit_v1",
            "task": "A3BaseStandPassiveStableCandidate-v0",
            "simulation_only": True,
            "policy_dt_s": dt,
            "healthy_steps": args_cli.healthy_steps,
            "mean_healthy_reward_per_step": mean_healthy,
            "last_healthy_reward_terms": healthy_terms,
            "forced_failure": {
                "root_height_m": 0.70,
                "terminated": bool(terminated.item()),
                "truncated": bool(truncated.item()),
                "termination_labels": failure_labels,
                "reward": float(failure_reward[0].item()),
                "reward_terms": failure_terms,
            },
            "termination_penalty_equivalent_alive_seconds": equivalent_alive_seconds,
            "cutoff_return_without_discount": cutoff_returns,
            "longer_survival_return_is_monotonic": monotonic,
            "passed": passed,
            "additional_ppo_smoke_approved": False,
            "stand_long_training_approved": False,
            "deployment_approved": False,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if passed else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
