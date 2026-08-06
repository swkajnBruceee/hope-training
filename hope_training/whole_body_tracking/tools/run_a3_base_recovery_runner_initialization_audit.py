#!/usr/bin/env python3
"""Audit the complete normalized Recovery-A Actor initialization chain."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--sample-count", type=int, default=200000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1 or args_cli.sample_count < 1000:
    parser.error("invalid audit counts")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import training.tasks.base_locomotion.config.a3  # noqa: F401
from training.tasks.base_locomotion.config.a3.agents.ppo import (
    A3BaseStandRecoveryAPPORunnerCfg,
)
from training.utils.a3_base_actor_init import initialize_zero_residual_actor_mean
from training.utils.my_on_policy_runner import MyOnPolicyRunner


def main() -> int:
    gym_env = None
    try:
        env_cfg = gym.spec("A3BaseStandRecoveryA-v0").kwargs["env_cfg_entry_point"]()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = 0
        env_cfg.sim.device = args_cli.device
        gym_env = gym.make("A3BaseStandRecoveryA-v0", cfg=env_cfg)
        vec_env = RslRlVecEnvWrapper(gym_env)
        runner_cfg = A3BaseStandRecoveryAPPORunnerCfg()
        runner_cfg.device = args_cli.device
        runner = MyOnPolicyRunner(
            vec_env, runner_cfg.to_dict(), log_dir=None, device=args_cli.device
        )
        output_layer = initialize_zero_residual_actor_mean(runner, action_dim=14)

        obs, _extras = vec_env.reset()
        normalized_obs = runner.obs_normalizer(obs)
        with torch.inference_mode():
            direct_mean = runner.alg.policy.actor(normalized_obs)
            inference_policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
            full_chain_mean = inference_policy(obs)

        std_tensor = runner.alg.policy.std.detach().flatten()
        if std_tensor.numel() == 1:
            std_by_joint = std_tensor.repeat(14)
        elif std_tensor.numel() == 14:
            std_by_joint = std_tensor
        else:
            raise RuntimeError(f"Unexpected policy std shape: {tuple(std_tensor.shape)}")
        generator = torch.Generator(device=vec_env.unwrapped.device)
        generator.manual_seed(20260719)
        sample = torch.randn(
            (args_cli.sample_count, 14),
            generator=generator,
            device=vec_env.unwrapped.device,
        ) * std_by_joint
        raw_clip_abs = float(vec_env.unwrapped.action_manager.get_term("base").cfg.raw_clip)
        clip_by_joint = (torch.abs(sample) >= raw_clip_abs).float().mean(dim=0)
        overall_clip = float((torch.abs(sample) >= raw_clip_abs).float().mean().item())

        output_weight_zero = bool(torch.count_nonzero(output_layer.weight) == 0)
        output_bias_zero = bool(torch.count_nonzero(output_layer.bias) == 0)
        direct_max_abs = float(torch.abs(direct_mean).max().item())
        full_chain_max_abs = float(torch.abs(full_chain_mean).max().item())
        finite = bool(
            torch.isfinite(obs).all()
            and torch.isfinite(normalized_obs).all()
            and torch.isfinite(full_chain_mean).all()
            and torch.isfinite(sample).all()
        )
        passed = bool(
            finite
            and output_weight_zero
            and output_bias_zero
            and direct_max_abs == 0.0
            and full_chain_max_abs == 0.0
            and torch.allclose(std_by_joint, torch.full_like(std_by_joint, 0.15))
            and overall_clip < 0.10
            and float(clip_by_joint.max().item()) < 0.105
        )
        result = {
            "schema_version": 1,
            "audit_id": "a3_base_recovery_runner_initialization_audit_v1",
            "task": "A3BaseStandRecoveryA-v0",
            "simulation_only": True,
            "num_envs": args_cli.num_envs,
            "actor_observation_dim": int(obs.shape[-1]),
            "empirical_normalization_enabled": bool(runner_cfg.empirical_normalization),
            "output_weight_exact_zero": output_weight_zero,
            "output_bias_exact_zero": output_bias_zero,
            "direct_normalized_actor_mean_max_abs": direct_max_abs,
            "full_raw_observation_to_actor_mean_max_abs": full_chain_max_abs,
            "init_noise_std_by_joint": [float(value) for value in std_by_joint.tolist()],
            "raw_action_clip_abs": raw_clip_abs,
            "gaussian_sample_count": args_cli.sample_count,
            "sampled_clip_fraction": overall_clip,
            "sampled_clip_fraction_by_joint": [
                float(value) for value in clip_by_joint.tolist()
            ],
            "runtime_integrity_passed": finite,
            "zero_actor_initialization_runtime_verified": passed,
            "untrained_stochastic_policy_safety_verified": False,
            "bounded_recovery_smoke_approved": False,
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
        if gym_env is not None:
            gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
