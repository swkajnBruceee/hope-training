#!/usr/bin/env python3
"""Deterministic target-position causality probe for V1.3B checkpoints.

This is evaluation-only.  It never calls PPO update and does not touch an
active training run.  Each checkpoint is evaluated on the same seven local
position targets: nominal and +/-5 cm along each local axis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--motion-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--upper-alphas", nargs="+", type=float, default=[0.9])
    args = ap.parse_args()
    for p in args.checkpoints:
        if not p.is_file():
            raise SystemExit(f"checkpoint does not exist: {p}")
    if not args.motion_manifest.is_file():
        raise SystemExit(f"motion manifest does not exist: {args.motion_manifest}")

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=args.device, enable_cameras=False).app
    try:
        import gymnasium as gym
        import torch
        from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
        from omegaconf import OmegaConf
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner
        import training.tasks  # noqa: F401
        from scripts.train import _apply_task_overrides
        from training.utils.ppo_cfg import runner_kwargs

        task_id = "HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescue-AgibotA3-v0"
        task_yaml = ROOT / "cfg/task/HOPEA3TargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescue.yaml"
        algo_yaml = ROOT / "cfg/algo/ppo_v13b_complete_priors.yaml"
        task_cfg = OmegaConf.load(task_yaml)
        task_cfg.training.schedule_total_iterations = 45000
        task_cfg.motion_manifest = str(args.motion_manifest.resolve())
        algo_cfg = OmegaConf.load(algo_yaml)
        torch.manual_seed(args.seed)
        # One environment is deliberately reused for each perturbation.  A
        # fixed seed before every reset keeps the initial state/motion draw
        # identical; this avoids confusing motion-to-motion variation with
        # target-position causality.
        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        _apply_task_overrides(env_cfg, task_cfg)
        env_cfg.commands.motion.motion_manifest = str(args.motion_manifest.resolve())
        env_cfg.commands.motion.motion_file = None
        env = gym.make(task_id, cfg=env_cfg, render_mode=None)
        raw = env.unwrapped
        raw.v13b_policy_progress = 0.10
        raw.v13b_force_lower_prior_alpha = 1.0
        raw.v13b_force_upper_prior_alpha = 0.9
        command = raw.command_manager.get_term("racket_target")
        command._v13b_policy_progress = 0.10
        env = RslRlVecEnvWrapper(env)
        agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(OmegaConf.to_container(algo_cfg, resolve=True), str(task_cfg.experiment_name)))
        agent_cfg.device = args.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        policy = runner.get_inference_policy(device=raw.device)
        ids = torch.arange(1, device=raw.device)
        deltas = torch.tensor(
            [[0., 0., 0.], [0.05, 0., 0.], [-0.05, 0., 0.],
             [0., 0.05, 0.], [0., -0.05, 0.], [0., 0., 0.05],
             [0., 0., -0.05]], device=raw.device
        )
        labels = ["nominal", "+x5cm", "-x5cm", "+y5cm", "-y5cm", "+z5cm", "-z5cm"]
        results = []
        for ckpt in args.checkpoints:
            runner.load(str(ckpt))
            policy = runner.get_inference_policy(device=raw.device)
            for upper_alpha in args.upper_alphas:
                raw.v13b_force_upper_prior_alpha = float(upper_alpha)
                rows = []
                nominal_action = None
                for i, label in enumerate(labels):
                    torch.manual_seed(args.seed)
                    env.reset()
                    command = raw.command_manager.get_term("racket_target")
                    command._resample_command(ids)
                    command._compute_strike_timing()
                    base_goal = command.racket_target_pos_b().detach().clone()
                    target = base_goal + deltas[i:i+1]
                    command.set_external_target_position_b(ids, target)
                    obs = env.get_observations()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    obs = obs.to(raw.device)
                    first_action = policy(obs).detach().clone()
                    if nominal_action is None:
                        nominal_action = first_action.clone()
                    hit = torch.zeros(1, dtype=torch.bool, device=raw.device)
                    pos = torch.full((1, 3), float("nan"), device=raw.device)
                    for _ in range(args.max_steps):
                        obs, _, terminated, truncated = env.step(policy(obs))
                        if isinstance(obs, tuple):
                            obs = obs[0]
                        obs = obs.to(raw.device)
                        command = raw.command_manager.get_term("racket_target")
                        now = (command.metrics["exact_strike_hit_rate"] > 0.5) & (~hit)
                        if torch.any(now):
                            actual_b = quat_rotate_inverse(
                                yaw_quat(command.base_quat_w[now]),
                                command.racket_pos_w[now] - command.base_pos_w[now],
                            )
                            pos[now] = actual_b
                            hit |= now
                        if bool(torch.all(hit)):
                            break
                    action_delta = torch.linalg.vector_norm(first_action - nominal_action, dim=-1)
                    rows.append({
                        "label": label,
                        "delta_local_m": [float(x) for x in deltas[i].cpu()],
                        "initial_action_delta_l2": float(action_delta[0].cpu()),
                        "target_position_local_b": [float(x) for x in target[0].cpu()],
                        "actual_position_local_b": None if not torch.isfinite(pos[0]).all() else [float(x) for x in pos[0].cpu()],
                        "strike_position_error_m": None if not torch.isfinite(pos[0]).all() else float(torch.linalg.vector_norm(pos[0] - target[0]).cpu()),
                        "hit": bool(hit[0].cpu()),
                    })
                results.append({"checkpoint": str(ckpt), "upper_alpha": float(upper_alpha), "rows": rows})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"probe": "v13b_position_causality_v1", "results": results}, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "checkpoints": len(results)}, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
