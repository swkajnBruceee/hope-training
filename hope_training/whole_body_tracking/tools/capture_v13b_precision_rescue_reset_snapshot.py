#!/usr/bin/env python3
"""Capture one isolated V1.3B reset snapshot for continuity comparison."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("completepriors", "rescue"), required=True)
    parser.add_argument("--source-progress", required=True, type=float)
    parser.add_argument("--source-lower-alpha", required=True, type=float)
    parser.add_argument("--source-upper-alpha", required=True, type=float)
    parser.add_argument("--source-iteration", required=True, type=int)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device, enable_cameras=False).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        from omegaconf import OmegaConf
        import training.tasks  # noqa: F401
        from scripts.train import _apply_task_overrides

        is_rescue = args.variant == "rescue"
        task = ("HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescue-AgibotA3-v0"
                if is_rescue else "HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriors-AgibotA3-v0")
        yaml_path = ROOT / ("cfg/task/HOPEA3TargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescue.yaml"
                            if is_rescue else "cfg/task/HOPEA3TargetConditionedReferenceFreeV13BCompletePriors.yaml")
        task_cfg = OmegaConf.load(yaml_path)
        if is_rescue:
            task_cfg.training.source_checkpoint = str(args.source_checkpoint)
            task_cfg.training.source_iteration = int(args.source_iteration)
            task_cfg.training.source_historical_progress = float(args.source_progress)
            task_cfg.training.source_lower_alpha = float(args.source_lower_alpha)
            task_cfg.training.source_upper_alpha = float(args.source_upper_alpha)
        torch.manual_seed(args.seed)
        env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
        _apply_task_overrides(env_cfg, task_cfg)
        manifest = pathlib.Path(str(task_cfg.motion_manifest)).expanduser()
        env_cfg.commands.motion.motion_manifest = str(manifest)
        env_cfg.commands.motion.motion_file = None
        env = gym.make(task, cfg=env_cfg, render_mode=None)
        raw = env.unwrapped
        raw.v13b_policy_progress = float(args.source_progress)
        cmd = raw.command_manager.get_term("racket_target")
        cmd._v13b_policy_progress = float(args.source_progress)
        torch.manual_seed(args.seed)
        env.reset()
        cmd._resample_command(torch.arange(args.num_envs, device=raw.device))
        cmd._compute_strike_timing()
        term = raw.action_manager.get_term("joint_pos")
        goal = torch.cat((cmd.racket_target_pos_w, cmd.racket_target_vel_w, cmd.racket_target_normal_w, cmd.time_to_strike.unsqueeze(-1)), dim=-1)
        snapshot = {
            "status": "pass", "variant": args.variant, "seed": args.seed, "num_envs": args.num_envs,
            "source_progress": args.source_progress,
            "goal_10d": goal.detach().cpu().tolist(),
            "runtime_progress": float(raw.v13b_policy_progress),
            "lower_alpha": float(term._scheduled_prior_alpha("lower", args.source_progress)),
            "upper_alpha": float(term._scheduled_prior_alpha("upper", args.source_progress)),
            "actor_obs_dim": 98, "action_dim": 26,
        }
        path = pathlib.Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(snapshot, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
