#!/usr/bin/env python3
"""First-reset continuity audit for a candidate PrecisionRescue source.

It creates the unmodified CompletePriors task and the opt-in Rescue task with
the same seed and historical progress, then compares their sampled public
10-D goal exactly.  No PPO optimizer or checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _configure(env_cfg, task_cfg, manifest: pathlib.Path) -> None:
    from scripts.train import _apply_task_overrides

    _apply_task_overrides(env_cfg, task_cfg)
    env_cfg.commands.motion.motion_manifest = str(manifest)
    env_cfg.commands.motion.motion_file = None


def _goal(command):
    import torch

    return torch.cat((
        command.racket_target_pos_w,
        command.racket_target_vel_w,
        command.racket_target_normal_w,
        command.time_to_strike.unsqueeze(-1),
    ), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-progress", required=True, type=float)
    parser.add_argument("--source-lower-alpha", required=True, type=float)
    parser.add_argument("--source-upper-alpha", required=True, type=float)
    parser.add_argument("--source-iteration", required=True, type=int)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", default="eval_outputs/v13b_complete_priors_precision_rescue/continuity/first_reset_equivalence.json")
    args = parser.parse_args()
    if not 0.0 <= args.source_progress <= 1.0:
        raise SystemExit("--source-progress must be in [0,1]")

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device, enable_cameras=False).app
    source_env = rescue_env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        from omegaconf import OmegaConf
        import training.tasks  # noqa: F401
        from training.utils.v13b_contract import lower_prior_alpha, upper_prior_alpha

        source_task = "HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriors-AgibotA3-v0"
        rescue_task = "HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescue-AgibotA3-v0"
        source_yaml = ROOT / "cfg/task/HOPEA3TargetConditionedReferenceFreeV13BCompletePriors.yaml"
        rescue_yaml = ROOT / "cfg/task/HOPEA3TargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescue.yaml"
        source_task_cfg = OmegaConf.load(source_yaml)
        rescue_task_cfg = OmegaConf.load(rescue_yaml)
        rescue_task_cfg.training.source_checkpoint = str(args.source_checkpoint)
        rescue_task_cfg.training.source_iteration = int(args.source_iteration)
        rescue_task_cfg.training.source_historical_progress = float(args.source_progress)
        rescue_task_cfg.training.source_lower_alpha = float(args.source_lower_alpha)
        rescue_task_cfg.training.source_upper_alpha = float(args.source_upper_alpha)
        manifest = pathlib.Path(str(source_task_cfg.motion_manifest)).expanduser()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)

        torch.manual_seed(args.seed)
        source_cfg = parse_env_cfg(source_task, device=args.device, num_envs=args.num_envs)
        _configure(source_cfg, source_task_cfg, manifest)
        source_env = gym.make(source_task, cfg=source_cfg, render_mode=None)
        source_raw = source_env.unwrapped
        source_raw.v13b_policy_progress = float(args.source_progress)
        source_cmd = source_raw.command_manager.get_term("racket_target")
        source_cmd._v13b_policy_progress = float(args.source_progress)
        torch.manual_seed(args.seed)
        source_env.reset()
        ids = torch.arange(args.num_envs, device=source_raw.device)
        source_cmd._resample_command(ids)
        source_cmd._compute_strike_timing()
        source_goal = _goal(source_cmd).detach().cpu()
        source_action = source_raw.action_manager.get_term("joint_pos")
        source_runtime_progress = float(source_raw.v13b_policy_progress)
        source_lower_runtime = float(source_action._scheduled_prior_alpha("lower", args.source_progress))
        source_upper_runtime = float(source_action._scheduled_prior_alpha("upper", args.source_progress))
        # Isaac scenes use the same /World/envs prim paths.  Close the source
        # environment before constructing the independent Rescue scene; the
        # captured tensors/scalars above are sufficient for an exact compare.
        source_env.close()
        source_env = None

        torch.manual_seed(args.seed)
        rescue_cfg = parse_env_cfg(rescue_task, device=args.device, num_envs=args.num_envs)
        _configure(rescue_cfg, rescue_task_cfg, manifest)
        rescue_env = gym.make(rescue_task, cfg=rescue_cfg, render_mode=None)
        rescue_raw = rescue_env.unwrapped
        rescue_cmd = rescue_raw.command_manager.get_term("racket_target")
        torch.manual_seed(args.seed)
        rescue_env.reset()
        rescue_cmd._resample_command(torch.arange(args.num_envs, device=rescue_raw.device))
        rescue_cmd._compute_strike_timing()
        rescue_goal = _goal(rescue_cmd).detach().cpu()
        rescue_action = rescue_raw.action_manager.get_term("joint_pos")

        expected_lower = lower_prior_alpha(args.source_progress)
        expected_upper = upper_prior_alpha(args.source_progress)
        goal_abs_max = float(torch.max(torch.abs(source_goal - rescue_goal)).item())
        report = {
            "status": "pass",
            "source": {
                "checkpoint": args.source_checkpoint,
                "iteration": args.source_iteration,
                "progress": args.source_progress,
                "lower_alpha": args.source_lower_alpha,
                "upper_alpha": args.source_upper_alpha,
            },
            "seed": args.seed,
            "num_envs": args.num_envs,
            "actor_obs_dim": 98,
            "action_dim": 26,
            "goal_abs_max_difference": goal_abs_max,
            "source_first_goal_10d": source_goal.tolist(),
            "rescue_first_goal_10d": rescue_goal.tolist(),
            "source_runtime_progress": source_runtime_progress,
            "rescue_runtime_progress": float(rescue_raw.v13b_policy_progress),
            "source_lower_alpha_runtime": source_lower_runtime,
            "rescue_lower_alpha_runtime": float(rescue_action._scheduled_prior_alpha("lower", args.source_progress)),
            "source_upper_alpha_runtime": source_upper_runtime,
            "rescue_upper_alpha_runtime": float(rescue_action._scheduled_prior_alpha("upper", args.source_progress)),
            "historical_schedule_expected": {"lower": expected_lower, "upper": expected_upper},
        }
        report["pass"] = bool(
            goal_abs_max <= 1.0e-6
            and abs(report["rescue_runtime_progress"] - args.source_progress) <= 1.0e-9
            and abs(report["rescue_lower_alpha_runtime"] - args.source_lower_alpha) <= 1.0e-6
            and abs(report["rescue_upper_alpha_runtime"] - args.source_upper_alpha) <= 1.0e-6
        )
        report["status"] = "pass" if report["pass"] else "fail"
        _write(pathlib.Path(args.output), report)
        print(json.dumps(report, indent=2), flush=True)
        if not report["pass"]:
            raise SystemExit("PrecisionRescue first-reset continuity audit failed")
    finally:
        if source_env is not None:
            source_env.close()
        if rescue_env is not None:
            rescue_env.close()
        app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Even a Kit/environment-creation failure must leave a durable audit
        # artifact.  This prevents a silent initialization stall from being
        # mistaken for a passed continuity test.
        output = "eval_outputs/v13b_complete_priors_precision_rescue/continuity/first_reset_equivalence.json"
        for index, token in enumerate(sys.argv[:-1]):
            if token == "--output":
                output = sys.argv[index + 1]
                break
        _write(pathlib.Path(output), {
            "status": "error",
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
