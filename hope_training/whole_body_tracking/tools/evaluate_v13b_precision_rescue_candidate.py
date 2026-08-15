#!/usr/bin/env python3
"""Read-only deterministic V1.3B checkpoint evaluation for Rescue selection.

This is deliberately an *evaluation* program: it never calls PPO update and
does not change either the active CompletePriors run or the Rescue task.

The important distinction is explicit in the CLI:

* native: evaluate at a checkpoint's historical training progress;
* common: evaluate every checkpoint at one fixed final-local progress.

For Common-set evaluation the curriculum clock is set to ``--progress`` but
the lower/upper prior amplitudes are independently forced to the candidate's
historical values.  This separates target difficulty from teacher reliance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALGO_YAML = ROOT / "cfg/algo/ppo_v13b_complete_priors.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task-mode", choices=("completepriors", "precision_rescue"), default="completepriors")
    parser.add_argument("--rescue-schedule-total-updates", type=int, default=None)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--set", choices=("native", "common"), required=True)
    parser.add_argument(
        "--condition",
        choices=("historical", "half_upper", "lower_off", "upper_off", "all_off"),
        required=True,
    )
    parser.add_argument(
        "--lower-alpha-override",
        type=float,
        default=None,
        help="Optional evaluation-only lower-prior alpha override.",
    )
    parser.add_argument(
        "--upper-alpha-override",
        type=float,
        default=None,
        help="Optional evaluation-only upper-prior alpha override.",
    )
    parser.add_argument(
        "--progress", type=float, required=True,
        help="Sampler/curriculum progress for this test set.",
    )
    parser.add_argument("--source-lower-alpha", required=True, type=float)
    parser.add_argument("--source-upper-alpha", required=True, type=float)
    parser.add_argument(
        "--lower-prior-checkpoint",
        type=Path,
        default=None,
        help="Optional frozen model_3396-compatible checkpoint used for this evaluation.",
    )
    parser.add_argument(
        "--upper-prior-checkpoint",
        type=Path,
        default=None,
        help="Optional frozen model_900-compatible checkpoint used for this evaluation.",
    )
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _condition_alphas(args: argparse.Namespace) -> tuple[float, float]:
    lower = float(args.source_lower_alpha)
    upper = float(args.source_upper_alpha)
    if args.condition == "half_upper":
        upper *= 0.5
    elif args.condition == "lower_off":
        lower = 0.0
    elif args.condition == "upper_off":
        upper = 0.0
    elif args.condition == "all_off":
        lower, upper = 0.0, 0.0
    return lower, upper


def _finite_mean(values, torch) -> float | None:
    valid = values[torch.isfinite(values)]
    return None if valid.numel() == 0 else float(valid.mean().item())


def _group_metrics(values, face_sign, torch) -> dict[str, float | int | None]:
    """Return hit metrics separately for the active red and black faces."""
    result: dict[str, float | int | None] = {}
    for label, mask in (
        ("forehand_red", face_sign > 0.0),
        ("backhand_black", face_sign < 0.0),
    ):
        valid = mask & torch.isfinite(values)
        result[label] = {
            "count": int(valid.sum().item()),
            "mean": _finite_mean(values[valid], torch) if torch.any(valid) else None,
        }
    return result


def _digest_goal(goal, torch) -> str:
    # Values are stored in robot local coordinates, so scene-tile origins do
    # not leak into the Common-set identity.
    data = goal.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    args = _parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise SystemExit("episodes and max-steps must be positive")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    if not 0.0 <= args.progress <= 1.0:
        raise SystemExit("progress must lie in [0, 1]")
    for name in ("source_lower_alpha", "source_upper_alpha"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must lie in [0, 1]")
    if args.task_mode == "precision_rescue" and (
        args.rescue_schedule_total_updates is None or args.rescue_schedule_total_updates < 2
    ):
        raise SystemExit("precision_rescue requires --rescue-schedule-total-updates >= 2")
    if args.task_mode == "precision_rescue":
        task_id = "HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescue-AgibotA3-v0"
        task_yaml = ROOT / "cfg/task/HOPEA3TargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescue.yaml"
    else:
        task_id = "HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriors-AgibotA3-v0"
        task_yaml = ROOT / "cfg/task/HOPEA3TargetConditionedReferenceFreeV13BCompletePriors.yaml"

    # AppLauncher must precede imports which transitively create Isaac/Omni
    # objects.  This is also why this tool is a standalone process.
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device, enable_cameras=False).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from omegaconf import OmegaConf
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner

        import training.tasks  # noqa: F401 -- register the task before gym.make
        from scripts.train import _apply_task_overrides
        from training.utils.ppo_cfg import runner_kwargs

        task_cfg = OmegaConf.load(task_yaml)
        if args.task_mode == "precision_rescue":
            task_cfg.training.schedule_total_iterations = int(args.rescue_schedule_total_updates)
        algo_cfg = OmegaConf.load(ALGO_YAML)
        torch.manual_seed(args.seed)
        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.episodes)
        _apply_task_overrides(env_cfg, task_cfg)
        # Keep the public student checkpoint separate from the two private
        # teacher checkpoints.  This makes teacher/student matrix evaluations
        # explicit and prevents a path change from silently changing the
        # default frozen-prior contract.
        if args.lower_prior_checkpoint is not None:
            if not args.lower_prior_checkpoint.is_file():
                raise SystemExit(f"lower prior checkpoint does not exist: {args.lower_prior_checkpoint}")
            env_cfg.actions.joint_pos.annealed_3396_prior_checkpoint = str(
                args.lower_prior_checkpoint.resolve()
            )
        if args.upper_prior_checkpoint is not None:
            if not args.upper_prior_checkpoint.is_file():
                raise SystemExit(f"upper prior checkpoint does not exist: {args.upper_prior_checkpoint}")
            env_cfg.actions.joint_pos.annealed_900_upper_prior_checkpoint = str(
                args.upper_prior_checkpoint.resolve()
            )
        env_cfg.commands.motion.motion_manifest = str(Path(task_cfg.motion_manifest))
        env_cfg.commands.motion.motion_file = None
        env = gym.make(task_id, cfg=env_cfg, render_mode=None)
        raw = env.unwrapped
        raw.v13b_policy_progress = float(args.progress)
        lower_alpha, upper_alpha = _condition_alphas(args)
        if args.lower_alpha_override is not None:
            lower_alpha = float(args.lower_alpha_override)
        if args.upper_alpha_override is not None:
            upper_alpha = float(args.upper_alpha_override)
        for name, value in (("lower", lower_alpha), ("upper", upper_alpha)):
            if not 0.0 <= value <= 1.0:
                raise SystemExit(f"{name} alpha override must lie in [0, 1]")
        # These are documented replay-only latches in actions.py.  They are
        # required on Common-set so all candidates share goals but retain
        # their own historical teacher amplitudes.
        raw.v13b_force_lower_prior_alpha = lower_alpha
        raw.v13b_force_upper_prior_alpha = upper_alpha
        command = raw.command_manager.get_term("racket_target")
        command._v13b_policy_progress = float(args.progress)

        agent_cfg = RslRlOnPolicyRunnerCfg(
            **runner_kwargs(
                OmegaConf.to_container(algo_cfg, resolve=True),
                str(task_cfg.experiment_name),
            )
        )
        agent_cfg.device = args.device
        env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if bool(checkpoint_payload.get("v13b_migrated_from_p5u", False)):
            expected = runner.alg.policy.state_dict()
            state = checkpoint_payload["model_state_dict"]
            unexpected = tuple(key for key in state if key not in expected)
            missing = tuple(key for key in expected if key not in state and not key.startswith("critic."))
            if unexpected or missing:
                raise RuntimeError(
                    f"migrated actor contract mismatch: missing={missing}, unexpected={unexpected}"
                )
            runner.alg.policy.load_state_dict(state, strict=False)
            runner.obs_normalizer.load_state_dict(checkpoint_payload["obs_norm_state_dict"])
        else:
            runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=raw.device)

        # Each row is one episode. Isaac auto-resets individual rows after a
        # terminal event, so all per-row results must be latched before the
        # reset can expose a second goal.
        recorded_hit = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
        ended = torch.zeros_like(recorded_hit)
        timed_out = torch.zeros_like(recorded_hit)
        physical_fall = torch.zeros_like(recorded_hit)
        position = torch.full((raw.num_envs,), float("nan"), device=raw.device)
        velocity = torch.full_like(position, float("nan"))
        normal = torch.full_like(position, float("nan"))
        hit_face_sign = torch.zeros_like(position)
        exact_success = torch.zeros_like(recorded_hit)

        # A process-local seed plus the explicit post-reset resample gives a
        # stable N-row goal suite.  This must happen *after* the wrapper's
        # reset: otherwise its own reset would silently replace the hashed
        # Common-set goals before the first policy action.
        torch.manual_seed(args.seed)
        env.reset()
        command = raw.command_manager.get_term("racket_target")
        ids = torch.arange(raw.num_envs, device=raw.device)
        command._resample_command(ids)
        command._compute_strike_timing()
        goal_local = torch.cat(
            (
                command.racket_target_pos_b(),
                command.racket_target_vel_b(),
                command.racket_target_normal_b(),
                command.time_to_strike.unsqueeze(-1),
            ),
            dim=-1,
        )
        goal_digest = _digest_goal(goal_local, torch)
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = obs.to(raw.device)
        for step in range(args.max_steps):
            actions = policy(obs)
            obs, _, terminated, truncated = env.step(actions)
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = obs.to(raw.device)
            command = raw.command_manager.get_term("racket_target")
            hit = (command.metrics["exact_strike_hit_rate"] > 0.5) & (~recorded_hit) & (~ended)
            if torch.any(hit):
                position[hit] = command.metrics["racket_pos_error_exact_strike"][hit]
                velocity[hit] = command.metrics["racket_vel_error_exact_strike"][hit]
                normal[hit] = command.metrics["racket_normal_error_deg_exact_strike"][hit]
                hit_face_sign[hit] = command.face_sign[hit]
                exact_success[hit] = (
                    (position[hit] < float(command.cfg.strike_success_pos_thresh))
                    & (velocity[hit] < float(command.cfg.strike_success_vel_thresh))
                    & (normal[hit] < float(command.cfg.strike_success_normal_thresh_deg))
                )
                recorded_hit[hit] = True
            done = torch.as_tensor(terminated, device=raw.device, dtype=torch.bool)
            if torch.is_tensor(truncated):
                timeout = torch.as_tensor(truncated, device=raw.device, dtype=torch.bool)
            else:
                timeout = torch.as_tensor(
                    truncated.get("time_outs", torch.zeros_like(done)),
                    device=raw.device,
                    dtype=torch.bool,
                )
            fresh_done = done & (~ended)
            timed_out[fresh_done] = timeout[fresh_done]
            physical_fall[fresh_done] = ~timeout[fresh_done]
            ended |= fresh_done
            if bool(torch.all(ended).item()):
                break

        hit_count = int(recorded_hit.sum().item())
        episode_count = int(raw.num_envs)
        report = {
            "status": "pass" if bool(torch.all(ended).item()) else "incomplete",
            "evaluation_contract": "v13b_precision_rescue_checkpoint_selection_v1",
            "task_mode": args.task_mode,
            "checkpoint": str(args.checkpoint.resolve()),
            "iteration": int(args.iteration),
            "set": args.set,
            "condition": args.condition,
            "seed": int(args.seed),
            "episodes": episode_count,
            "max_steps": int(args.max_steps),
            "sampler_progress": float(args.progress),
            "prior_alphas": {"lower": lower_alpha, "upper": upper_alpha},
            "teacher_checkpoints": {
                "lower": str(
                    (args.lower_prior_checkpoint or Path(env_cfg.actions.joint_pos.annealed_3396_prior_checkpoint)).resolve()
                ),
                "upper": str(
                    (args.upper_prior_checkpoint or Path(env_cfg.actions.joint_pos.annealed_900_upper_prior_checkpoint)).resolve()
                ),
            },
            "common_goal_local_sha256": goal_digest,
            "goal_contract": "[target_position_b, target_velocity_b, target_normal_b, signed_time_to_hit]",
            "metrics": {
                "episodes_ended": int(ended.sum().item()),
                "exact_hit_count": hit_count,
                "exact_hit_rate": hit_count / episode_count,
                "survival_10s": float(timed_out.float().mean().item()),
                "physical_fall_rate": float(physical_fall.float().mean().item()),
                "position_error_m": _finite_mean(position, torch),
                "velocity_error_mps": _finite_mean(velocity, torch),
                "normal_error_deg": _finite_mean(normal, torch),
                "combined_success": float(exact_success.float().mean().item()),
                "combined_success_given_hit": (
                    float(exact_success[recorded_hit].float().mean().item()) if hit_count else None
                ),
                "by_active_face": {
                    "position_error_m": _group_metrics(position, hit_face_sign, torch),
                    "velocity_error_mps": _group_metrics(velocity, hit_face_sign, torch),
                    "normal_error_deg": _group_metrics(normal, hit_face_sign, torch),
                    "combined_success": {
                        "forehand_red": float(
                            exact_success[(hit_face_sign > 0.0) & recorded_hit].float().mean().item()
                        )
                        if torch.any((hit_face_sign > 0.0) & recorded_hit)
                        else None,
                        "backhand_black": float(
                            exact_success[(hit_face_sign < 0.0) & recorded_hit].float().mean().item()
                        )
                        if torch.any((hit_face_sign < 0.0) & recorded_hit)
                        else None,
                    },
                },
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        if report["status"] != "pass":
            raise SystemExit("evaluation did not finish one 10-second episode per row")
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
