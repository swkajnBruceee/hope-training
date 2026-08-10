#!/usr/bin/env python3
"""Evaluate all Rescue selection conditions for one checkpoint in one Isaac app.

This is the process-reuse counterpart to
``evaluate_v13b_precision_rescue_candidate.py``.  It creates one 128-row
PhysX scene and loads the candidate actor once, then uses clean vector-env
resets to evaluate each Native/Common and prior-ablation condition.  It is
strictly read-only: no policy update, no checkpoint write, no task mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASK_ID = "HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriors-AgibotA3-v0"
TASK_YAML = ROOT / "cfg/task/HOPEA3TargetConditionedReferenceFreeV13BCompletePriors.yaml"
ALGO_YAML = ROOT / "cfg/algo/ppo_v13b_complete_priors.yaml"
# A common comparison must remain inside the period in which the private
# motion and the public goal are explicitly aligned.  A final-global goal
# (progress >= 0.60) paired with an early checkpoint's non-zero upper prior
# asks that prior to execute two incompatible strikes at once.
DEFAULT_COMMON_PROGRESS = 0.20


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--iteration", type=int)
    p.add_argument("--historical-progress", type=float)
    p.add_argument("--source-lower-alpha", type=float)
    p.add_argument("--source-upper-alpha", type=float)
    p.add_argument("--shortlist", type=Path, default=None, help="log_shortlist.json; reuse this one Isaac app across every listed checkpoint")
    p.add_argument("--sets", default="native,common")
    p.add_argument("--conditions", default="historical,upper_off,all_off")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument(
        "--common-progress", type=float, default=DEFAULT_COMMON_PROGRESS,
        help="Canonical teacher-aligned sampler progress for the Common-set.",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--common-goal-suite", type=Path, default=None,
        help=(
            "Immutable teacher-aligned Common-set suite. It freezes both the "
            "public 10-D goal and the private motion id/start-frame state."
        ),
    )
    return p.parse_args()


def _alphas(condition: str, lower: float, upper: float) -> tuple[float, float]:
    if condition == "historical":
        return lower, upper
    if condition == "half_upper":
        return lower, upper * 0.5
    if condition == "upper_off":
        return lower, 0.0
    if condition == "all_off":
        return 0.0, 0.0
    raise ValueError(f"unknown condition {condition!r}")


def _mean(value, torch):
    good = value[torch.isfinite(value)]
    return None if good.numel() == 0 else float(good.mean().item())


def _goal_hash(value, torch) -> str:
    data = value.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    args = _args()
    test_sets = tuple(part.strip() for part in args.sets.split(",") if part.strip())
    conditions = tuple(part.strip() for part in args.conditions.split(",") if part.strip())
    if not set(test_sets) <= {"native", "common"} or not conditions:
        raise SystemExit("sets must be native/common and conditions must be nonempty")
    if args.episodes <= 0 or args.max_steps <= 0:
        raise SystemExit("invalid episode/max-step count")
    if not 0.0 <= float(args.common_progress) < 0.60:
        raise SystemExit("common-progress must be in [0, 0.60) so the teacher/public alignment is active")
    if args.shortlist is not None:
        payload = json.loads(args.shortlist.read_text(encoding="utf-8"))
        candidates = list(payload.get("shortlist", ()))
        if not candidates:
            raise SystemExit("shortlist contains no candidates")
    else:
        if args.checkpoint is None or args.iteration is None or args.historical_progress is None or args.source_lower_alpha is None or args.source_upper_alpha is None:
            raise SystemExit("single-checkpoint mode requires checkpoint/iteration/progress/lower-alpha/upper-alpha")
        candidates = [{
            "checkpoint": str(args.checkpoint), "iteration": args.iteration,
            "historical_progress": args.historical_progress,
            "historical_lower_alpha": args.source_lower_alpha,
            "historical_upper_alpha": args.source_upper_alpha,
        }]
    for candidate in candidates:
        checkpoint = Path(candidate["checkpoint"])
        if not checkpoint.is_file() or not 0.0 <= float(candidate["historical_progress"]) <= 1.0:
            raise SystemExit(f"invalid shortlist row: {candidate}")
        for key in ("historical_lower_alpha", "historical_upper_alpha"):
            if not 0.0 <= float(candidate[key]) <= 1.0:
                raise SystemExit(f"invalid shortlist alpha: {candidate}")

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=args.device, enable_cameras=False).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        from omegaconf import OmegaConf
        from rsl_rl.runners import OnPolicyRunner

        import training.tasks  # noqa: F401
        from scripts.train import _apply_task_overrides
        from training.utils.ppo_cfg import runner_kwargs
        from isaaclab.utils.math import quat_apply, yaw_quat

        task_cfg = OmegaConf.load(TASK_YAML)
        algo_cfg = OmegaConf.load(ALGO_YAML)
        torch.manual_seed(args.seed)
        env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=args.episodes)
        _apply_task_overrides(env_cfg, task_cfg)
        env_cfg.commands.motion.motion_manifest = str(Path(task_cfg.motion_manifest))
        env_cfg.commands.motion.motion_file = None
        env = RslRlVecEnvWrapper(gym.make(TASK_ID, cfg=env_cfg, render_mode=None))
        raw = env.unwrapped
        agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(
            OmegaConf.to_container(algo_cfg, resolve=True), str(task_cfg.experiment_name)
        ))
        agent_cfg.device = args.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        def load_candidate(candidate: dict):
            payload = torch.load(candidate["checkpoint"], map_location="cpu", weights_only=False)
            if bool(payload.get("v13b_migrated_from_p5u", False)):
                expected, state = runner.alg.policy.state_dict(), payload["model_state_dict"]
                missing = tuple(key for key in expected if key not in state and not key.startswith("critic."))
                unexpected = tuple(key for key in state if key not in expected)
                if missing or unexpected:
                    raise RuntimeError(f"migrated actor mismatch missing={missing}, unexpected={unexpected}")
                runner.alg.policy.load_state_dict(state, strict=False)
                runner.obs_normalizer.load_state_dict(payload["obs_norm_state_dict"])
            else:
                runner.load(str(candidate["checkpoint"]))
            return runner.get_inference_policy(device=raw.device)
        suite_path = args.common_goal_suite
        if suite_path is None:
            suite_path = args.output_dir.parent.parent / (
                f"common_set_teacher_aligned_v2_p{args.common_progress:.3f}_"
                f"seed{args.seed}_n{args.episodes}.json"
            )
        common_suite = None
        if suite_path.is_file():
            saved = json.loads(suite_path.read_text(encoding="utf-8"))
            if (
                saved.get("contract") != "v13b_teacher_aligned_common_episode_v2"
                or int(saved.get("seed", -1)) != int(args.seed)
                or int(saved.get("episodes", -1)) != int(args.episodes)
                or float(saved.get("progress", -1.0)) != float(args.common_progress)
            ):
                raise RuntimeError(f"Common-set suite contract mismatch: {suite_path}")
            common_suite = {
                "goal_local_10d": torch.as_tensor(saved["goal_local_10d"], dtype=torch.float32, device=raw.device),
                "motion_ids": torch.as_tensor(saved["motion_ids"], dtype=torch.long, device=raw.device),
                "teacher_start_frames": torch.as_tensor(saved["teacher_start_frames"], dtype=torch.long, device=raw.device),
                "teacher_hit_frames": torch.as_tensor(saved["teacher_hit_frames"], dtype=torch.long, device=raw.device),
            }
            for name, value, shape in (
                ("goal_local_10d", common_suite["goal_local_10d"], (args.episodes, 10)),
                ("motion_ids", common_suite["motion_ids"], (args.episodes,)),
                ("teacher_start_frames", common_suite["teacher_start_frames"], (args.episodes,)),
                ("teacher_hit_frames", common_suite["teacher_hit_frames"], (args.episodes,)),
            ):
                if tuple(value.shape) != shape:
                    raise RuntimeError(f"Common-set {name} has wrong shape: {tuple(value.shape)}")

        def common_episode_hash(suite) -> str:
            hasher = hashlib.sha256()
            for value in (
                suite["goal_local_10d"].to(dtype=torch.float32),
                suite["motion_ids"].to(dtype=torch.int64),
                suite["teacher_start_frames"].to(dtype=torch.int64),
                suite["teacher_hit_frames"].to(dtype=torch.int64),
            ):
                hasher.update(value.detach().to("cpu").contiguous().numpy().tobytes())
            return hasher.hexdigest()

        def apply_common_episode(command, suite) -> None:
            """Restore a public goal *and* its private teacher episode.

            The historical upper prior is a motion-conditioned teacher.  A
            goal-only Common-set would therefore make its motion and public
            target disagree.  This restores the original legal pairing before
            applying the frozen 10-D target, while keeping all public actor
            inputs reference-free.
            """
            ids = torch.arange(raw.num_envs, device=raw.device)
            goal_local = suite["goal_local_10d"]
            motion = raw.command_manager.get_term("motion")
            motion.motion_ids[:] = suite["motion_ids"]
            motion.configure_v13b_episode_strike(ids, suite["teacher_start_frames"])
            teacher_target = raw.command_manager.get_term("teacher_racket_target")
            teacher_target._resample_command(ids)
            teacher_target._compute_strike_timing()
            heading = yaw_quat(command.base_quat_w)
            command.racket_target_pos_w[:] = command.base_pos_w + quat_apply(heading, goal_local[:, :3])
            command.racket_target_vel_w[:] = quat_apply(heading, goal_local[:, 3:6])
            normal_w = quat_apply(heading, goal_local[:, 6:9])
            command.racket_target_normal_w[:] = normal_w / torch.linalg.vector_norm(
                normal_w, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            event = command.strike_event
            event.motion_id[:] = suite["motion_ids"]
            event.teacher_start_frame[:] = suite["teacher_start_frames"]
            event.teacher_hit_frame[:] = suite["teacher_hit_frames"]
            fps = float(max(int(motion.motion.fps), 1))
            event.teacher_physical_strike_time_s[:] = (
                suite["teacher_hit_frames"] - suite["teacher_start_frames"]
            ).to(dtype=goal_local.dtype) / fps
            event.episode_strike_time_s[:] = goal_local[:, 9]
            event.sampled_position_b[:] = goal_local[:, :3]
            event.sampled_velocity_b[:] = goal_local[:, 3:6]
            event.sampled_normal_b[:] = goal_local[:, 6:9]
            event.sampled_timing_offset_s[:] = goal_local[:, 9] - float(command.cfg.nominal_time_to_hit_s)
            event.strike_armed[:] = True
            event.strike_consumed[:] = False
            event.strike_event_count.zero_()
            command._episode_time_s.zero_()
            command._previous_tau[:] = goal_local[:, 9]
            command._compute_strike_timing()
            command.racket_anchor_target_pos_w[:] = command.racket_target_pos_w
            command.racket_anchor_target_vel_w[:] = command.racket_target_vel_w
            command.racket_anchor_target_normal_w[:] = command.racket_target_normal_w
            command._v13b_previous_distance[:] = torch.linalg.vector_norm(
                command.racket_pos_w - command.racket_target_pos_w, dim=-1
            ).detach()
            command._write_event_metrics(ids)

        def run_case(candidate: dict, policy, test_set: str, condition: str) -> dict:
            progress = float(candidate["historical_progress"]) if test_set == "native" else float(args.common_progress)
            lower, upper = _alphas(condition, float(candidate["historical_lower_alpha"]), float(candidate["historical_upper_alpha"]))
            raw.v13b_policy_progress = float(progress)
            raw.v13b_force_lower_prior_alpha = float(lower)
            raw.v13b_force_upper_prior_alpha = float(upper)
            command = raw.command_manager.get_term("racket_target")
            command._v13b_policy_progress = float(progress)
            # Same seed and explicit resample make all conditions for a set
            # share an identical target suite.  Common-set identity is then
            # verified across candidates by the stored hash.
            # ``torch.manual_seed`` alone is insufficient after an Isaac
            # environment has already run one reset: its reset path also uses
            # Python/NumPy/Replicator random streams.  Reset all of those via
            # the environment seed API before every case.
            raw.seed(args.seed)
            torch.manual_seed(args.seed)
            env.reset()
            ids = torch.arange(raw.num_envs, device=raw.device)
            command._resample_command(ids)
            command._compute_strike_timing()
            sampled_goal = torch.cat((
                command.racket_target_pos_b(), command.racket_target_vel_b(),
                command.racket_target_normal_b(), command.time_to_strike.unsqueeze(-1),
            ), dim=-1)
            nonlocal common_suite
            if test_set == "common":
                if common_suite is None:
                    motion = raw.command_manager.get_term("motion")
                    event = command.strike_event
                    common_suite = {
                        "goal_local_10d": sampled_goal.detach().clone(),
                        "motion_ids": motion.motion_ids.detach().clone(),
                        "teacher_start_frames": event.teacher_start_frame.detach().clone(),
                        "teacher_hit_frames": event.teacher_hit_frame.detach().clone(),
                    }
                    suite_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = suite_path.with_suffix(suite_path.suffix + ".tmp")
                    tmp.write_text(json.dumps({
                        "contract": "v13b_teacher_aligned_common_episode_v2",
                        "seed": int(args.seed), "episodes": int(args.episodes),
                        "progress": float(args.common_progress),
                        "goal_local_10d": common_suite["goal_local_10d"].detach().cpu().tolist(),
                        "motion_ids": common_suite["motion_ids"].detach().cpu().tolist(),
                        "teacher_start_frames": common_suite["teacher_start_frames"].detach().cpu().tolist(),
                        "teacher_hit_frames": common_suite["teacher_hit_frames"].detach().cpu().tolist(),
                    }, indent=2) + "\n", encoding="utf-8")
                    tmp.replace(suite_path)
                apply_common_episode(command, common_suite)
                goal = common_suite["goal_local_10d"]
            else:
                goal = sampled_goal
            goal_digest = _goal_hash(goal, torch)
            obs = env.get_observations()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = obs.to(raw.device)
            hit = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
            ended = torch.zeros_like(hit)
            timeout = torch.zeros_like(hit)
            fall = torch.zeros_like(hit)
            pos = torch.full((raw.num_envs,), float("nan"), device=raw.device)
            vel = torch.full_like(pos, float("nan"))
            normal = torch.full_like(pos, float("nan"))
            success = torch.zeros_like(hit)
            for _ in range(args.max_steps):
                actions = policy(obs)
                obs, _, terminated, truncated = env.step(actions)
                if isinstance(obs, tuple):
                    obs = obs[0]
                obs = obs.to(raw.device)
                command = raw.command_manager.get_term("racket_target")
                now = (command.metrics["exact_strike_hit_rate"] > 0.5) & (~hit) & (~ended)
                if torch.any(now):
                    pos[now] = command.metrics["racket_pos_error_exact_strike"][now]
                    vel[now] = command.metrics["racket_vel_error_exact_strike"][now]
                    normal[now] = command.metrics["racket_normal_error_deg_exact_strike"][now]
                    success[now] = (
                        (pos[now] < float(command.cfg.strike_success_pos_thresh))
                        & (vel[now] < float(command.cfg.strike_success_vel_thresh))
                        & (normal[now] < float(command.cfg.strike_success_normal_thresh_deg))
                    )
                    hit[now] = True
                done = torch.as_tensor(terminated, dtype=torch.bool, device=raw.device)
                if torch.is_tensor(truncated):
                    is_timeout = torch.as_tensor(truncated, dtype=torch.bool, device=raw.device)
                else:
                    is_timeout = torch.as_tensor(
                        truncated.get("time_outs", torch.zeros_like(done)), dtype=torch.bool, device=raw.device
                    )
                fresh = done & (~ended)
                timeout[fresh] = is_timeout[fresh]
                fall[fresh] = ~is_timeout[fresh]
                ended |= fresh
                if bool(torch.all(ended).item()):
                    break
            count = int(hit.sum().item())
            return {
                "status": "pass" if bool(torch.all(ended).item()) else "incomplete",
                "evaluation_contract": "v13b_precision_rescue_checkpoint_selection_v1",
                "checkpoint": str(Path(candidate["checkpoint"]).resolve()), "iteration": int(candidate["iteration"]),
                "set": test_set, "condition": condition, "seed": int(args.seed),
                "episodes": int(raw.num_envs), "max_steps": int(args.max_steps),
                "sampler_progress": float(progress), "prior_alphas": {"lower": lower, "upper": upper},
                "common_goal_local_sha256": goal_digest,
                "common_episode_sha256": common_episode_hash(common_suite) if test_set == "common" else None,
                "goal_contract": "[target_position_b, target_velocity_b, target_normal_b, signed_time_to_hit]",
                "metrics": {
                    "episodes_ended": int(ended.sum().item()), "exact_hit_count": count,
                    "exact_hit_rate": count / int(raw.num_envs),
                    "survival_10s": float(timeout.float().mean().item()),
                    "physical_fall_rate": float(fall.float().mean().item()),
                    "position_error_m": _mean(pos, torch), "velocity_error_mps": _mean(vel, torch),
                    "normal_error_deg": _mean(normal, torch),
                    "combined_success": float(success.float().mean().item()),
                    "combined_success_given_hit": float(success[hit].float().mean().item()) if count else None,
                },
            }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        multi = len(candidates) > 1
        for candidate in candidates:
            policy = load_candidate(candidate)
            candidate_dir = args.output_dir / f"model_{candidate['iteration']}" if multi else args.output_dir
            candidate_dir.mkdir(parents=True, exist_ok=True)
            for test_set in test_sets:
                for condition in conditions:
                    report = run_case(candidate, policy, test_set, condition)
                    path = candidate_dir / f"{test_set}_{condition}.json"
                    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    print(f"[done] {candidate['iteration']} {test_set}/{condition}: {report['metrics']}", flush=True)
                    if report["status"] != "pass":
                        raise SystemExit(f"incomplete rollout: {test_set}/{condition}")
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
