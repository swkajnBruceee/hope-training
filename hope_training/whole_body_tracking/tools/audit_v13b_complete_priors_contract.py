#!/usr/bin/env python3
"""Deterministic contract audit for V1.3B CompletePriors.

Run this before PPO preflight.  It proves the private teacher has been
rephased to the public short time-to-hit and records the one-shot event
timeline without exposing any of that private metadata to the actor.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK = "HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriors-AgibotA3-v0"
TASK_CFG = ROOT / "cfg/task/HOPEA3TargetConditionedReferenceFreeV13BCompletePriors.yaml"
DEFAULT_OUT = ROOT / "eval_outputs/v13b_complete_priors_contract"


class AuditTimeout(RuntimeError):
    """Raised when Isaac startup/environment creation exceeds the audit budget."""


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise AuditTimeout("Isaac audit phase timed out")


def tensor_list(value: torch.Tensor) -> list[float]:
    return [float(x) for x in value.detach().cpu().flatten()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1, choices=(1, 100))
    parser.add_argument("--rollout-steps", type=int, default=500)
    parser.add_argument("--progress", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--startup-timeout-s", type=int, default=180)
    parser.add_argument("--env-timeout-s", type=int, default=180)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    out = pathlib.Path(args.output_dir).expanduser().resolve()
    name = "alignment_100.json" if args.episodes == 100 else "one_strike_10s.json"
    result_path = out / name
    status_path = out / (name.replace(".json", ".status.json"))
    started = time.time()
    phase = "preflight"

    def status(state: str, **extra) -> None:
        _write_json(
            status_path,
            {
                "task": TASK,
                "episodes": args.episodes,
                "state": state,
                "phase": phase,
                "elapsed_s": round(time.time() - started, 3),
                "pid": os.getpid(),
                **extra,
            },
        )
        print(f"[v13b-audit] {state} phase={phase}", flush=True)

    # Write before importing/starting Isaac.  If Kit or an asset importer hangs,
    # the caller still gets a durable diagnostic instead of an empty directory.
    status("started", result_path=str(result_path))
    signal.signal(signal.SIGALRM, _alarm_handler)
    app = None
    try:
        phase = "import_isaac"
        status("launching_isaac")
        # Isaac/Kit must not see this audit script's private arguments.
        sys.argv = [sys.argv[0]]
        from isaaclab.app import AppLauncher

        signal.alarm(max(1, int(args.startup_timeout_s)))
        app = AppLauncher(headless=args.headless, device=args.device, enable_cameras=False).app
        signal.alarm(0)
        phase = "isaac_ready"
        status("isaac_ready")

        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        from omegaconf import OmegaConf
        import training.tasks  # noqa: F401
        from scripts.train import _apply_task_overrides

        phase = "create_env"
        status("creating_env")
        task_cfg = OmegaConf.load(TASK_CFG)
        env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.episodes)
        _apply_task_overrides(env_cfg, task_cfg)
        # The standalone audit does not pass through Hydra's top-level
        # ``motion_manifest=...`` override, so wire the manifest explicitly
        # before gym.make.  Leaving this None makes Isaac fail deep inside the
        # MotionCommand constructor with an opaque ``stat(None)`` error.
        manifest = task_cfg.get("motion_manifest")
        if manifest is None:
            raise RuntimeError(f"CompletePriors audit task config has no motion_manifest: {TASK_CFG}")
        manifest_path = pathlib.Path(str(manifest)).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = (ROOT / manifest_path).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"CompletePriors audit motion_manifest does not exist: {manifest_path}")
        env_cfg.commands.motion.motion_manifest = str(manifest_path)
        env_cfg.commands.motion.motion_file = None
        signal.alarm(max(1, int(args.env_timeout_s)))
        env = gym.make(TASK, cfg=env_cfg, render_mode=None)
        signal.alarm(0)
        try:
            raw = env.unwrapped
            raw.v13b_policy_progress = float(args.progress)
            raw.v13b_private_motion_disabled = False
            phase = "reset"
            status("resetting_env")
            signal.alarm(max(1, int(args.env_timeout_s)))
            env.reset()
            signal.alarm(0)
            phase = "audit"
            status("auditing")
            command = raw.command_manager.get_term("racket_target")
            event = command.strike_event
            dt = float(raw.step_dt)
            result = {
                "task": TASK,
                "episodes": args.episodes,
                "progress": args.progress,
                "control_dt_s": dt,
                "public_strike_time_s": tensor_list(event.episode_strike_time_s),
                "teacher_physical_strike_time_s": tensor_list(event.teacher_physical_strike_time_s),
                "teacher_start_frame": [int(x) for x in event.teacher_start_frame.detach().cpu()],
                "teacher_hit_frame": [int(x) for x in event.teacher_hit_frame.detach().cpu()],
                "motion_id": [int(x) for x in event.motion_id.detach().cpu()],
                "goal_teacher_position_error_m": tensor_list(command.metrics["v13b_goal_teacher_position_error_m"]),
                "goal_teacher_velocity_error_mps": tensor_list(command.metrics["v13b_goal_teacher_velocity_error_mps"]),
                "goal_teacher_normal_error_deg": tensor_list(command.metrics["v13b_goal_teacher_normal_error_deg"]),
                "teacher_public_time_error_s": tensor_list(command.metrics["v13b_teacher_public_time_error_s"]),
                "goal_sample_count": [int(x) for x in event.goal_sample_count.detach().cpu()],
                "goal_resample_count_after_reset": [int(x) for x in event.goal_resample_count_after_reset.detach().cpu()],
            }
            initial_pass = bool(
                torch.all((event.episode_strike_time_s >= 0.20) & (event.episode_strike_time_s <= 0.60))
                and torch.all(event.goal_sample_count == 1)
                and torch.all(event.goal_resample_count_after_reset == 0)
                and torch.all(event.motion_id >= 0)
                and torch.all(torch.abs(event.episode_strike_time_s - event.teacher_physical_strike_time_s) <= dt + 1.0e-6)
            )

            # The 100-episode audit proves reset-time geometric/time alignment.
            # The one-environment audit additionally verifies one crossing and
            # no second event for a full 10 second horizon unless strict fall
            # ends that particular episode.
            timeline = []
            first_done = None
            if args.episodes == 1:
                zero = torch.zeros((1, 26), device=raw.device)
                for step in range(args.rollout_steps):
                    _obs, _rew, terminated, truncated, _info = env.step(zero)
                    e = command.strike_event
                    timeline.append({
                        "step": step + 1,
                        "tau_s": float(command.time_to_strike[0].detach().cpu()),
                        "teacher_frame": int(command.metrics["v13b_teacher_frame"][0].detach().cpu()),
                        "strike_trigger": int(command.metrics["v13b_strike_reward_trigger"][0].detach().cpu()),
                        "strike_event_count": int(e.strike_event_count[0].detach().cpu()),
                        "post_hit_phase": int(command.metrics["v13b_post_hit_phase"][0].detach().cpu()),
                    })
                    if bool((terminated | truncated)[0].detach().cpu()):
                        first_done = {"step": step + 1, "terminated": bool(terminated[0]), "truncated": bool(truncated[0])}
                        break
                e = command.strike_event
                result["first_done"] = first_done
                result["completed_steps"] = len(timeline)
                result["goal_sample_count_final"] = int(e.goal_sample_count[0].detach().cpu())
                result["goal_resample_count_final"] = int(e.goal_resample_count_after_reset[0].detach().cpu())
                result["strike_event_count_final"] = int(e.strike_event_count[0].detach().cpu())
                result["upper_prior_wrap_count_final"] = int(e.upper_prior_wrap_count[0].detach().cpu())
                result["timeline"] = timeline
                initial_pass = initial_pass and result["goal_sample_count_final"] == 1 and result["goal_resample_count_final"] == 0 and result["strike_event_count_final"] <= 1 and result["upper_prior_wrap_count_final"] == 0

            result["pass"] = initial_pass
            _write_json(result_path, result)
            _write_json(status_path, {"task": TASK, "episodes": args.episodes, "state": "pass" if initial_pass else "fail", "phase": "complete", "elapsed_s": round(time.time() - started, 3), "pid": os.getpid(), "result_path": str(result_path)})
            print(json.dumps(result, indent=2), flush=True)
            if not initial_pass:
                raise SystemExit("V1.3B CompletePriors contract audit failed")
        finally:
            env.close()
    except BaseException as exc:
        failure = {
            "task": TASK,
            "episodes": args.episodes,
            "pass": False,
            "state": "timeout" if isinstance(exc, AuditTimeout) else "error",
            "phase": phase,
            "elapsed_s": round(time.time() - started, 3),
            "pid": os.getpid(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(result_path, failure)
        _write_json(status_path, failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
    finally:
        signal.alarm(0)
        if app is not None:
            try:
                app.close()
            except Exception as exc:  # pragma: no cover - best effort during startup failure
                print(f"[v13b-audit] app close warning: {exc}", flush=True)


if __name__ == "__main__":
    main()
