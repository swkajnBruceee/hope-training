#!/usr/bin/env python3
"""One-env-per-case V1.3B runtime contract smoke.

This is deliberately not a learning run.  It checks that the runtime loads the
direct-action envelope, accepts small target perturbations, preserves signed
time-to-hit, and survives a fixed 10 s zero-action horizon.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import torch
from isaaclab.app import AppLauncher

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK_ID = "HOPE-FloatingTargetConditionedReferenceFreeV13B-AgibotA3-v0"
STEPS = 500  # 10 s at the 50 Hz environment step


def main() -> None:
    app = AppLauncher(headless=True, device="cuda:0", enable_cameras=False).app
    try:
        import gymnasium as gym
        import training.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        cases = ["nominal_zero", "position_plus_y", "position_minus_y", "normal_plus", "speed_plus", "timing_after_hit"]
        cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=len(cases))
        cfg.scene.num_envs = len(cases)
        env = gym.make(TASK_ID, cfg=cfg, render_mode=None)
        try:
            env.reset()
            raw = env.unwrapped
            command = raw.command_manager.get_term("racket_target")
            action_term = raw.action_manager.get_term("joint_pos")
            device = raw.device

            # Exercise the runner-facing curriculum hook before making six
            # explicit, paired perturbations of the sampled nominal goal.
            raw.v13b_policy_progress = 0.50
            command._v13b_policy_progress = 0.50
            command._resample_command(torch.arange(len(cases), device=device))
            progress = float(command.metrics["v13b_curriculum_progress"].mean().detach().cpu())

            # Keep the same base-relative target semantics while making six
            # explicit, paired perturbations of the sampled nominal goal.
            p0 = command.racket_target_pos_w.clone()
            n0 = command.racket_target_normal_w.clone()
            v0 = command.racket_target_vel_w.clone()
            t0 = command._hit_time.clone()
            command.racket_target_pos_w[1, 1] += 0.01
            command.racket_target_pos_w[2, 1] -= 0.01
            # A small tangent normal perturbation, followed by renormalization.
            command.racket_target_normal_w[3, 2] += 0.02
            command.racket_target_normal_w[3] /= torch.linalg.vector_norm(command.racket_target_normal_w[3]).clamp_min(1.0e-6)
            command.racket_target_vel_w[4] *= 1.05
            command._hit_time[5] = -0.10
            command._compute_strike_timing()
            target_deltas = command.racket_target_pos_w - p0
            signed_tau_initial = command.time_to_strike.detach().cpu().tolist()

            zero = torch.zeros((len(cases), 26), device=device)
            finite = True
            first_done = None
            first_terminated = False
            max_tilt = 0.0
            min_height = float("inf")
            for step in range(STEPS):
                obs, _, terminated, truncated, _ = env.step(zero)
                done = terminated | truncated
                if bool(done.any().detach().cpu()) and first_done is None:
                    ids = torch.nonzero(done, as_tuple=False).flatten().detach().cpu().tolist()
                    first_done = {
                        "step": step + 1,
                        "env_ids": ids,
                        "terminated": bool(terminated.any().detach().cpu()),
                        "truncated": bool(truncated.any().detach().cpu()),
                    }
                    first_terminated = bool(terminated.any().detach().cpu())
                robot = raw.scene["robot"]
                finite = finite and bool(torch.isfinite(robot.data.joint_pos).all().detach().cpu())
                g = robot.data.projected_gravity_b
                tilt = torch.arccos(torch.clamp(-g[:, 2], -1.0, 1.0))
                max_tilt = max(max_tilt, float(tilt.max().detach().cpu()))
                min_height = min(min_height, float(robot.data.root_pos_w[:, 2].min().detach().cpu()))

            signed_tau = command.time_to_strike.detach().cpu().tolist()
            results = {
                "status": "v13b_runtime_smoke_complete",
                "task": TASK_ID,
                "cases": cases,
                "steps": STEPS,
                "runtime_direct_scale_status": getattr(raw, "v13b_direct_scale_status", None),
                "loaded_lower_scale_rad": [float(x) for x in action_term._lower_scale_direct[0].detach().cpu()],
                "loaded_upper_scale_rad": [float(x) for x in action_term._upper_scale_direct[0].detach().cpu()],
                "curriculum_progress_after_hook": progress,
                "position_delta_m": target_deltas.detach().cpu().tolist(),
                "signed_time_to_hit_s_initial": signed_tau_initial,
                "signed_time_to_hit_s_after_override": signed_tau,
                "finite": finite,
                "first_done": first_done,
                "max_tilt_deg": math.degrees(max_tilt),
                "min_root_height_m": min_height,
                "pass": bool(finite and not first_terminated and progress > 0.0 and signed_tau_initial[5] < 0.0),
            }
            out_dir = ROOT / "eval_outputs" / "target_conditioned_v13b_runtime_smoke"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(results, indent=2), flush=True)
        finally:
            env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
