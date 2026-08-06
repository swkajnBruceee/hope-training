#!/usr/bin/env python3
"""Audit the zero-residual passive baseline for Recovery-A."""

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
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settled-tilt-rad", type=float, default=0.01)
parser.add_argument("--settled-angular-velocity-rad-s", type=float, default=0.05)
parser.add_argument("--settled-consecutive-steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if min(args_cli.num_envs, args_cli.steps, args_cli.settled_consecutive_steps) < 1:
    parser.error("counts must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

import training.tasks.base_locomotion.config.a3  # noqa: F401


def _tilt(projected_gravity: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.clamp(-projected_gravity[:, 2], min=-1.0, max=1.0))


def _group_stats(values: torch.Tensor, mask: torch.Tensor) -> dict:
    selected = values[mask]
    if selected.numel() == 0:
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {
        "count": int(selected.numel()),
        "mean": float(selected.float().mean().item()),
        "median": float(selected.float().median().item()),
        "max": float(selected.max().item()),
    }


def main() -> int:
    env = None
    try:
        cfg = gym.spec("A3BaseStandRecoveryA-v0").kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = args_cli.num_envs
        cfg.seed = args_cli.seed
        cfg.sim.device = args_cli.device
        env = gym.make("A3BaseStandRecoveryA-v0", cfg=cfg)
        env.reset(seed=args_cli.seed)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        disturbed = unwrapped.recovery_disturbed_mask.clone()
        initial_pose = unwrapped.recovery_initial_roll_pitch_rad.clone()
        initial_ang_vel = unwrapped.recovery_initial_angular_velocity_rad_s.clone()
        initial_tilt = _tilt(robot.data.projected_gravity_b).clone()

        active = torch.ones(args_cli.num_envs, dtype=torch.bool, device=unwrapped.device)
        completed_length = torch.zeros(args_cli.num_envs, dtype=torch.long, device=unwrapped.device)
        success_timeout = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=unwrapped.device)
        max_tilt = initial_tilt.clone()
        max_ang_vel = torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=-1)
        settled_run = torch.zeros(args_cli.num_envs, dtype=torch.long, device=unwrapped.device)
        recovery_step = torch.full(
            (args_cli.num_envs,), -1, dtype=torch.long, device=unwrapped.device
        )
        termination_counts = {
            name: 0 for name in unwrapped.termination_manager.active_terms
        }
        zero = torch.zeros((args_cli.num_envs, 14), device=unwrapped.device)
        finite = True

        for step in range(args_cli.steps):
            tilt = _tilt(robot.data.projected_gravity_b)
            angular_speed = torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=-1)
            max_tilt = torch.where(active, torch.maximum(max_tilt, tilt), max_tilt)
            max_ang_vel = torch.where(active, torch.maximum(max_ang_vel, angular_speed), max_ang_vel)
            settled_now = (
                (tilt <= args_cli.settled_tilt_rad)
                & (angular_speed <= args_cli.settled_angular_velocity_rad_s)
                & active
            )
            settled_run = torch.where(settled_now, settled_run + 1, torch.zeros_like(settled_run))
            newly_recovered = (
                (recovery_step < 0)
                & (settled_run >= args_cli.settled_consecutive_steps)
                & disturbed
            )
            recovery_step[newly_recovered] = step - args_cli.settled_consecutive_steps + 1

            _obs, _reward, terminated, truncated, _extras = env.step(zero)
            done = (terminated | truncated) & active
            if done.any():
                completed_length[done] = step + 1
                success_timeout[done] = truncated[done] & (~terminated[done])
                for name in termination_counts:
                    termination_counts[name] += int(
                        (unwrapped.termination_manager.get_term(name) & done).sum().item()
                    )
                active[done] = False
            finite = finite and bool(
                torch.isfinite(robot.data.root_state_w).all()
                and torch.isfinite(robot.data.joint_pos).all()
                and torch.isfinite(robot.data.joint_vel).all()
            )
            if not active.any():
                break

        completed_length[active] = args_cli.steps
        recovery_time_s = recovery_step.float() * float(unwrapped.step_dt)
        recovered = recovery_step >= 0
        undisturbed = ~disturbed
        disturbance_contract_passed = bool(
            disturbed.any()
            and undisturbed.any()
            and torch.count_nonzero(initial_pose[undisturbed]) == 0
            and torch.count_nonzero(initial_ang_vel[undisturbed]) == 0
            and torch.max(torch.abs(initial_pose)) <= 0.035 + 1.0e-7
            and torch.max(torch.abs(initial_ang_vel)) <= 0.20 + 1.0e-7
            and 0.25 <= float(undisturbed.float().mean()) <= 0.45
        )
        result = {
            "schema_version": 1,
            "audit_id": "a3_base_stand_recovery_a_passive_baseline_v1",
            "task": "A3BaseStandRecoveryA-v0",
            "simulation_only": True,
            "controller": "Base14 PD_STAND plant plus exact zero residual",
            "num_envs": args_cli.num_envs,
            "policy_steps": args_cli.steps,
            "policy_dt_s": float(unwrapped.step_dt),
            "disturbance_contract": {
                "requested_undisturbed_fraction": 0.35,
                "realized_undisturbed_fraction": float(undisturbed.float().mean()),
                "roll_pitch_range_rad": [-0.035, 0.035],
                "angular_velocity_range_rad_s": [-0.20, 0.20],
                "actor_observes_disturbance_mask": False,
                "passed": disturbance_contract_passed,
            },
            "metrics": {
                "undisturbed_timeout_success_fraction": float(success_timeout[undisturbed].float().mean()),
                "disturbed_timeout_success_fraction": float(success_timeout[disturbed].float().mean()),
                "undisturbed_max_tilt_rad": _group_stats(max_tilt, undisturbed),
                "disturbed_max_tilt_rad": _group_stats(max_tilt, disturbed),
                "undisturbed_max_angular_velocity_rad_s": _group_stats(max_ang_vel, undisturbed),
                "disturbed_max_angular_velocity_rad_s": _group_stats(max_ang_vel, disturbed),
                "disturbed_recovered_fraction": float(recovered[disturbed].float().mean()),
                "disturbed_recovery_time_s": _group_stats(recovery_time_s, disturbed & recovered),
                "termination_term_counts": termination_counts,
            },
            "runtime_integrity_passed": finite,
            "passive_baseline_measured": disturbance_contract_passed and finite,
            "recovery_training_approved": False,
            "deployment_approved": False,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["passive_baseline_measured"] else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
