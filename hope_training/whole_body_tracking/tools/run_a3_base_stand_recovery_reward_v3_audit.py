#!/usr/bin/env python3
"""Numerically audit Recovery-A reward-v3 masking and action semantics."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

import training.tasks.base_locomotion.config.a3  # noqa: F401


def _step_terms(env) -> dict[str, list[float]]:
    manager = env.unwrapped.reward_manager
    dt = float(env.unwrapped.step_dt)
    return {
        name: [float(value) for value in (manager._step_reward[:, index] * dt).tolist()]
        for index, name in enumerate(manager.active_terms)
    }


def main() -> int:
    env = None
    try:
        cfg = gym.spec("A3BaseStandRecoveryA-v0").kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = 2
        cfg.seed = 0
        cfg.sim.device = args_cli.device
        # Reward masking is isolated from dynamics here: both environments are
        # clean, then one mask is set to disturbed solely for the term audit.
        cfg.events.reset_all.params["undisturbed_fraction"] = 1.0
        env = gym.make("A3BaseStandRecoveryA-v0", cfg=cfg)
        env.reset(seed=0)
        unwrapped = env.unwrapped
        dt = float(unwrapped.step_dt)

        zero = torch.zeros((2, 14), device=unwrapped.device)
        clean_action = zero.clone()
        clean_action[1] = 0.10
        _obs, clean_reward, terminated, truncated, _extras = env.step(clean_action)
        clean_terms = _step_terms(env)

        # Start a fresh reward state.  The manager reset makes the first
        # progress sample exactly zero, preventing reset leakage.
        env.reset(seed=0)
        unwrapped.recovery_disturbed_mask[:] = torch.tensor(
            [False, True], dtype=torch.bool, device=unwrapped.device
        )
        robot = unwrapped.scene["robot"]
        pose = robot.data.root_state_w[:, :7].clone()
        from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

        delta = quat_from_euler_xyz(
            torch.tensor([0.0, 0.035], device=unwrapped.device),
            torch.zeros(2, device=unwrapped.device),
            torch.zeros(2, device=unwrapped.device),
        )
        pose[:, 3:7] = quat_mul(pose[:, 3:7], delta)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_vel_w))

        _obs, first_reward, first_terminated, first_truncated, _extras = env.step(zero)
        first_terms = _step_terms(env)
        previous_error = torch.sum(
            torch.square(robot.data.projected_gravity_b[:, :2]), dim=-1
        ).clone()
        _obs, second_reward, second_terminated, second_truncated, _extras = env.step(zero)
        current_error = torch.sum(
            torch.square(robot.data.projected_gravity_b[:, :2]), dim=-1
        ).clone()
        second_terms = _step_terms(env)
        # The term returns an error-reduction rate; RewardManager's dt then
        # integrates it back to 2.0 * (previous_error - current_error).
        expected_integrated_progress = 2.0 * (previous_error - current_error)

        finite = bool(
            torch.isfinite(clean_reward).all()
            and torch.isfinite(first_reward).all()
            and torch.isfinite(second_reward).all()
        )
        clean_zero_action_term = clean_terms["undisturbed_action_magnitude"][0]
        clean_nonzero_action_term = clean_terms["undisturbed_action_magnitude"][1]
        passed = bool(
            finite
            and not bool((terminated | truncated).any())
            and not bool((first_terminated | first_truncated).any())
            and not bool((second_terminated | second_truncated).any())
            and abs(clean_zero_action_term) <= 1.0e-12
            and clean_nonzero_action_term < 0.0
            and abs(first_terms["recovery_tilt_progress"][0]) <= 1.0e-12
            and abs(first_terms["recovery_tilt_progress"][1]) <= 1.0e-12
            and abs(second_terms["recovery_tilt_progress"][0]) <= 1.0e-12
            and abs(
                second_terms["recovery_tilt_progress"][1]
                - float(expected_integrated_progress[1].item())
            )
            <= 1.0e-7
            and clean_reward[0] > 0.0
        )
        result = {
            "schema_version": 1,
            "audit_id": "a3_base_stand_recovery_reward_v3_audit_v1",
            "task": "A3BaseStandRecoveryA-v0",
            "simulation_only": True,
            "policy_dt_s": dt,
            "clean_action_mask_audit": {
                "raw_action_env0": 0.0,
                "raw_action_env1": 0.10,
                "undisturbed_action_term_env0": clean_zero_action_term,
                "undisturbed_action_term_env1": clean_nonzero_action_term,
                "zero_action_total_reward_positive": bool(clean_reward[0] > 0.0),
            },
            "recovery_progress_audit": {
                "clean_mask_progress_first_step": first_terms["recovery_tilt_progress"][0],
                "disturbed_mask_progress_first_step": first_terms["recovery_tilt_progress"][1],
                "clean_mask_progress_second_step": second_terms["recovery_tilt_progress"][0],
                "disturbed_mask_progress_second_step": second_terms["recovery_tilt_progress"][1],
                "expected_disturbed_progress_second_step": float(
                    expected_integrated_progress[1].item()
                ),
            },
            "runtime_integrity_passed": finite,
            "reward_v3_semantics_passed": passed,
            "recovery_training_approved": False,
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
