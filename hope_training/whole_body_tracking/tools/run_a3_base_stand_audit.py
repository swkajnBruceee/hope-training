#!/usr/bin/env python3
"""Audit reset and bounded deterministic rollouts of ``A3BaseStand-v0``.

This runner is intentionally not a PPO entry point.  It exercises only the
sequence authorized by ``stand_fixture_gate_v1`` and writes evidence without
promoting any later training or deployment gate.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--task",
    choices=(
        "A3BaseStand-v0",
        "A3BaseStandAuthorityCandidate-v0",
        "A3BaseStandClipCandidate-v0",
        "A3BaseStandAuthorityClipCandidate-v0",
        "A3BaseStandPassiveStableCandidate-v0",
        "A3CatchReadyStand-v0",
    ),
    default="A3BaseStand-v0",
)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--zero-steps", type=int, default=250)
parser.add_argument("--scripted-steps", type=int, default=100)
parser.add_argument("--random-steps", type=int, default=100)
parser.add_argument("--random-action-abs", type=float, default=0.05)
parser.add_argument(
    "--traceback-after-s",
    type=float,
    help="Diagnostic: dump Python thread stacks if the audit has not progressed after this many seconds.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1:
    parser.error("--num-envs must be positive")
if min(args_cli.zero_steps, args_cli.scripted_steps, args_cli.random_steps) < 0:
    parser.error("rollout lengths must be non-negative")
if not 0.0 <= args_cli.random_action_abs <= 0.25:
    parser.error("--random-action-abs must be inside the approved [0, 0.25] clip")
if args_cli.traceback_after_s is not None:
    if args_cli.traceback_after_s <= 0.0:
        parser.error("--traceback-after-s must be positive")
    faulthandler.enable()
    faulthandler.dump_traceback_later(args_cli.traceback_after_s, repeat=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

import training.tasks.base_locomotion.config.a3  # noqa: F401
from training.robots.agibot_a3 import A3_BACKEND_JOINTS, A3_BASE_ACTION_JOINTS
from training.tasks.base_locomotion.mdp.observations import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM,
    HISTORY_LENGTH,
    PROPRIO_DIM,
)


def _tilt_rad(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.clamp(-projected_gravity_b[:, 2], min=-1.0, max=1.0))


def _fresh_reset(env):
    obs, _ = env.reset(seed=0)
    return obs


def _stage(env, name: str, steps: int, action_fn) -> dict:
    _fresh_reset(env)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    min_height = math.inf
    max_tilt = 0.0
    reward_sum = 0.0
    terminated_count = 0
    truncated_count = 0
    nonfinite_count = 0
    max_abs_action = 0.0
    first_non_timeout_termination_step = None
    termination_term_counts = {
        term_name: 0 for term_name in unwrapped.termination_manager.active_terms
    }
    sensor = unwrapped.scene.sensors["contact_forces"]
    max_contact_force_by_body = torch.zeros(len(sensor.body_names), device=unwrapped.device)
    for step in range(steps):
        action = action_fn(step).to(unwrapped.device)
        max_abs_action = max(max_abs_action, float(torch.max(torch.abs(action)).item()))
        obs, reward, terminated, truncated, _ = env.step(action)
        actor = obs["policy"]
        critic = obs["critic"]
        finite = (
            torch.isfinite(actor).all()
            and torch.isfinite(critic).all()
            and torch.isfinite(reward).all()
            and torch.isfinite(robot.data.root_state_w).all()
            and torch.isfinite(robot.data.joint_pos).all()
            and torch.isfinite(robot.data.joint_vel).all()
        )
        if not bool(finite):
            nonfinite_count += 1
        min_height = min(min_height, float(robot.data.root_pos_w[:, 2].min().item()))
        max_tilt = max(max_tilt, float(_tilt_rad(robot.data.projected_gravity_b).max().item()))
        reward_sum += float(reward.mean().item())
        terminated_count += int(terminated.sum().item())
        truncated_count += int(truncated.sum().item())
        for term_name in termination_term_counts:
            termination_term_counts[term_name] += int(
                unwrapped.termination_manager.get_term(term_name).sum().item()
            )
        if (
            first_non_timeout_termination_step is None
            and any(
                bool(unwrapped.termination_manager.get_term(term_name).any())
                for term_name in termination_term_counts
                if term_name != "time_out"
            )
        ):
            first_non_timeout_termination_step = step + 1
        contact_force = torch.linalg.vector_norm(sensor.data.net_forces_w, dim=-1).amax(dim=0)
        max_contact_force_by_body = torch.maximum(max_contact_force_by_body, contact_force)
    non_timeout_termination_count = sum(
        count for term_name, count in termination_term_counts.items() if term_name != "time_out"
    )
    top_contact_forces = sorted(
        (
            {"body_name": body_name, "max_force_n": float(force)}
            for body_name, force in zip(sensor.body_names, max_contact_force_by_body.tolist())
            if force > 0.0
        ),
        key=lambda item: item["max_force_n"],
        reverse=True,
    )[:8]
    return {
        "stage": name,
        "steps": steps,
        "max_abs_raw_action": max_abs_action,
        "min_root_height_m": min_height if steps else None,
        "max_root_tilt_rad": max_tilt if steps else None,
        "mean_reward_per_step": reward_sum / steps if steps else None,
        "terminated_env_count": terminated_count,
        "truncated_env_count": truncated_count,
        "termination_term_counts": termination_term_counts,
        "non_timeout_termination_count": non_timeout_termination_count,
        "first_non_timeout_termination_step": first_non_timeout_termination_step,
        "top_contact_forces": top_contact_forces,
        "nonfinite_step_count": nonfinite_count,
        "passed_finite": nonfinite_count == 0,
        "completed_without_safety_reset": non_timeout_termination_count == 0,
        "passed_runtime_integrity": nonfinite_count == 0,
    }


def main() -> int:
    env = None
    try:
        cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = args_cli.num_envs
        cfg.sim.device = args_cli.device
        env = gym.make(args_cli.task, cfg=cfg)
        unwrapped = env.unwrapped
        obs = _fresh_reset(env)
        action_term = unwrapped.action_manager.get_term("base")
        robot = unwrapped.scene["robot"]

        actor = obs["policy"]
        critic = obs["critic"]
        actor_history = actor[:, : HISTORY_LENGTH * PROPRIO_DIM].reshape(
            args_cli.num_envs, HISTORY_LENGTH, PROPRIO_DIM
        )
        reset_history_repeated = bool(
            torch.equal(actor_history, actor_history[:, :1].repeat(1, HISTORY_LENGTH, 1))
        )
        reset_previous_actions_zero = bool(torch.count_nonzero(actor_history[:, :, -14:]) == 0)
        context = actor[:, HISTORY_LENGTH * PROPRIO_DIM :]
        expected_command = torch.tensor(
            [0.0, 0.0, 0.0, 1.0684, 0.0], device=unwrapped.device
        ).repeat(args_cli.num_envs, 1)
        reset_contract = {
            "action_dimension": unwrapped.action_manager.total_action_dim,
            "actor_observation_dimension": actor.shape[-1],
            "critic_observation_dimension": critic.shape[-1],
            "actor_critic_shared_prefix_equal": bool(torch.equal(actor, critic[:, :ACTOR_OBSERVATION_DIM])),
            "history_repeats_current_proprioception": reset_history_repeated,
            "history_previous_action_is_zero": reset_previous_actions_zero,
            "base_command_is_zero_velocity_nominal_height": bool(torch.allclose(context[:, :5], expected_command)),
            "strike_context_is_zero": bool(torch.count_nonzero(context[:, 5:]) == 0),
            "actor_finite": bool(torch.isfinite(actor).all()),
            "critic_finite": bool(torch.isfinite(critic).all()),
            "backend_joint_set_matches": set(robot.joint_names) == set(A3_BACKEND_JOINTS),
            "base_joint_order_matches": list(action_term.cfg.base_joint_names) == list(A3_BASE_ACTION_JOINTS),
            "raw_action_clip_abs": float(action_term.cfg.raw_clip),
            "target_transport": "zero_order_hold",
        }
        reset_contract["passed"] = all(
            (
                reset_contract["action_dimension"] == 14,
                reset_contract["actor_observation_dimension"] == ACTOR_OBSERVATION_DIM,
                reset_contract["critic_observation_dimension"] == CRITIC_OBSERVATION_DIM,
                reset_contract["actor_critic_shared_prefix_equal"],
                reset_contract["history_repeats_current_proprioception"],
                reset_contract["history_previous_action_is_zero"],
                reset_contract["base_command_is_zero_velocity_nominal_height"],
                reset_contract["strike_context_is_zero"],
                reset_contract["actor_finite"],
                reset_contract["critic_finite"],
                reset_contract["backend_joint_set_matches"],
                reset_contract["base_joint_order_matches"],
                reset_contract["raw_action_clip_abs"]
                == (
                    0.5
                    if args_cli.task
                    in ("A3BaseStandClipCandidate-v0", "A3BaseStandAuthorityClipCandidate-v0")
                    else 0.25
                ),
            )
        )

        zeros = lambda _step: torch.zeros((args_cli.num_envs, 14))

        def scripted(step: int) -> torch.Tensor:
            action = torch.zeros((args_cli.num_envs, 14))
            action[:, step % 14] = 0.05 if (step // 14) % 2 == 0 else -0.05
            return action

        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260718)

        def bounded_random(_step: int) -> torch.Tensor:
            return (
                2.0 * torch.rand((args_cli.num_envs, 14), generator=generator) - 1.0
            ) * args_cli.random_action_abs

        stages = [
            _stage(env, "zero_action", args_cli.zero_steps, zeros),
            _stage(env, "bounded_scripted_action", args_cli.scripted_steps, scripted),
            _stage(env, "bounded_random_action", args_cli.random_steps, bounded_random),
        ]

        # Verify composition independently of physics response using one bounded
        # action vector and the action term's actual target buffer.
        probe = torch.zeros((args_cli.num_envs, 14), device=unwrapped.device)
        probe[:, 0] = 0.05
        probe[:, 12] = -0.05
        unwrapped.action_manager.process_action(probe)
        expected = robot.data.default_joint_pos.clone()
        base_ids, names = robot.find_joints(list(A3_BASE_ACTION_JOINTS), preserve_order=True)
        expected[:, base_ids] += probe * action_term._scale
        composer_passed = names == list(A3_BASE_ACTION_JOINTS) and torch.allclose(
            action_term.full_joint_targets, expected, atol=1.0e-7, rtol=0.0
        )

        result = {
            "schema_version": 1,
            "audit_id": "a3_base_stand_deterministic_audit_v1",
            "task": args_cli.task,
            "simulation_only": True,
            "physics_dt_s": float(unwrapped.physics_dt),
            "policy_dt_s": float(unwrapped.step_dt),
            "num_envs": args_cli.num_envs,
            "reset_contract": reset_contract,
            "composer_full_target_passed": bool(composer_passed),
            "stages": stages,
            "zero_action_baseline_stable_for_requested_window": stages[0][
                "completed_without_safety_reset"
            ],
            "stand_long_training_approved": False,
            "locomotion_command_approved": False,
            "deployment_approved": False,
        }
        result["passed"] = bool(
            reset_contract["passed"]
            and composer_passed
            and all(
                stage["passed_runtime_integrity"]
                for stage in stages
            )
        )
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    except BaseException:
        # Print before Kit shutdown: on some headless Isaac builds shutdown can
        # block while handling a partially initialized environment, otherwise
        # the actual manager/configuration exception is hidden.
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
