#!/usr/bin/env python3
"""Run a bounded free-base waist-pitch working-point scan for Stand v0.

This is a diagnostic rollout, not training and not a gate promotion. Every
environment receives zero leg/waist-roll action and one constant, bounded
waist-pitch action. The result is evidence for a later control decision; it
does not itself authorize scale, gain, reward, termination, or gate changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import traceback
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--replicates", type=int, default=8)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument(
    "--waist-pitch-kp",
    type=float,
    help="Diagnostic-only stiffness override; omitted means the task value (50 Nm/rad).",
)
parser.add_argument(
    "--waist-pitch-kd",
    type=float,
    help="Diagnostic-only damping override; omitted means the task value (2 Nms/rad).",
)
parser.add_argument(
    "--action-values",
    type=float,
    nargs="+",
    default=(-0.25, -0.125, 0.0, 0.125, 0.25),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.replicates < 1 or args_cli.steps < 1:
    parser.error("--replicates and --steps must be positive")
if not args_cli.action_values:
    parser.error("--action-values cannot be empty")
if any(not math.isfinite(value) or abs(value) > 0.25 for value in args_cli.action_values):
    parser.error("all --action-values must be finite and inside [-0.25, 0.25]")
if args_cli.waist_pitch_kp is not None and (
    not math.isfinite(args_cli.waist_pitch_kp) or args_cli.waist_pitch_kp <= 0.0
):
    parser.error("--waist-pitch-kp must be finite and positive")
if args_cli.waist_pitch_kd is not None and (
    not math.isfinite(args_cli.waist_pitch_kd) or args_cli.waist_pitch_kd < 0.0
):
    parser.error("--waist-pitch-kd must be finite and non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

import training.tasks.base_locomotion.config.a3  # noqa: F401
from training.robots.agibot_a3 import A3_ANCHOR_BODY, A3_BASE_ACTION_JOINTS
from training.tasks.base_locomotion.base_env_cfg import A3_BASE_ACTION_SCALE_RAD


def _tilt_rad(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.clamp(-projected_gravity_b[:, 2], min=-1.0, max=1.0))


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": statistics.fmean(values), "max": max(values)}


def main() -> int:
    env = None
    try:
        num_values = len(args_cli.action_values)
        num_envs = num_values * args_cli.replicates
        cfg = gym.spec("A3BaseStand-v0").kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = num_envs
        cfg.seed = 0
        cfg.sim.device = args_cli.device
        waist_actuator_cfg = cfg.scene.robot.actuators["waist"]
        waist_stiffness = dict(waist_actuator_cfg.stiffness)
        waist_damping = dict(waist_actuator_cfg.damping)
        if args_cli.waist_pitch_kp is not None:
            waist_stiffness["waist_pitch_joint"] = args_cli.waist_pitch_kp
        if args_cli.waist_pitch_kd is not None:
            waist_damping["waist_pitch_joint"] = args_cli.waist_pitch_kd
        waist_actuator_cfg.stiffness = waist_stiffness
        waist_actuator_cfg.damping = waist_damping
        env = gym.make("A3BaseStand-v0", cfg=cfg)
        env.reset(seed=0)

        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        waist_action_index = list(A3_BASE_ACTION_JOINTS).index("waist_pitch_joint")
        waist_joint_ids, waist_joint_names = robot.find_joints(
            ["waist_pitch_joint"], preserve_order=True
        )
        torso_ids, torso_names = robot.find_bodies([A3_ANCHOR_BODY], preserve_order=True)
        if waist_joint_names != ["waist_pitch_joint"] or torso_names != [A3_ANCHOR_BODY]:
            raise RuntimeError("A3 waist-pitch or torso mapping changed")
        waist_joint_id = waist_joint_ids[0]
        torso_id = torso_ids[0]

        action_value_by_env = torch.tensor(
            [value for value in args_cli.action_values for _ in range(args_cli.replicates)],
            dtype=torch.float,
            device=unwrapped.device,
        )
        actions = torch.zeros((num_envs, 14), device=unwrapped.device)
        actions[:, waist_action_index] = action_value_by_env

        active = torch.ones(num_envs, dtype=torch.bool, device=unwrapped.device)
        ages = torch.zeros(num_envs, dtype=torch.long, device=unwrapped.device)
        completed_lengths = torch.zeros(num_envs, dtype=torch.long, device=unwrapped.device)
        termination_labels: list[list[str]] = [[] for _ in range(num_envs)]
        joint_limit_labels: list[list[str]] = [[] for _ in range(num_envs)]
        joint_limit_term = unwrapped.termination_manager.get_term_cfg("joint_limit").func
        waist_q_min = robot.data.joint_pos[:, waist_joint_id].clone()
        waist_q_max = waist_q_min.clone()
        waist_abs_torque_max = torch.zeros(num_envs, device=unwrapped.device)
        torso_tilt_max = torch.zeros(num_envs, device=unwrapped.device)
        root_height_min = robot.data.root_pos_w[:, 2].clone()
        nonfinite_step_count = 0

        for _step in range(args_cli.steps):
            if not bool(active.any()):
                break

            # Sample the still-live state. ManagerBasedRLEnv resets done envs
            # inside step(), so post-step terminal states would be ambiguous.
            waist_q = robot.data.joint_pos[:, waist_joint_id]
            waist_q_min[active] = torch.minimum(waist_q_min[active], waist_q[active])
            waist_q_max[active] = torch.maximum(waist_q_max[active], waist_q[active])
            waist_abs_torque_max[active] = torch.maximum(
                waist_abs_torque_max[active],
                torch.abs(robot.data.applied_torque[active, waist_joint_id]),
            )
            torso_quat = robot.data.body_quat_w[:, torso_id]
            torso_gravity = math_utils.quat_rotate_inverse(torso_quat, robot.data.GRAVITY_VEC_W)
            torso_tilt_max[active] = torch.maximum(
                torso_tilt_max[active], _tilt_rad(torso_gravity)[active]
            )
            root_height_min[active] = torch.minimum(
                root_height_min[active], robot.data.root_pos_w[active, 2]
            )

            obs, reward, terminated, truncated, _ = env.step(actions)
            ages[active] += 1
            finite = (
                torch.isfinite(obs["policy"]).all()
                and torch.isfinite(obs["critic"]).all()
                and torch.isfinite(reward).all()
                and torch.isfinite(robot.data.root_state_w).all()
                and torch.isfinite(robot.data.joint_pos).all()
                and torch.isfinite(robot.data.joint_vel).all()
            )
            if not bool(finite):
                nonfinite_step_count += 1

            first_done = active & (terminated | truncated)
            if first_done.any():
                for env_id in torch.nonzero(first_done, as_tuple=False).flatten().tolist():
                    termination_labels[env_id] = [
                        name
                        for name in unwrapped.termination_manager.active_terms
                        if bool(unwrapped.termination_manager.get_term(name)[env_id])
                    ]
                    if "joint_limit" in termination_labels[env_id]:
                        joint_limit_labels[env_id] = [
                            joint_name
                            for joint_name, violated in zip(
                                robot.joint_names,
                                joint_limit_term.violation_mask[env_id].tolist(),
                            )
                            if violated
                        ]
                completed_lengths[first_done] = ages[first_done]
                active[first_done] = False

        completed_lengths[active] = args_cli.steps

        default_q = float(robot.data.default_joint_pos[0, waist_joint_id].item())
        hard_limits = robot.data.joint_pos_limits[0, waist_joint_id].tolist()
        soft_limits = robot.data.soft_joint_pos_limits[0, waist_joint_id].tolist()
        scale_rad = float(A3_BASE_ACTION_SCALE_RAD[waist_action_index])
        groups = []
        for value_index, action_value in enumerate(args_cli.action_values):
            first = value_index * args_cli.replicates
            env_ids = range(first, first + args_cli.replicates)
            lengths = [float(completed_lengths[env_id].item()) for env_id in env_ids]
            labels = [label for env_id in env_ids for label in termination_labels[env_id]]
            limit_joints = [label for env_id in env_ids for label in joint_limit_labels[env_id]]
            groups.append(
                {
                    "normalized_action": action_value,
                    "target_residual_rad": action_value * scale_rad,
                    "unclipped_target_rad": default_q + action_value * scale_rad,
                    "survived_full_window_count": sum(
                        not termination_labels[env_id] for env_id in env_ids
                    ),
                    "first_episode_length_steps": _summary(lengths),
                    "termination_counts": dict(sorted(Counter(labels).items())),
                    "joint_limit_counts_by_joint": dict(sorted(Counter(limit_joints).items())),
                    "last_alive_waist_position_min_rad": min(
                        float(waist_q_min[env_id].item()) for env_id in env_ids
                    ),
                    "last_alive_waist_position_max_rad": max(
                        float(waist_q_max[env_id].item()) for env_id in env_ids
                    ),
                    "last_alive_waist_torque_max_abs_nm": max(
                        float(waist_abs_torque_max[env_id].item()) for env_id in env_ids
                    ),
                    "last_alive_torso_tilt_max_rad": max(
                        float(torso_tilt_max[env_id].item()) for env_id in env_ids
                    ),
                    "last_alive_root_height_min_m": min(
                        float(root_height_min[env_id].item()) for env_id in env_ids
                    ),
                }
            )

        result = {
            "schema_version": 1,
            "diagnostic_id": "a3_base_stand_waist_pitch_scan_v1",
            "task": "A3BaseStand-v0",
            "simulation_only": True,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "replicates_per_action": args_cli.replicates,
            "requested_policy_steps": args_cli.steps,
            "physics_dt_s": float(unwrapped.physics_dt),
            "policy_dt_s": float(unwrapped.step_dt),
            "waist_pitch_action_index": waist_action_index,
            "waist_pitch_action_scale_rad": scale_rad,
            "waist_pitch_kp_nm_per_rad": float(waist_stiffness["waist_pitch_joint"]),
            "waist_pitch_kd_nms_per_rad": float(waist_damping["waist_pitch_joint"]),
            "gain_override_is_diagnostic_only": (
                args_cli.waist_pitch_kp is not None or args_cli.waist_pitch_kd is not None
            ),
            "waist_pitch_default_position_rad": default_q,
            "waist_pitch_soft_limits_rad": soft_limits,
            "waist_pitch_hard_limits_rad": hard_limits,
            "groups": groups,
            "nonfinite_step_count": nonfinite_step_count,
            "runtime_integrity_passed": nonfinite_step_count == 0,
            "changes_authorized_by_this_report": [],
            "stand_phase1_qualified": False,
            "stand_long_training_approved": False,
            "deployment_approved": False,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["runtime_integrity_passed"] else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
