#!/usr/bin/env python3
"""Audit Stand failure causality and reward return without changing the policy.

The audit records first-event times for every rollout and a detailed trace for
environment zero.  Support margin is explicitly a geometric approximation;
the current ContactSensor does not expose true contact points or center of
pressure.
"""

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
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--task",
    choices=(
        "A3BaseStand-v0",
        "A3BaseStandAuthorityClipCandidate-v0",
        "A3BaseStandRecoveryA-v0",
    ),
    default="A3BaseStand-v0",
)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument(
    "--robot-asset-path",
    type=Path,
    help="Diagnostic-only URDF override; all policy and task parameters remain unchanged.",
)
parser.add_argument("--waist-depart-rad", type=float, default=0.05)
parser.add_argument("--ankle-target-depart-rad", type=float, default=0.05)
parser.add_argument("--joint-limit-proximity-rad", type=float, default=1.0e-3)
parser.add_argument(
    "--freeze-waist-pitch",
    action="store_true",
    help="Diagnostic-only: replace the policy waist-pitch action with zero (nominal target).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1 or args_cli.steps < 1:
    parser.error("--num-envs and --steps must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch
import trimesh
from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import training.tasks.base_locomotion.config.a3  # noqa: F401
from training.robots.agibot_a3 import A3_BASE_ACTION_JOINTS, A3_FEET_BODIES
from training.utils.ppo_cfg import load_ppo_params, runner_kwargs


CUTOFFS_S = (0.5, 1.0, 1.5, 2.0)


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _signed_margin(point: tuple[float, float], hull: list[tuple[float, float]]) -> float:
    if len(hull) < 3:
        return -math.inf
    distances = []
    for start, end in zip(hull, hull[1:] + hull[:1]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        edge_length = math.hypot(dx, dy)
        distances.append(
            (dx * (point[1] - start[1]) - dy * (point[0] - start[0])) / edge_length
        )
    return min(distances)


def _sole_vertices(side: str, device: str) -> torch.Tensor:
    path = (
        PROJECT_ROOT
        / "training"
        / "assets"
        / "agibot_a3"
        / "meshes"
        / f"{side}_ankle_roll_Link.STL"
    )
    mesh = trimesh.load(path, force="mesh")
    vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
    sole = vertices[vertices[:, 2] <= vertices[:, 2].min() + 0.005]
    if sole.shape[0] < 3:
        raise RuntimeError(f"Could not derive sole vertices from {path}")
    return sole


def _first(events: dict, name: str, time_s: float, **payload):
    if events[name] is None:
        events[name] = {"time_s": time_s, **payload}


def _new_episode(env_index: int, episode_index: int, reward_names: list[str]) -> dict:
    return {
        "env_index": env_index,
        "episode_index": episode_index,
        "age_steps": 0,
        "events": {
            "waist_pitch_depart": None,
            "waist_pitch_soft_limit": None,
            "com_margin_cross": None,
            "right_ankle_target_depart": None,
            "right_ankle_actual_limit": None,
            "right_ankle_torque_peak": None,
            "base_height_fail": None,
            "termination": None,
        },
        "reward_components_integrated": {name: 0.0 for name in reward_names},
        "return_integrated": 0.0,
        "cutoff_returns": {},
        "right_ankle_torque_peak_abs_nm": -1.0,
    }


def _finalize_episode(state: dict, status: str, termination_labels: list[str]) -> dict:
    available = [state["cutoff_returns"][str(value)] for value in CUTOFFS_S if str(value) in state["cutoff_returns"]]
    monotone = all(
        later["return_integrated"] >= earlier["return_integrated"] - 1.0e-9
        for earlier, later in zip(available, available[1:])
    )
    return {
        "env_index": state["env_index"],
        "episode_index": state["episode_index"],
        "status": status,
        "length_steps": state["age_steps"],
        "termination_labels": termination_labels,
        "events": state["events"],
        "return_integrated": state["return_integrated"],
        "reward_components_integrated": state["reward_components_integrated"],
        "hypothetical_cutoff_returns": state["cutoff_returns"],
        "available_cutoff_returns_monotone_non_decreasing": monotone if len(available) > 1 else None,
    }


def main() -> int:
    gym_env = None
    try:
        checkpoint = args_cli.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = 0
        env_cfg.sim.device = args_cli.device
        robot_asset_override = None
        if args_cli.robot_asset_path is not None:
            robot_asset_path = args_cli.robot_asset_path.expanduser().resolve()
            if not robot_asset_path.is_file():
                raise FileNotFoundError(robot_asset_path)
            env_cfg.scene.robot.spawn.asset_path = str(robot_asset_path)
            env_cfg.scene.robot.spawn.force_usd_conversion = True
            env_cfg.scene.robot.spawn.usd_dir = str(
                args_cli.output.expanduser().resolve().parent / "contact_policy_usd"
            )
            robot_asset_override = {
                "path": str(robot_asset_path),
                "sha256": hashlib.sha256(robot_asset_path.read_bytes()).hexdigest(),
            }
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        vec_env = RslRlVecEnvWrapper(gym_env)

        runner_cfg = RslRlOnPolicyRunnerCfg(
            **runner_kwargs(load_ppo_params(), "a3_base_stand_causal_audit")
        )
        runner_cfg.device = args_cli.device
        runner = OnPolicyRunner(vec_env, runner_cfg.to_dict(), log_dir=None, device=args_cli.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        obs, _ = vec_env.reset()
        env = vec_env.unwrapped
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("base")
        contact_sensor = env.scene.sensors["contact_forces"]
        foot_ids, foot_names = robot.find_bodies(A3_FEET_BODIES, preserve_order=True)
        if foot_names != A3_FEET_BODIES:
            raise RuntimeError(f"A3 foot mapping changed: {foot_names}")
        sensor_foot_ids = [contact_sensor.body_names.index(name) for name in A3_FEET_BODIES]
        base_joint_ids, base_joint_names = robot.find_joints(A3_BASE_ACTION_JOINTS, preserve_order=True)
        if base_joint_names != A3_BASE_ACTION_JOINTS:
            raise RuntimeError(f"A3 Base joint mapping changed: {base_joint_names}")
        waist_action_index = A3_BASE_ACTION_JOINTS.index("waist_pitch_joint")
        ankle_action_index = A3_BASE_ACTION_JOINTS.index("right_ankle_pitch_joint")
        waist_joint_id = base_joint_ids[waist_action_index]
        ankle_joint_id = base_joint_ids[ankle_action_index]
        joint_limit_term = env.termination_manager.get_term_cfg("joint_limit").func
        base_height_term = env.termination_manager.get_term_cfg("base_height").func

        reward_names = list(env.reward_manager.active_terms)
        reward_weights = {
            name: float(env.reward_manager.get_term_cfg(name).weight) for name in reward_names
        }
        reward_name_to_index = {name: index for index, name in enumerate(reward_names)}
        reward_term_presence = {
            "alive": "alive" in reward_names,
            "base_height": "base_height" in reward_names,
            "torque_or_power": any("torque" in name or "power" in name for name in reward_names),
            "termination_penalty": any("termination" in name for name in reward_names),
        }

        sole_vertices = [
            _sole_vertices("left", env.device),
            _sole_vertices("right", env.device),
        ]
        masses = robot.data.default_mass[0].to(env.device)
        episode_states = [_new_episode(i, 0, reward_names) for i in range(args_cli.num_envs)]
        episode_counts = [0] * args_cli.num_envs
        completed: list[dict] = []
        trace_env0: list[dict] = []
        runtime_finite = True
        max_reward_reconstruction_error = 0.0
        policy_dt = float(env.step_dt)

        for global_step in range(args_cli.steps):
            # Capture the causal input state before publishing this policy command.
            body_com = robot.data.body_com_pos_w
            system_com = torch.sum(body_com * masses[None, :, None], dim=1) / masses.sum()
            q = robot.data.joint_pos.clone()
            dq = robot.data.joint_vel.clone()
            torque = robot.data.applied_torque.clone()
            root_height = robot.data.root_pos_w[:, 2].clone()
            hard_limits = robot.data.joint_pos_limits.clone()
            soft_limits = robot.data.soft_joint_pos_limits.clone()
            default_q = robot.data.default_joint_pos.clone()
            foot_positions = robot.data.body_pos_w[:, foot_ids].clone()
            foot_quaternions = robot.data.body_quat_w[:, foot_ids].clone()
            foot_velocities = robot.data.body_lin_vel_w[:, foot_ids].clone()
            foot_forces = contact_sensor.data.net_forces_w[:, sensor_foot_ids].clone()

            support_margins = []
            for env_index in range(args_cli.num_envs):
                support_points: list[tuple[float, float]] = []
                for side_index in range(2):
                    local = sole_vertices[side_index]
                    world = math_utils.quat_apply(
                        foot_quaternions[env_index, side_index].unsqueeze(0).expand(local.shape[0], -1),
                        local,
                    ) + foot_positions[env_index, side_index]
                    support_points.extend(
                        (float(point[0]), float(point[1])) for point in world[:, :2].cpu().tolist()
                    )
                hull = _convex_hull(support_points)
                support_margins.append(
                    _signed_margin(
                        (float(system_com[env_index, 0]), float(system_com[env_index, 1])), hull
                    )
                )

            with torch.inference_mode():
                raw_action = policy(obs)
                if args_cli.freeze_waist_pitch:
                    raw_action[:, waist_action_index] = 0.0
            obs, reward, dones, _extras = vec_env.step(raw_action)
            effective_action = action_term.raw_actions.clone()
            target = action_term.processed_actions.clone()
            reward_rates = env.reward_manager._step_reward.clone()  # weighted rates; diagnostic API
            reconstructed_reward = reward_rates.sum(dim=1) * policy_dt
            max_reward_reconstruction_error = max(
                max_reward_reconstruction_error,
                float(torch.max(torch.abs(reconstructed_reward - reward)).item()),
            )
            done_mask = dones.bool()
            termination_masks = {
                name: env.termination_manager.get_term(name).clone()
                for name in env.termination_manager.active_terms
            }

            for env_index in range(args_cli.num_envs):
                state = episode_states[env_index]
                time_pre = state["age_steps"] * policy_dt
                time_post = (state["age_steps"] + 1) * policy_dt
                events = state["events"]
                waist_q = float(q[env_index, waist_joint_id])
                waist_nominal = float(default_q[env_index, waist_joint_id])
                ankle_q = float(q[env_index, ankle_joint_id])
                ankle_nominal = float(default_q[env_index, ankle_joint_id])
                waist_target = float(target[env_index, waist_action_index])
                ankle_target = float(target[env_index, ankle_action_index])
                ankle_torque = float(torque[env_index, ankle_joint_id])
                waist_soft = soft_limits[env_index, waist_joint_id]
                ankle_hard = hard_limits[env_index, ankle_joint_id]

                if abs(waist_q - waist_nominal) >= args_cli.waist_depart_rad:
                    _first(events, "waist_pitch_depart", time_pre, q_actual_rad=waist_q)
                if waist_q <= float(waist_soft[0]) or waist_q >= float(waist_soft[1]):
                    _first(events, "waist_pitch_soft_limit", time_pre, q_actual_rad=waist_q)
                if support_margins[env_index] < 0.0:
                    _first(
                        events,
                        "com_margin_cross",
                        time_pre,
                        support_margin_approx_m=support_margins[env_index],
                    )
                if abs(ankle_target - ankle_nominal) >= args_cli.ankle_target_depart_rad:
                    _first(
                        events,
                        "right_ankle_target_depart",
                        time_pre,
                        q_target_rad=ankle_target,
                        action_contribution_rad=ankle_target - ankle_nominal,
                    )
                ankle_margin = min(ankle_q - float(ankle_hard[0]), float(ankle_hard[1]) - ankle_q)
                if ankle_margin <= args_cli.joint_limit_proximity_rad:
                    _first(
                        events,
                        "right_ankle_actual_limit",
                        time_pre,
                        q_actual_rad=ankle_q,
                        hard_limit_margin_rad=ankle_margin,
                        source="sampled_pre_step_proximity",
                    )
                if abs(ankle_torque) > state["right_ankle_torque_peak_abs_nm"]:
                    state["right_ankle_torque_peak_abs_nm"] = abs(ankle_torque)
                    events["right_ankle_torque_peak"] = {
                        "time_s": time_pre,
                        "applied_torque_nm": ankle_torque,
                    }

                for reward_name, reward_index in reward_name_to_index.items():
                    integrated = float(reward_rates[env_index, reward_index]) * policy_dt
                    state["reward_components_integrated"][reward_name] += integrated
                state["return_integrated"] += float(reward[env_index])
                state["age_steps"] += 1
                for cutoff in CUTOFFS_S:
                    key = str(cutoff)
                    if key not in state["cutoff_returns"] and time_post + 1.0e-9 >= cutoff:
                        state["cutoff_returns"][key] = {
                            "time_s": time_post,
                            "return_integrated": state["return_integrated"],
                            "reward_components_integrated": dict(state["reward_components_integrated"]),
                        }

                labels = [name for name, mask in termination_masks.items() if bool(mask[env_index])]
                if bool(termination_masks["joint_limit"][env_index]) and bool(
                    joint_limit_term.violation_mask[env_index, ankle_joint_id]
                ):
                    terminal_q = float(joint_limit_term.joint_position_rad[env_index, ankle_joint_id])
                    _first(
                        events,
                        "right_ankle_actual_limit",
                        time_post,
                        q_actual_rad=terminal_q,
                        hard_limit_excess_rad=float(
                            joint_limit_term.excess_rad[env_index, ankle_joint_id]
                        ),
                        q_target_rad=ankle_target,
                        applied_torque_nm=float(
                            joint_limit_term.applied_torque_nm[env_index, ankle_joint_id]
                        ),
                        source="terminal_predicate_snapshot",
                    )
                if bool(termination_masks["base_height"][env_index]):
                    _first(
                        events,
                        "base_height_fail",
                        time_post,
                        root_height_m=float(base_height_term.height_m[env_index]),
                    )
                if bool(done_mask[env_index]):
                    _first(events, "termination", time_post, labels=labels)

                if env_index == 0:
                    foot_fz = [max(0.0, float(value)) for value in foot_forces[0, :, 2]]
                    total_fz = sum(foot_fz)
                    trace_env0.append(
                        {
                            "global_policy_step": global_step,
                            "episode_index": state["episode_index"],
                            "episode_time_pre_s": time_pre,
                            "episode_time_post_s": time_post,
                            "waist_pitch": {
                                "q_nominal_rad": waist_nominal,
                                "q_target_rad": waist_target,
                                "q_actual_rad": waist_q,
                                "dq_rad_s": float(dq[0, waist_joint_id]),
                                "applied_torque_nm": float(torque[0, waist_joint_id]),
                                "raw_action": float(effective_action[0, waist_action_index]),
                                "action_contribution_rad": waist_target - waist_nominal,
                            },
                            "right_ankle_pitch": {
                                "q_nominal_rad": ankle_nominal,
                                "q_target_rad": ankle_target,
                                "q_actual_rad": ankle_q,
                                "dq_rad_s": float(dq[0, ankle_joint_id]),
                                "applied_torque_nm": ankle_torque,
                                "raw_action": float(effective_action[0, ankle_action_index]),
                                "hard_limit_margin_rad": ankle_margin,
                            },
                            "support": {
                                "system_com_w_m": system_com[0].tolist(),
                                "support_margin_approx_m": support_margins[0],
                                "root_height_m": float(root_height[0]),
                                "foot_normal_load_fraction": [
                                    value / total_fz if total_fz > 1.0e-6 else None for value in foot_fz
                                ],
                                "foot_origin_w_m": foot_positions[0].tolist(),
                                "foot_linear_velocity_w_mps": foot_velocities[0].tolist(),
                            },
                            "reward_integrated_this_step": float(reward[0]),
                            "reward_components_integrated_this_step": {
                                name: float(reward_rates[0, index]) * policy_dt
                                for name, index in reward_name_to_index.items()
                            },
                            "termination_labels": labels,
                        }
                    )

                if bool(done_mask[env_index]):
                    completed.append(_finalize_episode(state, "terminated", labels))
                    episode_counts[env_index] += 1
                    episode_states[env_index] = _new_episode(
                        env_index, episode_counts[env_index], reward_names
                    )

            finite = (
                torch.isfinite(obs).all()
                and torch.isfinite(reward).all()
                and torch.isfinite(raw_action).all()
                and torch.isfinite(robot.data.root_state_w).all()
                and torch.isfinite(robot.data.joint_pos).all()
                and torch.isfinite(robot.data.joint_vel).all()
            )
            runtime_finite = runtime_finite and bool(finite)

        censored = [
            _finalize_episode(state, "censored", [])
            for state in episode_states
            if state["age_steps"] > 0
        ]
        analyzed = completed + censored
        completed_returns = [item["return_integrated"] for item in completed]
        completed_lengths = [item["length_steps"] for item in completed]
        if len(completed) >= 2 and len(set(completed_lengths)) > 1:
            lengths_tensor = torch.tensor(completed_lengths, dtype=torch.float64)
            returns_tensor = torch.tensor(completed_returns, dtype=torch.float64)
            return_length_correlation = float(
                torch.corrcoef(torch.stack((lengths_tensor, returns_tensor)))[0, 1]
            )
        else:
            return_length_correlation = None
        monotone_values = [
            item["available_cutoff_returns_monotone_non_decreasing"]
            for item in analyzed
            if item["available_cutoff_returns_monotone_non_decreasing"] is not None
        ]
        event_presence = {
            name: sum(item["events"][name] is not None for item in analyzed)
            for name in analyzed[0]["events"]
        } if analyzed else {}

        result = {
            "schema_version": 1,
            "audit_id": "a3_base_stand_causal_reward_audit_v1",
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "num_envs": args_cli.num_envs,
            "requested_policy_steps": args_cli.steps,
            "physics_dt_s": float(env.physics_dt),
            "policy_dt_s": policy_dt,
            "thresholds": {
                "waist_pitch_depart_rad": args_cli.waist_depart_rad,
                "right_ankle_target_depart_rad": args_cli.ankle_target_depart_rad,
                "joint_limit_proximity_rad": args_cli.joint_limit_proximity_rad,
                "base_height_minimum_m": 0.75,
            },
            "diagnostic_intervention": {
                "freeze_waist_pitch_to_nominal": args_cli.freeze_waist_pitch,
                "robot_asset_override": robot_asset_override,
                "other_action_dimensions_preserved_from_policy": True,
                "training_or_contract_change_authorized": False,
            },
            "support_semantics": {
                "margin": "convex_hull_of_lowest_5mm_urdf_collision_mesh_vertices_projected_to_world_xy_approximation",
                "true_contact_points_available": False,
                "center_of_pressure_available": False,
                "contact_load": "ContactSensor net force per foot; not a contact-point distribution",
            },
            "reward_semantics": {
                "active_terms": reward_names,
                "weights": reward_weights,
                "term_presence": reward_term_presence,
                "manager_step_terms_are_weighted_rates": True,
                "reported_component_returns_are_rate_times_policy_dt": True,
                "hypothetical_cutoffs_add_no_unconfigured_terminal_penalty": True,
                "max_return_reconstruction_abs_error": max_reward_reconstruction_error,
            },
            "completed_episode_count": len(completed),
            "censored_episode_count": len(censored),
            "mean_completed_length_steps": (
                sum(completed_lengths) / len(completed_lengths) if completed_lengths else None
            ),
            "mean_completed_return": (
                sum(completed_returns) / len(completed_returns) if completed_returns else None
            ),
            "completed_return_vs_length_pearson": return_length_correlation,
            "cutoff_monotone_episode_count": sum(value is True for value in monotone_values),
            "cutoff_nonmonotone_episode_count": sum(value is False for value in monotone_values),
            "event_presence_count": event_presence,
            "episodes": analyzed,
            "trace_env0": trace_env0,
            "runtime_integrity_passed": runtime_finite,
            "changes_authorized_by_this_report": [],
            "extended_smoke_approved": False,
            "stand_long_training_approved": False,
            "deployment_approved": False,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        summary = {key: value for key, value in result.items() if key not in ("episodes", "trace_env0")}
        print(json.dumps(summary, indent=2))
        return 0 if runtime_finite and max_reward_reconstruction_error < 1.0e-5 else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if gym_env is not None:
            gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
