#!/usr/bin/env python3
"""Dynamic qualification replay for canonical priors in the formal P1 scene.

This is a deterministic PD replay, not a policy rollout and not training.  The
50 Hz motion is linearly resampled onto the formal scene's 90 Hz control clock;
the floating base and formal implicit actuators evolve at 360 Hz physics.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--canonical-manifest", type=Path, required=True)
parser.add_argument("--motion-ids", default="0,2,3,4,5")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference-mode", choices=("motion", "hold_frame0"), default="motion")
parser.add_argument("--post-hold-s", type=float, default=0.30)
parser.add_argument("--loaded-foot-force-n", type=float, default=20.0)
parser.add_argument("--non-foot-contact-force-n", type=float, default=20.0)
parser.add_argument("--dangerous-contact-force-n", type=float, default=100.0)
parser.add_argument("--max-loaded-foot-slip-mps", type=float, default=0.30)
parser.add_argument("--max-root-tilt-deg", type=float, default=20.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

from training.robots.agibot_a3 import A3_FEET_BODIES  # noqa: E402
from training.tasks.table_tennis.config.agibot_a3.table_tennis_env_cfg import (  # noqa: E402
    AgibotA3TableTennisEnvCfg,
)
from training.tasks.table_tennis.mdp.racket import racket_spatial_state_w  # noqa: E402


def _vec(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1)]


def _tilt_deg(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    x = quaternion_wxyz[:, 1]
    y = quaternion_wxyz[:, 2]
    r_zz = (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(r_zz))


def _angle_deg(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(lhs * rhs, dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def _interpolate(values: np.ndarray, source_fps: float, time_s: float) -> np.ndarray:
    coordinate = min(max(time_s * source_fps, 0.0), values.shape[0] - 1.0)
    lower = int(math.floor(coordinate))
    upper = min(lower + 1, values.shape[0] - 1)
    alpha = coordinate - lower
    return (1.0 - alpha) * values[lower] + alpha * values[upper]


def _classification(metrics: dict, self_collision_observable: bool) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if metrics["finite_state"] is False:
        return "D", ["non_finite_state"]
    if metrics["minimum_actual_hard_joint_margin_rad"] < -1.0e-4:
        reasons.append("actual_hard_joint_limit_violation")
    if metrics["minimum_root_height_w_m"] < -0.1:
        reasons.append("formal_robot_fell_termination")
    if metrics["max_non_foot_contact_force_n"] >= metrics["dangerous_contact_force_n"]:
        reasons.append("dangerous_non_foot_contact")
    if reasons:
        return "D", reasons

    if metrics["max_non_foot_contact_force_n"] >= metrics["non_foot_contact_force_n"]:
        reasons.append("non_foot_contact")
    if metrics["max_loaded_foot_tangential_speed_mps"] > metrics["max_loaded_foot_slip_mps"]:
        reasons.append("loaded_foot_slip")
    if metrics["max_root_tilt_deg"] > metrics["max_root_tilt_allowed_deg"]:
        reasons.append("root_tilt")
    if metrics["hit_position_error_m"] > 0.075:
        reasons.append("hit_position_not_preserved")
    if reasons:
        return "C", reasons

    if metrics["minimum_reference_soft_joint_margin_rad"] <= 0.0:
        reasons.append("reference_soft_joint_margin_non_positive")
    if metrics["minimum_actual_soft_joint_margin_rad"] <= 0.0:
        reasons.append("actual_soft_joint_margin_non_positive")
    if metrics["hit_normal_error_deg"] > 15.0:
        reasons.append("hit_normal_not_preserved")
    if metrics["hit_velocity_error_mps"] > 0.5:
        reasons.append("hit_velocity_not_preserved")
    if not self_collision_observable:
        reasons.append("self_collision_not_observable_in_formal_asset_contract")
    if reasons:
        return "B", reasons
    return "A", ["all_configured_dynamic_gates_passed"]


def _build_scene_cfg(env_cfg, num_envs: int):
    base_scene = env_cfg.scene
    base_scene_type = type(base_scene)

    @configclass
    class DynamicAuditSceneCfg(base_scene_type):
        robot_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
            debug_vis=False,
        )

    scene_cfg = DynamicAuditSceneCfg(
        num_envs=num_envs,
        env_spacing=float(base_scene.env_spacing),
    )
    for field in dataclasses.fields(base_scene):
        if field.name not in ("num_envs", "env_spacing"):
            setattr(scene_cfg, field.name, copy.deepcopy(getattr(base_scene, field.name)))
    for asset_name in ("floor", "table", "net", "ball"):
        asset_cfg = getattr(scene_cfg, asset_name)
        if hasattr(asset_cfg.spawn, "visual_material"):
            asset_cfg.spawn.visual_material = None
    for decoration_name in (
        "net_post_left",
        "net_post_right",
        "center_line",
        "light",
        "sky_light",
    ):
        setattr(scene_cfg, decoration_name, None)
    return scene_cfg


def _run() -> dict:
    manifest_path = args_cli.canonical_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested_ids = [int(value) for value in args_cli.motion_ids.split(",") if value.strip()]
    entries_by_id = {int(entry["motion_id"]): entry for entry in manifest["motions"]}
    if not requested_ids or any(motion_id not in entries_by_id for motion_id in requested_ids):
        raise ValueError("--motion-ids must select existing canonical motions")
    entries = [entries_by_id[motion_id] for motion_id in requested_ids]
    source_manifest = json.loads(Path(manifest["source_manifest"]).read_text(encoding="utf-8"))
    source_joint_names = list(source_manifest["momentum_preview_contract"]["joint_names"])

    arrays = []
    for entry in entries:
        path = manifest_path.parent / entry["canonical_motion_npz"]
        with np.load(path, allow_pickle=False) as data:
            arrays.append(
                {
                    "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
                    "joint_pos": np.asarray(data["joint_pos"], dtype=np.float32),
                    "joint_vel": np.asarray(data["joint_vel"], dtype=np.float32),
                    "root_pos": np.asarray(data["body_pos_b0"][:, 0], dtype=np.float32),
                    "root_quat": np.asarray(data["body_quat_b0_wxyz"][:, 0], dtype=np.float32),
                    "root_lin": np.asarray(data["body_lin_vel_b0"][:, 0], dtype=np.float32),
                    "root_ang": np.asarray(data["body_ang_vel_b0"][:, 0], dtype=np.float32),
                }
            )
    if len({item["fps"] for item in arrays}) != 1:
        raise ValueError("selected motions must share one source frequency")
    if len({item["joint_pos"].shape for item in arrays}) != 1:
        raise ValueError("selected motions must share one joint trajectory shape")

    env_cfg = AgibotA3TableTennisEnvCfg()
    env_cfg.sim.device = str(args_cli.device)
    env_cfg.scene.robot.spawn.fix_base = False
    num_envs = len(entries)
    scene_cfg = _build_scene_cfg(env_cfg, num_envs)
    sim = SimulationContext(env_cfg.sim)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    contacts = scene["robot_contacts"]
    ball = scene["ball"]
    device = sim.device
    fake_env = SimpleNamespace(scene=scene, device=device, num_envs=num_envs)

    missing = [name for name in source_joint_names if name not in robot.joint_names]
    if missing:
        raise RuntimeError(f"formal robot is missing source joints: {missing}")
    joint_ids = torch.as_tensor(
        [robot.joint_names.index(name) for name in source_joint_names],
        dtype=torch.long,
        device=device,
    )
    robot_foot_body_ids = torch.as_tensor(
        [robot.body_names.index(name) for name in A3_FEET_BODIES],
        dtype=torch.long,
        device=device,
    )
    sensor_foot_body_ids = torch.as_tensor(
        [contacts.body_names.index(name) for name in A3_FEET_BODIES],
        dtype=torch.long,
        device=device,
    )
    non_foot_sensor_ids = torch.as_tensor(
        [index for index, name in enumerate(contacts.body_names) if name not in A3_FEET_BODIES],
        dtype=torch.long,
        device=device,
    )

    env_origins = scene.env_origins
    default_root = robot.data.default_root_state.clone()
    # Articulation.data.default_root_state stores the configured environment-
    # local initial pose; env origins are added only when writing world state.
    p1_anchor_local = default_root[:, :3].clone()
    initial_q = robot.data.default_joint_pos.clone()
    initial_qd = torch.zeros_like(initial_q)
    for env_index, data in enumerate(arrays):
        initial_q[env_index, joint_ids] = torch.as_tensor(data["joint_pos"][0], device=device)
        if args_cli.reference_mode == "motion":
            initial_qd[env_index, joint_ids] = torch.as_tensor(data["joint_vel"][0], device=device)
        default_root[env_index, :3] = (
            env_origins[env_index]
            + p1_anchor_local[env_index]
            + torch.as_tensor(data["root_pos"][0], device=device)
        )
        default_root[env_index, 3:7] = torch.as_tensor(data["root_quat"][0], device=device)
        if args_cli.reference_mode == "motion":
            default_root[env_index, 7:10] = torch.as_tensor(data["root_lin"][0], device=device)
            default_root[env_index, 10:13] = torch.as_tensor(data["root_ang"][0], device=device)
        else:
            default_root[env_index, 7:13] = 0.0
    robot.write_root_state_to_sim(default_root)
    robot.write_joint_state_to_sim(initial_q, initial_qd)
    robot.set_joint_position_target(initial_q)
    robot.set_joint_velocity_target(torch.zeros_like(initial_qd))
    ball_state = ball.data.default_root_state.clone()
    ball_state[:, :3] = env_origins + torch.tensor((10.0, 10.0, 5.0), device=device)
    ball_state[:, 7:] = 0.0
    ball.write_root_state_to_sim(ball_state)
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)
    contacts.reset()

    physics_dt = float(env_cfg.sim.dt)
    decimation = int(env_cfg.decimation)
    control_dt = physics_dt * decimation
    source_fps = arrays[0]["fps"]
    motion_duration = (arrays[0]["joint_pos"].shape[0] - 1) / source_fps
    hit_times = [float(entry["hit_frame"]) / source_fps for entry in entries]
    replay_control_steps = int(math.ceil(motion_duration / control_dt))
    post_control_steps = int(math.ceil(float(args_cli.post_hold_s) / control_dt))
    total_control_steps = replay_control_steps + post_control_steps

    traces: list[list[dict]] = [[] for _ in entries]
    hit_samples: list[dict | None] = [None for _ in entries]
    best_hit_time_error = [float("inf") for _ in entries]
    initial_root_local = default_root[:, :3] - env_origins
    finite_state = torch.ones(num_envs, dtype=torch.bool, device=device)

    for control_step in range(total_control_steps):
        source_time = min(control_step * control_dt, motion_duration)
        reference_time = source_time if args_cli.reference_mode == "motion" else 0.0
        phase = (
            "motion" if args_cli.reference_mode == "motion" and control_step < replay_control_steps
            else "post_hold" if args_cli.reference_mode == "motion"
            else "hold_frame0"
        )
        q_reference = robot.data.default_joint_pos.clone()
        for env_index, data in enumerate(arrays):
            q_reference[env_index, joint_ids] = torch.as_tensor(
                _interpolate(data["joint_pos"], source_fps, reference_time), device=device
            )
        robot.set_joint_position_target(q_reference)
        robot.set_joint_velocity_target(torch.zeros_like(q_reference))

        for substep in range(decimation):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(physics_dt)
            elapsed_s = control_step * control_dt + (substep + 1) * physics_dt

            racket_pos_w, racket_vel_w, racket_quat_w, _ = racket_spatial_state_w(
                fake_env,
                SceneEntityCfg("robot"),
                racket_body_name="__force_wrist_offset_racket_fk__",
            )
            racket_normal_w = matrix_from_quat(racket_quat_w)[:, :, 1]
            root_pos_local = robot.data.root_pos_w - env_origins
            root_quat = robot.data.root_quat_w
            tilt = _tilt_deg(root_quat)
            actual_q = robot.data.joint_pos
            actual_qd = robot.data.joint_vel
            soft_limits = robot.data.soft_joint_pos_limits
            hard_limits = robot.data.joint_pos_limits
            actual_soft_margin = torch.minimum(
                actual_q - soft_limits[:, :, 0], soft_limits[:, :, 1] - actual_q
            )
            actual_hard_margin = torch.minimum(
                actual_q - hard_limits[:, :, 0], hard_limits[:, :, 1] - actual_q
            )
            reference_soft_margin = torch.minimum(
                q_reference - soft_limits[:, :, 0], soft_limits[:, :, 1] - q_reference
            )
            reference_hard_margin = torch.minimum(
                q_reference - hard_limits[:, :, 0], hard_limits[:, :, 1] - q_reference
            )
            force_norm = torch.linalg.vector_norm(contacts.data.net_forces_w, dim=-1)
            foot_force = force_norm[:, sensor_foot_body_ids]
            non_foot_force = force_norm[:, non_foot_sensor_ids]
            foot_speed = torch.linalg.vector_norm(
                robot.data.body_lin_vel_w[:, robot_foot_body_ids, :2], dim=-1
            )
            loaded_foot_speed = torch.where(
                foot_force >= float(args_cli.loaded_foot_force_n), foot_speed, torch.zeros_like(foot_speed)
            )
            torque_ratio = torch.abs(robot.data.applied_torque) / torch.clamp(
                robot.data.joint_effort_limits, min=1.0e-6
            )
            velocity_ratio = torch.abs(actual_qd) / torch.clamp(
                robot.data.joint_vel_limits, min=1.0e-6
            )
            finite_now = (
                torch.isfinite(root_pos_local).all(dim=-1)
                & torch.isfinite(root_quat).all(dim=-1)
                & torch.isfinite(actual_q).all(dim=-1)
                & torch.isfinite(actual_qd).all(dim=-1)
            )
            finite_state &= finite_now

            for env_index, entry in enumerate(entries):
                min_actual_soft_index = int(torch.argmin(actual_soft_margin[env_index]).cpu())
                min_actual_hard_index = int(torch.argmin(actual_hard_margin[env_index]).cpu())
                min_reference_soft_index = int(torch.argmin(reference_soft_margin[env_index]).cpu())
                min_reference_hard_index = int(torch.argmin(reference_hard_margin[env_index]).cpu())
                max_non_foot_index = int(torch.argmax(non_foot_force[env_index]).cpu())
                sample = {
                    "elapsed_s": elapsed_s,
                    "control_step": control_step,
                    "physics_substep": substep,
                    "phase": phase,
                    "source_time_s": source_time,
                    "root_position_w_m": _vec(root_pos_local[env_index]),
                    "root_quat_wxyz": _vec(root_quat[env_index]),
                    "root_linear_velocity_w_mps": _vec(robot.data.root_lin_vel_w[env_index]),
                    "root_angular_velocity_w_radps": _vec(robot.data.root_ang_vel_w[env_index]),
                    "root_tilt_deg": float(tilt[env_index].cpu()),
                    "joint_position_rad": _vec(actual_q[env_index]),
                    "joint_velocity_radps": _vec(actual_qd[env_index]),
                    "joint_position_target_rad": _vec(q_reference[env_index]),
                    "joint_tracking_error_rad": _vec(actual_q[env_index] - q_reference[env_index]),
                    "minimum_actual_soft_joint_margin_rad": float(
                        actual_soft_margin[env_index, min_actual_soft_index].cpu()
                    ),
                    "minimum_actual_soft_joint_margin_joint": robot.joint_names[min_actual_soft_index],
                    "minimum_actual_hard_joint_margin_rad": float(
                        actual_hard_margin[env_index, min_actual_hard_index].cpu()
                    ),
                    "minimum_actual_hard_joint_margin_joint": robot.joint_names[min_actual_hard_index],
                    "minimum_reference_soft_joint_margin_rad": float(
                        reference_soft_margin[env_index, min_reference_soft_index].cpu()
                    ),
                    "minimum_reference_soft_joint_margin_joint": robot.joint_names[min_reference_soft_index],
                    "minimum_reference_hard_joint_margin_rad": float(
                        reference_hard_margin[env_index, min_reference_hard_index].cpu()
                    ),
                    "minimum_reference_hard_joint_margin_joint": robot.joint_names[min_reference_hard_index],
                    "racket_link_position_w_m": _vec(racket_pos_w[env_index] - env_origins[env_index]),
                    "racket_link_velocity_w_mps": _vec(racket_vel_w[env_index]),
                    "racket_normal_w": _vec(racket_normal_w[env_index]),
                    "foot_contact_force_n": _vec(foot_force[env_index]),
                    "foot_position_w_m": _vec(
                        robot.data.body_pos_w[env_index, robot_foot_body_ids] - env_origins[env_index]
                    ),
                    "foot_tangential_speed_mps": _vec(foot_speed[env_index]),
                    "loaded_foot_tangential_speed_mps": _vec(loaded_foot_speed[env_index]),
                    "max_non_foot_contact_force_n": float(non_foot_force[env_index].max().cpu()),
                    "max_non_foot_contact_body": contacts.body_names[
                        int(non_foot_sensor_ids[max_non_foot_index].cpu())
                    ],
                    "applied_torque_nm": _vec(robot.data.applied_torque[env_index]),
                    "max_effort_limit_ratio": float(torque_ratio[env_index].max().cpu()),
                    "max_velocity_limit_ratio": float(velocity_ratio[env_index].max().cpu()),
                    "finite_state": bool(finite_now[env_index].cpu()),
                }
                traces[env_index].append(sample)

                time_error = abs(elapsed_s - hit_times[env_index])
                if time_error < best_hit_time_error[env_index]:
                    best_hit_time_error[env_index] = time_error
                    target = entry["strike_target_b0"]
                    target_pos = p1_anchor_local[env_index] + torch.tensor(
                        target["racket_position_b0_m"], device=device
                    )
                    target_vel = torch.tensor(target["racket_velocity_b0_mps"], device=device)
                    target_normal = torch.tensor(target["racket_normal_b0"], device=device)
                    hit_samples[env_index] = {
                        "sample_elapsed_s": elapsed_s,
                        "time_error_s": time_error,
                        "target_position_w_m": _vec(target_pos),
                        "actual_position_w_m": _vec(racket_pos_w[env_index] - env_origins[env_index]),
                        "position_error_m": float(
                            torch.linalg.vector_norm(
                                racket_pos_w[env_index] - env_origins[env_index] - target_pos
                            ).cpu()
                        ),
                        "target_velocity_w_mps": _vec(target_vel),
                        "actual_velocity_w_mps": _vec(racket_vel_w[env_index]),
                        "velocity_error_mps": float(
                            torch.linalg.vector_norm(racket_vel_w[env_index] - target_vel).cpu()
                        ),
                        "target_normal_w": _vec(target_normal),
                        "actual_normal_w": _vec(racket_normal_w[env_index]),
                        "normal_error_deg": float(
                            _angle_deg(
                                racket_normal_w[env_index].unsqueeze(0),
                                target_normal.unsqueeze(0),
                            )[0].cpu()
                        ),
                    }

    self_collision_observable = bool(
        getattr(scene_cfg.robot.spawn.rigid_props, "enabled_self_collisions", False)
    )
    results = []
    for env_index, entry in enumerate(entries):
        trace = traces[env_index]
        hit = hit_samples[env_index]
        assert hit is not None
        metrics = {
            "finite_state": bool(finite_state[env_index].cpu()),
            "minimum_root_height_w_m": min(sample["root_position_w_m"][2] for sample in trace),
            "max_root_tilt_deg": max(sample["root_tilt_deg"] for sample in trace),
            "max_root_displacement_m": max(
                float(
                    np.linalg.norm(
                        np.asarray(sample["root_position_w_m"])
                        - initial_root_local[env_index].detach().cpu().numpy()
                    )
                )
                for sample in trace
            ),
            "minimum_actual_soft_joint_margin_rad": min(
                sample["minimum_actual_soft_joint_margin_rad"] for sample in trace
            ),
            "minimum_actual_hard_joint_margin_rad": min(
                sample["minimum_actual_hard_joint_margin_rad"] for sample in trace
            ),
            "minimum_reference_soft_joint_margin_rad": min(
                sample["minimum_reference_soft_joint_margin_rad"] for sample in trace
            ),
            "minimum_reference_hard_joint_margin_rad": min(
                sample["minimum_reference_hard_joint_margin_rad"] for sample in trace
            ),
            "max_loaded_foot_tangential_speed_mps": max(
                max(sample["loaded_foot_tangential_speed_mps"]) for sample in trace
            ),
            "max_non_foot_contact_force_n": max(
                sample["max_non_foot_contact_force_n"] for sample in trace
            ),
            "max_effort_limit_ratio": max(sample["max_effort_limit_ratio"] for sample in trace),
            "max_velocity_limit_ratio": max(sample["max_velocity_limit_ratio"] for sample in trace),
            "max_joint_tracking_error_rad": max(
                max(abs(value) for value in sample["joint_tracking_error_rad"]) for sample in trace
            ),
            "hit_position_error_m": hit["position_error_m"],
            "hit_normal_error_deg": hit["normal_error_deg"],
            "hit_velocity_error_mps": hit["velocity_error_mps"],
            "loaded_foot_force_n": float(args_cli.loaded_foot_force_n),
            "non_foot_contact_force_n": float(args_cli.non_foot_contact_force_n),
            "dangerous_contact_force_n": float(args_cli.dangerous_contact_force_n),
            "max_loaded_foot_slip_mps": float(args_cli.max_loaded_foot_slip_mps),
            "max_root_tilt_allowed_deg": float(args_cli.max_root_tilt_deg),
        }
        if args_cli.reference_mode == "motion":
            category, reasons = _classification(metrics, self_collision_observable)
        else:
            category, reasons = "BASELINE", ["frame0_hold_control"]
        results.append(
            {
                "motion_id": int(entry["motion_id"]),
                "episode_id": str(entry["episode_id"]),
                "classification": category,
                "classification_reasons": reasons,
                "hit": hit,
                "metrics": metrics,
                "trace": trace,
            }
        )

    return {
        "purpose": "P4A formal P1 full-trajectory dynamic qualification replay",
        "canonical_manifest": str(manifest_path),
        "canonical_contract_version": manifest["contract_version"],
        "motion_ids": requested_ids,
        "training_started": False,
        "policy_actions_applied": False,
        "reference_mode": args_cli.reference_mode,
        "controller": "formal A3 implicit PD, absolute joint position reference, zero velocity target",
        "scene": "formal HOPE P1 floor/table/net, floating A3 base, ball parked out of play",
        "physics_frequency_hz": 1.0 / physics_dt,
        "control_frequency_hz": 1.0 / control_dt,
        "source_motion_frequency_hz": source_fps,
        "resampling": "linear joint-position interpolation from 50 Hz to formal 90 Hz control",
        "motion_duration_s": motion_duration,
        "post_hold_s": float(args_cli.post_hold_s),
        "self_collision_observable": self_collision_observable,
        "self_collision_note": (
            "The formal A3 asset currently has enabled_self_collisions=false; no motion can receive "
            "an unconditional A qualification until a separate collision-distance/self-contact audit passes."
        ),
        "joint_names": list(robot.joint_names),
        "contact_body_names": list(contacts.body_names),
        "classification_contract": {
            "A": "all configured gates pass, including observable self collision",
            "B": "shape/trajectory dynamically usable but requires deterministic repair or missing safety gate",
            "C": "shape only; material dynamic reconstruction required",
            "D": "unsafe; exclude from teacher library",
        },
        "results": results,
    }


def main() -> None:
    try:
        report = _run()
        output = args_cli.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(output)
        print(output, flush=True)
    finally:
        if os.name == "posix":
            signal.signal(signal.SIGALRM, lambda *_: os._exit(0))
            signal.alarm(10)
        simulation_app.close(wait_for_replicator=False)
        if os.name == "posix":
            signal.alarm(0)


if __name__ == "__main__":
    main()
