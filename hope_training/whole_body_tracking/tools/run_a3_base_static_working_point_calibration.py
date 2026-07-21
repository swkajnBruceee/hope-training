#!/usr/bin/env python3
"""Derive and validate a static A3 Base target preload without PPO.

The first rollout uses production PD_STAND gains only as a diagnostic reference.
Its steady torque is converted to an equivalent position-target preload under
the normal policy gains.  The second rollout restores normal gains and applies
that constant 14-DOF preload.  No gain, action, or deployment contract is
promoted by this tool.
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
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--robot-asset-path", type=Path)
parser.add_argument("--reference-steps", type=int, default=500)
parser.add_argument("--settle-window-steps", type=int, default=100)
parser.add_argument("--validation-steps", type=int, default=500)
parser.add_argument("--raw-action-clip-abs", type=float, default=0.25)
parser.add_argument(
    "--diagnostic-validation-clip-abs",
    type=float,
    default=None,
    help="Diagnostic-only validation bound; may be wider than the frozen v1 bound but cannot promote it.",
)
parser.add_argument("--trace-stride", type=int, default=10)
parser.add_argument(
    "--passive-gain-ablation",
    action="store_true",
    help="Run diagnostic zero-action gain-group profiles after preload validation.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if min(args_cli.reference_steps, args_cli.settle_window_steps, args_cli.validation_steps) < 1:
    parser.error("step counts must be positive")
if args_cli.settle_window_steps > args_cli.reference_steps:
    parser.error("--settle-window-steps cannot exceed --reference-steps")
if not 0.0 < args_cli.raw_action_clip_abs <= 0.25:
    parser.error("this qualification may not exceed the v1 raw action clip of 0.25")
if args_cli.diagnostic_validation_clip_abs is None:
    args_cli.diagnostic_validation_clip_abs = args_cli.raw_action_clip_abs
if not args_cli.raw_action_clip_abs <= args_cli.diagnostic_validation_clip_abs <= 1.0:
    parser.error("--diagnostic-validation-clip-abs must be between the v1 bound and 1.0")
if args_cli.trace_stride < 1:
    parser.error("--trace-stride must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

import training.tasks.base_locomotion.config.a3  # noqa: F401
from training.robots.agibot_a3 import A3_ANCHOR_BODY, A3_BASE_ACTION_JOINTS, A3_FEET_BODIES


def _configure_pd_stand_reference(cfg) -> None:
    """Apply checked-in production PD_STAND gains to the diagnostic reference only."""
    legs = cfg.scene.robot.actuators["legs"]
    legs.stiffness = {
        ".*_hip_pitch_joint": 1500.0,
        ".*_hip_roll_joint": 400.0,
        ".*_hip_yaw_joint": 300.0,
        ".*_knee_joint": 2000.0,
    }
    legs.damping = {
        ".*_hip_pitch_joint": 8.0,
        ".*_hip_roll_joint": 7.0,
        ".*_hip_yaw_joint": 7.0,
        ".*_knee_joint": 8.0,
    }
    feet = cfg.scene.robot.actuators["feet"]
    feet.stiffness = 500.0
    feet.damping = 5.0
    waist = cfg.scene.robot.actuators["waist"]
    waist.stiffness = {
        "waist_yaw_joint": 400.0,
        "waist_roll_joint": 500.0,
        "waist_pitch_joint": 500.0,
    }
    waist.damping = {
        "waist_yaw_joint": 4.0,
        "waist_roll_joint": 4.0,
        "waist_pitch_joint": 4.0,
    }
    arms = cfg.scene.robot.actuators["arms"]
    arms.stiffness = {
        ".*_shoulder_pitch_joint": 200.0,
        ".*_shoulder_roll_joint": 200.0,
        ".*_shoulder_yaw_joint": 100.0,
        ".*_elbow_joint": 200.0,
        ".*_wrist_roll_joint": 100.0,
        ".*_wrist_pitch_joint": 50.0,
        ".*_wrist_yaw_joint": 50.0,
    }
    arms.damping = {
        ".*_shoulder_pitch_joint": 2.0,
        ".*_shoulder_roll_joint": 2.0,
        ".*_shoulder_yaw_joint": 1.0,
        ".*_elbow_joint": 1.0,
        ".*_wrist_roll_joint": 1.0,
        ".*_wrist_pitch_joint": 1.0,
        ".*_wrist_yaw_joint": 1.0,
    }


def _make_env():
    cfg = gym.spec("A3BaseStand-v0").kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = 1
    cfg.seed = 0
    cfg.sim.device = args_cli.device
    cfg.actions.base.raw_clip = args_cli.diagnostic_validation_clip_abs
    asset_metadata = None
    if args_cli.robot_asset_path is not None:
        path = args_cli.robot_asset_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        cfg.scene.robot.spawn.asset_path = str(path)
        cfg.scene.robot.spawn.force_usd_conversion = True
        cfg.scene.robot.spawn.usd_dir = str(
            args_cli.output.expanduser().resolve().parent / "static_working_point_usd"
        )
        asset_metadata = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    env = gym.make("A3BaseStand-v0", cfg=cfg)
    env.reset(seed=0)
    return env, asset_metadata


def _pd_stand_gain_vectors(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the checked-in PD_STAND gains in articulation joint order."""
    cfg = gym.spec("A3BaseStand-v0").kwargs["env_cfg_entry_point"]()
    _configure_pd_stand_reference(cfg)
    gain_groups = cfg.scene.robot.actuators
    kp = robot.data.joint_stiffness.clone()
    kd = robot.data.joint_damping.clone()
    for actuator_name, actuator in robot.actuators.items():
        actuator_cfg = gain_groups[actuator_name]
        joint_names = [robot.joint_names[index] for index in actuator.joint_indices]
        group_kp = actuator._parse_joint_parameter(actuator_cfg.stiffness, kp[:, actuator.joint_indices])
        group_kd = actuator._parse_joint_parameter(actuator_cfg.damping, kd[:, actuator.joint_indices])
        if group_kp.shape[-1] != len(joint_names) or group_kd.shape[-1] != len(joint_names):
            raise RuntimeError(f"PD_STAND gain resolution failed for actuator {actuator_name}")
        kp[:, actuator.joint_indices] = group_kp
        kd[:, actuator.joint_indices] = group_kd
    return kp, kd


def _write_gains(robot, kp: torch.Tensor, kd: torch.Tensor) -> None:
    """Update both PhysX drives and implicit-actuator diagnostic buffers."""
    robot.write_joint_stiffness_to_sim(kp)
    robot.write_joint_damping_to_sim(kd)
    for actuator in robot.actuators.values():
        actuator.stiffness[:] = kp[:, actuator.joint_indices]
        actuator.damping[:] = kd[:, actuator.joint_indices]


def _gain_profile(
    robot,
    normal_kp: torch.Tensor,
    normal_kd: torch.Tensor,
    pd_stand_kp: torch.Tensor,
    pd_stand_kd: torch.Tensor,
    profile: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve one fail-closed passive-stability gain ablation profile."""
    selectors = {
        "normal_all": lambda name: False,
        "pd_waist_pitch_only": lambda name: name == "waist_pitch_joint",
        "pd_waist": lambda name: name.startswith("waist_"),
        "pd_waist_ankles": lambda name: name.startswith("waist_") or "ankle_" in name,
        "pd_base14": lambda name: name in A3_BASE_ACTION_JOINTS,
        "pd_all": lambda name: True,
    }
    if profile not in selectors:
        raise ValueError(f"Unknown gain profile: {profile}")
    mask = torch.tensor(
        [selectors[profile](name) for name in robot.joint_names],
        dtype=torch.bool,
        device=robot.device,
    ).unsqueeze(0)
    return torch.where(mask, pd_stand_kp, normal_kp), torch.where(mask, pd_stand_kd, normal_kd)


def _tilt_rad(gravity_b: torch.Tensor) -> float:
    return float(torch.acos(torch.clamp(-gravity_b[2], min=-1.0, max=1.0)).item())


def _rollout(env, action: torch.Tensor, steps: int, phase: str) -> tuple[dict, dict[str, torch.Tensor]]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    sensor = unwrapped.scene.sensors["contact_forces"]
    action_term = unwrapped.action_manager.get_term("base")
    base_ids, base_names = robot.find_joints(A3_BASE_ACTION_JOINTS, preserve_order=True)
    if base_names != A3_BASE_ACTION_JOINTS:
        raise RuntimeError(f"Base joint mapping changed: {base_names}")
    foot_ids, foot_names = robot.find_bodies(A3_FEET_BODIES, preserve_order=True)
    if foot_names != A3_FEET_BODIES:
        raise RuntimeError(f"Foot mapping changed: {foot_names}")
    sensor_foot_ids = [sensor.body_names.index(name) for name in A3_FEET_BODIES]
    torso_ids, torso_names = robot.find_bodies([A3_ANCHOR_BODY], preserve_order=True)
    if torso_names != [A3_ANCHOR_BODY]:
        raise RuntimeError("Torso mapping changed")

    trace = []
    buffers = {
        "q_rad": [],
        "dq_rad_s": [],
        "applied_torque_nm": [],
        "hard_limit_margin_rad": [],
        "foot_vertical_load_fraction": [],
        "root_height_m": [],
        "root_tilt_rad": [],
        "torso_tilt_rad": [],
    }
    termination_labels: list[str] = []
    runtime_finite_tensor = torch.ones((), dtype=torch.bool, device=unwrapped.device)
    for step in range(steps):
        gravity_w = robot.data.GRAVITY_VEC_W
        if gravity_w.ndim > 1:
            gravity_w = gravity_w[0]
        torso_gravity = math_utils.quat_rotate_inverse(
            robot.data.body_quat_w[0, torso_ids[0]], gravity_w
        )
        foot_fz = torch.clamp(sensor.data.net_forces_w[0, sensor_foot_ids, 2], min=0.0)
        q = robot.data.joint_pos[0, base_ids]
        dq = robot.data.joint_vel[0, base_ids]
        torque = robot.data.applied_torque[0, base_ids]
        hard_limits = robot.data.joint_pos_limits[0, base_ids]
        hard_margin = torch.minimum(q - hard_limits[:, 0], hard_limits[:, 1] - q)
        load_fraction = foot_fz / torch.clamp(foot_fz.sum(), min=1.0e-6)
        root_tilt = torch.acos(
            torch.clamp(-robot.data.projected_gravity_b[0, 2], min=-1.0, max=1.0)
        )
        torso_tilt = torch.acos(torch.clamp(-torso_gravity[2], min=-1.0, max=1.0))
        buffers["q_rad"].append(q.clone())
        buffers["dq_rad_s"].append(dq.clone())
        buffers["applied_torque_nm"].append(torque.clone())
        buffers["hard_limit_margin_rad"].append(hard_margin.clone())
        buffers["foot_vertical_load_fraction"].append(load_fraction.clone())
        buffers["root_height_m"].append(robot.data.root_pos_w[0, 2].clone())
        buffers["root_tilt_rad"].append(root_tilt.clone())
        buffers["torso_tilt_rad"].append(torso_tilt.clone())
        if step % args_cli.trace_stride == 0 or step == steps - 1:
            total_fz = float(foot_fz.sum().item())
            trace.append(
                {
                    "policy_step": step,
                    "time_s": step * float(unwrapped.step_dt),
                    "q_rad": q.tolist(),
                    "dq_rad_s": dq.tolist(),
                    "applied_torque_nm": torque.tolist(),
                    "hard_limit_margin_rad": hard_margin.tolist(),
                    "root_height_m": float(robot.data.root_pos_w[0, 2]),
                    "root_tilt_rad": float(root_tilt),
                    "torso_tilt_rad": float(torso_tilt),
                    "foot_vertical_load_fraction": (
                        load_fraction.tolist() if total_fz > 1.0e-6 else [None, None]
                    ),
                }
            )
        finite = (
            torch.isfinite(q).all()
            and torch.isfinite(dq).all()
            and torch.isfinite(torque).all()
            and torch.isfinite(robot.data.root_state_w).all()
        )
        runtime_finite_tensor &= finite
        _obs, _reward, terminated, truncated, _extras = env.step(action)
        if bool((terminated | truncated).item()):
            termination_labels = [
                name
                for name in unwrapped.termination_manager.active_terms
                if bool(unwrapped.termination_manager.get_term(name)[0])
            ]
            break
        if (step + 1) % 100 == 0:
            print(f"[static-working-point] {phase}: {step + 1}/{steps} steps", flush=True)

    non_timeout = [name for name in termination_labels if name != "time_out"]
    report = {
        "phase": phase,
        "requested_steps": steps,
        "recorded_steps": len(buffers["q_rad"]),
        "trace_samples": len(trace),
        "termination_labels": termination_labels,
        "non_timeout_failure": bool(non_timeout),
        "runtime_integrity_passed": bool(runtime_finite_tensor.item()),
        "effective_raw_action": action_term.raw_actions[0].tolist(),
        "effective_target_rad": action_term.processed_actions[0].tolist(),
        "joint_names": A3_BASE_ACTION_JOINTS,
        "trace": trace,
    }
    stacked = {key: torch.stack(value) for key, value in buffers.items()}
    return report, stacked


def _window_tensor(buffers: dict[str, torch.Tensor], key: str, window: int) -> torch.Tensor:
    values = buffers[key]
    return values[-min(window, values.shape[0]) :].double().cpu()


def main() -> int:
    env_handle = None
    try:
        env_handle, asset_metadata = _make_env()
        env = env_handle.unwrapped
        robot = env.scene["robot"]
        normal_kp_all = robot.data.joint_stiffness.clone()
        normal_kd_all = robot.data.joint_damping.clone()
        pd_stand_kp, pd_stand_kd = _pd_stand_gain_vectors(robot)
        _write_gains(robot, pd_stand_kp, pd_stand_kd)
        env_handle.reset(seed=0)
        reference_action = torch.zeros((1, 14), device=env.device)
        reference, reference_buffers = _rollout(
            env_handle, reference_action, args_cli.reference_steps, "pd_stand_reference"
        )
        reference_stage_path = args_cli.output.with_suffix(".reference.json")
        reference_stage_path.parent.mkdir(parents=True, exist_ok=True)
        reference_stage_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage_id": "a3_base_static_working_point_reference_v1",
                    "robot_asset_override": asset_metadata,
                    "reference_rollout": reference,
                    "qualification_status": {
                        "reference_full_horizon": reference["recorded_steps"] == args_cli.reference_steps
                        and not reference["non_timeout_failure"],
                        "static_working_point_approved": False,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_gains(robot, normal_kp_all, normal_kd_all)
        env_handle.reset(seed=0)
        action_term = env.action_manager.get_term("base")
        base_ids, base_names = robot.find_joints(A3_BASE_ACTION_JOINTS, preserve_order=True)
        if base_names != A3_BASE_ACTION_JOINTS:
            raise RuntimeError(f"Base joint mapping changed: {base_names}")

        q_window = _window_tensor(reference_buffers, "q_rad", args_cli.settle_window_steps)
        dq_window = _window_tensor(reference_buffers, "dq_rad_s", args_cli.settle_window_steps)
        torque_window = _window_tensor(
            reference_buffers, "applied_torque_nm", args_cli.settle_window_steps
        )
        q_equilibrium = q_window.mean(dim=0)
        dq_equilibrium = dq_window.mean(dim=0)
        torque_required = torque_window.mean(dim=0)
        normal_kp = robot.data.joint_stiffness[0, base_ids].double().cpu()
        normal_kd = robot.data.joint_damping[0, base_ids].double().cpu()
        default_q = robot.data.default_joint_pos[0, base_ids].double().cpu()
        scale = action_term._scale[0].double().cpu()

        # tau = kp * (q_target - q) - kd * dq
        target_preload = q_equilibrium + (torque_required + normal_kd * dq_equilibrium) / normal_kp
        requested_raw = (target_preload - default_q) / scale
        clipped_raw = torch.clamp(
            requested_raw,
            -args_cli.diagnostic_validation_clip_abs,
            args_cli.diagnostic_validation_clip_abs,
        )
        validation_action = clipped_raw.float().unsqueeze(0).to(env.device)
        validation, validation_buffers = _rollout(
            env_handle, validation_action, args_cli.validation_steps, "normal_gain_validation"
        )

        passive_gain_ablation = None
        if args_cli.passive_gain_ablation:
            passive_gain_ablation = {}
            zero_action = torch.zeros((1, 14), device=env.device)
            for profile in (
                "normal_all",
                "pd_waist_pitch_only",
                "pd_waist",
                "pd_waist_ankles",
                "pd_base14",
                "pd_all",
            ):
                profile_kp, profile_kd = _gain_profile(
                    robot,
                    normal_kp_all,
                    normal_kd_all,
                    pd_stand_kp,
                    pd_stand_kd,
                    profile,
                )
                _write_gains(robot, profile_kp, profile_kd)
                env_handle.reset(seed=0)
                profile_report, _profile_buffers = _rollout(
                    env_handle,
                    zero_action,
                    args_cli.validation_steps,
                    f"passive_gain_ablation:{profile}",
                )
                passive_gain_ablation[profile] = profile_report

        mean_load = _window_tensor(
            reference_buffers, "foot_vertical_load_fraction", args_cli.settle_window_steps
        ).mean(dim=0).tolist()
        min_hard_margin = float(
            _window_tensor(
                reference_buffers, "hard_limit_margin_rad", args_cli.settle_window_steps
            ).min()
        )
        reference_steady = {
            "mean_abs_dq_rad_s": torch.abs(dq_window).mean(dim=0).tolist(),
            "max_mean_abs_dq_rad_s": float(torch.abs(dq_window).mean(dim=0).max()),
            "mean_foot_vertical_load_fraction": mean_load,
            "minimum_hard_limit_margin_rad": min_hard_margin,
            "last_root_height_m": float(reference_buffers["root_height_m"][-1]),
            "last_root_tilt_rad": float(reference_buffers["root_tilt_rad"][-1]),
            "last_torso_tilt_rad": float(reference_buffers["torso_tilt_rad"][-1]),
        }
        requested_within_clip = bool(
            torch.all(torch.abs(requested_raw) <= args_cli.raw_action_clip_abs + 1.0e-9)
        )
        reference_full_horizon = (
            reference["recorded_steps"] == args_cli.reference_steps
            and not reference["non_timeout_failure"]
        )
        validation_full_horizon = (
            validation["recorded_steps"] == args_cli.validation_steps
            and not validation["non_timeout_failure"]
        )
        static_working_point_approved = bool(
            reference_full_horizon
            and requested_within_clip
            and validation_full_horizon
            and reference["runtime_integrity_passed"]
            and validation["runtime_integrity_passed"]
            and reference_steady["max_mean_abs_dq_rad_s"] <= 0.05
            and reference_steady["minimum_hard_limit_margin_rad"] >= 0.05
            and min(mean_load) >= 0.2
        )

        result = {
            "schema_version": 1,
            "calibration_id": "a3_base_static_working_point_calibration_v1",
            "simulation_only": True,
            "task": "A3BaseStand-v0",
            "robot_asset_override": asset_metadata,
            "policy_dt_s": float(env.step_dt),
            "physics_dt_s": float(env.physics_dt),
            "base_joint_names": A3_BASE_ACTION_JOINTS,
            "semantics": {
                "physical_posture_reference": "mean q of the final PD_STAND diagnostic window",
                "normal_gain_target_preload": "q_equilibrium + (required_torque + kd*dq_equilibrium)/kp",
                "pd_stand_is_diagnostic_only": True,
                "normal_policy_gains_restored_for_validation": True,
                "action_clip_or_scale_expansion_authorized": False,
                "diagnostic_validation_clip_abs": args_cli.diagnostic_validation_clip_abs,
            },
            "normal_policy_kp_nm_per_rad": normal_kp.tolist(),
            "normal_policy_kd_nms_per_rad": normal_kd.tolist(),
            "candidate": {
                "q_posture_reference_rad": q_equilibrium.tolist(),
                "mean_dq_reference_rad_s": dq_equilibrium.tolist(),
                "required_torque_nm": torque_required.tolist(),
                "q_target_preload_rad": target_preload.tolist(),
                "target_offset_from_current_default_rad": (target_preload - default_q).tolist(),
                "requested_raw_action": requested_raw.tolist(),
                "clipped_raw_action_used_for_validation": clipped_raw.tolist(),
                "requested_raw_action_within_v1_clip": requested_within_clip,
                "requested_raw_action_within_diagnostic_validation_clip": bool(
                    torch.all(torch.abs(requested_raw) <= args_cli.diagnostic_validation_clip_abs + 1.0e-9)
                ),
                "max_abs_requested_raw_action": float(torch.abs(requested_raw).max()),
            },
            "reference_steady_window": reference_steady,
            "reference_rollout": reference,
            "validation_rollout": validation,
            "passive_gain_ablation": passive_gain_ablation,
            "qualification_status": {
                "reference_full_horizon": reference_full_horizon,
                "requested_target_preload_within_v1_action_contract": requested_within_clip,
                "validation_full_horizon": validation_full_horizon,
                "diagnostic_preload_validation_passed": validation_full_horizon
                and validation["runtime_integrity_passed"],
                "static_working_point_approved": static_working_point_approved,
                "additional_ppo_smoke_approved": False,
                "stand_long_training_approved": False,
                "deployment_approved": False,
            },
            "changes_authorized_by_this_report": [],
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        summary = {
            key: value
            for key, value in result.items()
            if key not in ("reference_rollout", "validation_rollout", "passive_gain_ablation")
        }
        print(json.dumps(summary, indent=2))
        return 0 if reference["runtime_integrity_passed"] and validation["runtime_integrity_passed"] else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env_handle is not None:
            env_handle.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
