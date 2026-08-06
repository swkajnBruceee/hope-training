#!/usr/bin/env python3
"""Audit joint and joint-group noise effects on a fixed clean Recovery-A trace."""

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
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--expected-trace-sha256", required=True)
parser.add_argument("--envelope-decision", type=Path, required=True)
parser.add_argument("--expected-envelope-decision-sha256", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--group-size", type=int, default=16)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--noise-std", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=20260719)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.group_size < 1:
    parser.error("--group-size must be positive")
if args_cli.steps < 1:
    parser.error("--steps must be positive")
if args_cli.noise_std <= 0.0 or not math.isfinite(args_cli.noise_std):
    parser.error("--noise-std must be finite and positive")
for option, value in (
    ("--expected-trace-sha256", args_cli.expected_trace_sha256),
    (
        "--expected-envelope-decision-sha256",
        args_cli.expected_envelope_decision_sha256,
    ),
):
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        parser.error(f"{option} must be a 64-character hexadecimal SHA-256")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
import training.tasks.base_locomotion.config.a3  # noqa: F401
from tools.analyze_a3_base_recovery_manual_review import episode_events, summarize_events
from training.robots.agibot_a3 import A3_BASE_ACTION_JOINTS
from training.tasks.base_locomotion.base_env_cfg import A3_NOMINAL_BODY_HEIGHT_M


GROUP_NAMES = (
    "passive_zero",
    "all_base14",
    "hip_knee_only",
    "ankle_only",
    "waist_only",
    "ankle_waist_frozen",
)
CORE_NAMES = (
    "abs_pelvis_roll_rad",
    "abs_pelvis_pitch_rad",
    "abs_root_angular_velocity_x_rad_s",
    "abs_root_angular_velocity_y_rad_s",
    "abs_root_linear_velocity_x_m_s",
    "abs_root_linear_velocity_y_m_s",
    "abs_base_height_error_m",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_group_masks(joint_names: list[str]) -> dict[str, np.ndarray]:
    """Build exact Base14 masks without depending on articulation-native order."""
    names = list(joint_names)
    hip_knee = np.asarray(
        [
            (("hip_" in name) or ("knee_" in name))
            and (name.startswith("left_") or name.startswith("right_"))
            for name in names
        ],
        dtype=bool,
    )
    ankle = np.asarray(
        [
            name
            in {
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            }
            for name in names
        ],
        dtype=bool,
    )
    waist = np.asarray(
        [name in {"waist_roll_joint", "waist_pitch_joint"} for name in names],
        dtype=bool,
    )
    masks = {
        "passive_zero": np.zeros(len(names), dtype=bool),
        "all_base14": np.ones(len(names), dtype=bool),
        "hip_knee_only": hip_knee,
        "ankle_only": ankle,
        "waist_only": waist,
        "ankle_waist_frozen": ~(ankle | waist),
    }
    if len(names) != 14:
        raise ValueError(f"Expected 14 action joints, got {len(names)}")
    if not np.array_equal(masks["ankle_waist_frozen"], masks["hip_knee_only"]):
        raise ValueError("ankle_waist_frozen mask must equal hip_knee_only")
    return masks


def _signal_metrics(values: np.ndarray) -> dict[str, float]:
    values64 = np.asarray(values, dtype=np.float64).reshape(-1)
    if values64.size == 0:
        return {"mean": None, "std": None, "rms": None, "peak_abs": None}
    return {
        "mean": float(values64.mean()),
        "std": float(values64.std()),
        "rms": float(np.sqrt(np.mean(np.square(values64)))),
        "peak_abs": float(np.max(np.abs(values64))),
    }


def _signed_clip_fractions(values: np.ndarray, clip: float) -> dict[str, float | int]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0:
        return {"count": 0, "positive": 0.0, "negative": 0.0, "two_sided": 0.0}
    positive = float(np.mean(flattened >= clip))
    negative = float(np.mean(flattened <= -clip))
    return {
        "count": int(flattened.size),
        "positive": positive,
        "negative": negative,
        "two_sided": positive + negative,
    }


def _outside_inside_action(
    absolute_action: np.ndarray, outside: np.ndarray, valid: np.ndarray
) -> dict[str, float | None]:
    """Describe association only; this statistic does not establish causation."""
    action = np.asarray(absolute_action, dtype=np.float64)
    outside_mask = np.asarray(outside, dtype=bool) & np.asarray(valid, dtype=bool)
    inside_mask = (~np.asarray(outside, dtype=bool)) & np.asarray(valid, dtype=bool)

    def summarize(mask: np.ndarray) -> tuple[float | None, float | None, int]:
        selected = action[mask]
        if not selected.size:
            return None, None, 0
        return (
            float(np.mean(selected)),
            float(np.sqrt(np.mean(np.square(selected)))),
            int(selected.size),
        )

    outside_mean, outside_rms, outside_count = summarize(outside_mask)
    inside_mean, inside_rms, inside_count = summarize(inside_mask)

    def ratio(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator == 0.0:
            return None
        return float(numerator / denominator)

    return {
        "outside_count": outside_count,
        "inside_count": inside_count,
        "outside_mean_abs_action": outside_mean,
        "inside_mean_abs_action": inside_mean,
        "outside_inside_mean_ratio": ratio(outside_mean, inside_mean),
        "outside_rms_action": outside_rms,
        "inside_rms_action": inside_rms,
        "outside_inside_rms_ratio": ratio(outside_rms, inside_rms),
    }


def _core_state(robot) -> dict[str, torch.Tensor]:
    gravity = robot.data.projected_gravity_b
    return {
        "abs_pelvis_roll_rad": torch.abs(
            torch.atan2(-gravity[:, 1], -gravity[:, 2])
        ),
        "abs_pelvis_pitch_rad": torch.abs(
            torch.asin(torch.clamp(gravity[:, 0], -1.0, 1.0))
        ),
        "abs_root_angular_velocity_x_rad_s": torch.abs(robot.data.root_ang_vel_b[:, 0]),
        "abs_root_angular_velocity_y_rad_s": torch.abs(robot.data.root_ang_vel_b[:, 1]),
        "abs_root_linear_velocity_x_m_s": torch.abs(robot.data.root_lin_vel_b[:, 0]),
        "abs_root_linear_velocity_y_m_s": torch.abs(robot.data.root_lin_vel_b[:, 1]),
        "abs_base_height_error_m": torch.abs(
            robot.data.root_pos_w[:, 2] - A3_NOMINAL_BODY_HEIGHT_M
        ),
    }


def _select_clean_trace(trace: dict[str, np.ndarray], count: int) -> np.ndarray:
    required = {"trace_index", "profile_id", "roll_pitch_rad", "angular_velocity_rad_s"}
    if not required.issubset(trace):
        raise ValueError(f"Trace is missing fields: {sorted(required - trace.keys())}")
    selected = np.flatnonzero(trace["profile_id"] == 0)[:count]
    if selected.size != count:
        raise ValueError(f"Clean trace has {selected.size} rows, requires {count}")
    if not np.all(trace["roll_pitch_rad"][selected] == 0.0):
        raise ValueError("Selected clean trace contains nonzero roll/pitch")
    if not np.all(trace["angular_velocity_rad_s"][selected] == 0.0):
        raise ValueError("Selected clean trace contains nonzero angular velocity")
    return selected


def _write_repeated_clean_trace(
    unwrapped,
    robot,
    trace: dict[str, np.ndarray],
    selected: np.ndarray,
) -> None:
    group_count = len(GROUP_NAMES)
    pose_one = trace["roll_pitch_rad"][selected]
    velocity_one = trace["angular_velocity_rad_s"][selected]
    pose = torch.as_tensor(
        np.concatenate([pose_one] * group_count),
        device=unwrapped.device,
        dtype=torch.float32,
    )
    velocity = torch.as_tensor(
        np.concatenate([velocity_one] * group_count),
        device=unwrapped.device,
        dtype=torch.float32,
    )
    total_envs = group_count * selected.size
    env_ids = torch.arange(total_envs, device=unwrapped.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += unwrapped.scene.env_origins
    delta = math_utils.quat_from_euler_xyz(
        pose[:, 0], pose[:, 1], torch.zeros(total_envs, device=unwrapped.device)
    )
    root_state[:, 3:7] = math_utils.quat_mul(root_state[:, 3:7], delta)
    root_state[:, 10:12] += velocity
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
        env_ids=env_ids,
    )
    unwrapped.recovery_disturbed_mask[:] = False
    unwrapped.recovery_initial_roll_pitch_rad[:] = pose
    unwrapped.recovery_initial_angular_velocity_rad_s[:] = velocity


def _grouped(value: torch.Tensor, group_size: int) -> torch.Tensor:
    return value.reshape(len(GROUP_NAMES), group_size, *value.shape[1:])


def _numpy(value: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(value.detach().to(device="cpu", dtype=torch.float32).numpy())


def _termination_counts(unwrapped, done: torch.Tensor, group_size: int) -> dict:
    result = {
        group: {name: 0 for name in unwrapped.termination_manager.active_terms}
        for group in GROUP_NAMES
    }
    for term_name in unwrapped.termination_manager.active_terms:
        term = _grouped(
            unwrapped.termination_manager.get_term(term_name) & done, group_size
        )
        for group_index, group_name in enumerate(GROUP_NAMES):
            result[group_name][term_name] += int(term[group_index].sum().item())
    return result


def _merge_termination_counts(target: dict, update: dict) -> None:
    for group_name in GROUP_NAMES:
        for term_name, count in update[group_name].items():
            target[group_name][term_name] += count


def _optional_joint_physics(robot, base_ids: torch.Tensor, joint_names: list[str]) -> dict:
    result = {}
    for output_name, attribute in (
        ("default_stiffness_nm_per_rad", "default_joint_stiffness"),
        ("default_damping_nm_s_per_rad", "default_joint_damping"),
        ("effort_limit_nm", "joint_effort_limits"),
    ):
        value = getattr(robot.data, attribute, None)
        if value is None:
            result[output_name] = None
            continue
        selected = value[0, base_ids] if value.ndim == 2 else value[base_ids]
        result[output_name] = {
            name: float(selected[index].item()) for index, name in enumerate(joint_names)
        }
    return result


def _validate_inputs() -> tuple[Path, str, Path, str, dict, dict[str, np.ndarray]]:
    trace_path = args_cli.trace.expanduser().resolve()
    decision_path = args_cli.envelope_decision.expanduser().resolve()
    trace_sha = _sha256(trace_path)
    decision_sha = _sha256(decision_path)
    if trace_sha.lower() != args_cli.expected_trace_sha256.lower():
        raise ValueError(
            f"Trace SHA-256 mismatch: expected={args_cli.expected_trace_sha256}, actual={trace_sha}"
        )
    if decision_sha.lower() != args_cli.expected_envelope_decision_sha256.lower():
        raise ValueError(
            "Envelope decision SHA-256 mismatch: "
            f"expected={args_cli.expected_envelope_decision_sha256}, actual={decision_sha}"
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    approved = decision.get("approved_envelope", {})
    if (
        decision.get("recovery_envelope_approved") is not True
        or approved.get("name") != "B_core_only"
        or decision.get("authorizes_ppo") is not False
    ):
        raise ValueError("Envelope decision is not the approved B_core_only no-training decision")
    with np.load(trace_path, allow_pickle=False) as payload:
        trace = {name: payload[name].copy() for name in payload.files}
    return trace_path, trace_sha, decision_path, decision_sha, decision, trace


def main() -> int:
    env = None
    try:
        trace_path, trace_sha, decision_path, decision_sha, decision, trace = _validate_inputs()
        selected = _select_clean_trace(trace, args_cli.group_size)
        selected_trace_index = trace["trace_index"][selected].astype(np.int32)
        joint_names = list(A3_BASE_ACTION_JOINTS)
        masks_np = _build_group_masks(joint_names)

        cfg = gym.spec("A3BaseStandRecoveryA-v0").kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = len(GROUP_NAMES) * args_cli.group_size
        cfg.seed = args_cli.seed
        cfg.sim.device = args_cli.device
        env = gym.make("A3BaseStandRecoveryA-v0", cfg=cfg)
        env.reset(seed=args_cli.seed)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        action_term = unwrapped.action_manager.get_term("base")
        _write_repeated_clean_trace(unwrapped, robot, trace, selected)

        base_ids_list, resolved_names = robot.find_joints(joint_names, preserve_order=True)
        joint_order_passed = bool(
            resolved_names == joint_names
            and list(action_term.cfg.base_joint_names) == joint_names
            and list(action_term._base_joint_ids) == list(base_ids_list)
        )
        if not joint_order_passed:
            raise ValueError(
                f"A3 Base action joint order mismatch: expected={joint_names}, resolved={resolved_names}"
            )
        base_ids = torch.tensor(base_ids_list, dtype=torch.long, device=unwrapped.device)
        group_masks = torch.as_tensor(
            np.stack([masks_np[name] for name in GROUP_NAMES]),
            device=unwrapped.device,
            dtype=torch.float32,
        )
        raw_clip = float(action_term.cfg.raw_clip)
        scale = action_term._scale.detach().flatten()
        default_q = robot.data.default_joint_pos[:, base_ids]
        total_envs = len(GROUP_NAMES) * args_cli.group_size
        generator = torch.Generator(device=unwrapped.device)
        generator.manual_seed(args_cli.seed)

        signal_names = (
            "sampled_raw_action",
            "effective_clipped_action",
            "effective_residual",
            "action_rate",
            "q_target_residual",
            "q_actual_deviation_from_default",
            "tracking_error",
            "applied_torque",
        )
        records: dict[str, list[torch.Tensor]] = {name: [] for name in signal_names}
        core_records: dict[str, list[torch.Tensor]] = {name: [] for name in CORE_NAMES}
        active_records: list[torch.Tensor] = []
        outside_records: list[torch.Tensor] = []
        active = torch.ones(total_envs, dtype=torch.bool, device=unwrapped.device)
        previous_effective = torch.zeros((total_envs, 14), device=unwrapped.device)
        termination_counts = {
            group: {name: 0 for name in unwrapped.termination_manager.active_terms}
            for group in GROUP_NAMES
        }
        runtime_finite = True
        completed_steps = 0
        approved_channels = decision["approved_envelope"]["channels"]
        enter_threshold = torch.tensor(
            [float(approved_channels[name]["enter_threshold"]) for name in CORE_NAMES],
            device=unwrapped.device,
        )

        for step in range(args_cli.steps):
            current_core = _core_state(robot)
            core_matrix = torch.stack([current_core[name] for name in CORE_NAMES], dim=-1)
            for name in CORE_NAMES:
                core_records[name].append(_grouped(current_core[name], args_cli.group_size).clone())
            active_records.append(_grouped(active, args_cli.group_size).clone())

            # One Base14 draw per trace row is shared by every group.  Masks are
            # the only command difference, so the synchronized comparison is attributable.
            base_sample = torch.randn(
                (args_cli.group_size, 14),
                generator=generator,
                device=unwrapped.device,
            ) * args_cli.noise_std
            sampled = base_sample.unsqueeze(0) * group_masks.unsqueeze(1)
            action = sampled.reshape(total_envs, 14)
            _obs, _reward, terminated, truncated, _extras = env.step(action)
            post_core = _core_state(robot)
            post_core_matrix = torch.stack(
                [post_core[name] for name in CORE_NAMES], dim=-1
            )
            post_outside = torch.any(
                post_core_matrix > enter_threshold.unsqueeze(0), dim=-1
            )
            outside_records.append(
                _grouped(post_outside, args_cli.group_size).clone()
            )

            effective = action_term.raw_actions.detach().clone()
            target = action_term.processed_actions.detach().clone()
            effective_residual = effective * scale.unsqueeze(0)
            q_target_residual = target - default_q
            actual_q = robot.data.joint_pos[:, base_ids]
            applied_torque = robot.data.applied_torque[:, base_ids]
            current = {
                "sampled_raw_action": action,
                "effective_clipped_action": effective,
                "effective_residual": effective_residual,
                "action_rate": effective - previous_effective,
                "q_target_residual": q_target_residual,
                "q_actual_deviation_from_default": actual_q - default_q,
                "tracking_error": target - actual_q,
                "applied_torque": applied_torque,
            }
            previous_effective.copy_(effective)
            for name, value in current.items():
                records[name].append(_grouped(value, args_cli.group_size).clone())
            done = (terminated | truncated) & active
            if done.any():
                _merge_termination_counts(
                    termination_counts,
                    _termination_counts(unwrapped, done, args_cli.group_size),
                )
                active[done] = False
            runtime_finite &= bool(
                torch.isfinite(core_matrix).all()
                and torch.isfinite(post_core_matrix).all()
                and all(torch.isfinite(value).all() for value in current.values())
                and torch.isfinite(robot.data.root_state_w).all()
                and torch.isfinite(robot.data.joint_vel).all()
            )
            completed_steps = step + 1
            if completed_steps == 1 or completed_steps % 25 == 0:
                print(
                    f"[joint-noise] step {completed_steps}/{args_cli.steps} "
                    f"active={int(active.sum().item())} finite={runtime_finite}",
                    flush=True,
                )

        stacked = {name: _numpy(torch.stack(values)) for name, values in records.items()}
        core_np = {name: _numpy(torch.stack(values)) for name, values in core_records.items()}
        active_np = _numpy(torch.stack(active_records)).astype(bool)
        outside_np = _numpy(torch.stack(outside_records)).astype(bool)

        group_metrics = {}
        ranking_rows = []
        dwell_s = float(decision["approved_envelope"]["dwell_s"])
        for group_index, group_name in enumerate(GROUP_NAMES):
            valid = active_np[:, group_index]
            group_core = {name: values[:, group_index] for name, values in core_np.items()}
            safety_terminations = sum(
                count
                for term_name, count in termination_counts[group_name].items()
                if term_name != "time_out"
            )
            events = episode_events(
                group_core,
                valid,
                approved_channels,
                float(unwrapped.step_dt),
                dwell_s=dwell_s,
            )
            recovery = summarize_events(events, safety_terminations)
            by_joint = {}
            for joint_index, joint_name in enumerate(joint_names):
                sampled_joint = stacked["sampled_raw_action"][
                    :, group_index, :, joint_index
                ]
                effective_joint = stacked["effective_clipped_action"][
                    :, group_index, :, joint_index
                ]
                joint_valid = valid
                metrics = {
                    "sampled_raw_action": _signal_metrics(sampled_joint[joint_valid]),
                    "sampled_raw_clip_fraction": _signed_clip_fractions(
                        sampled_joint[joint_valid], raw_clip
                    ),
                    "effective_clipped_action_rms": _signal_metrics(
                        effective_joint[joint_valid]
                    )["rms"],
                    "effective_residual_rms_rad": _signal_metrics(
                        stacked["effective_residual"][:, group_index, :, joint_index][joint_valid]
                    )["rms"],
                    "action_rate_rms_per_policy_step": _signal_metrics(
                        stacked["action_rate"][:, group_index, :, joint_index][joint_valid]
                    )["rms"],
                    "q_target_residual_rms_rad": _signal_metrics(
                        stacked["q_target_residual"][:, group_index, :, joint_index][joint_valid]
                    )["rms"],
                    "q_actual_deviation_from_default_rms_rad": _signal_metrics(
                        stacked["q_actual_deviation_from_default"][
                            :, group_index, :, joint_index
                        ][joint_valid]
                    )["rms"],
                    "tracking_error_rms_rad": _signal_metrics(
                        stacked["tracking_error"][:, group_index, :, joint_index][joint_valid]
                    )["rms"],
                    "applied_torque_rms_nm": _signal_metrics(
                        stacked["applied_torque"][:, group_index, :, joint_index][joint_valid]
                    )["rms"],
                    "applied_torque_peak_abs_nm": _signal_metrics(
                        stacked["applied_torque"][:, group_index, :, joint_index][joint_valid]
                    )["peak_abs"],
                    "effective_saturation_fraction": _signed_clip_fractions(
                        effective_joint[joint_valid], raw_clip - 1.0e-6
                    ),
                    "b_core_outside_vs_inside_action": _outside_inside_action(
                        np.abs(effective_joint),
                        outside_np[:, group_index],
                        joint_valid,
                    ),
                }
                by_joint[joint_name] = metrics
                if group_name == "all_base14":
                    ranking_rows.append(
                        {
                            "joint": joint_name,
                            "torque_rms_nm": metrics["applied_torque_rms_nm"],
                            "tracking_error_rms_rad": metrics["tracking_error_rms_rad"],
                            "outside_inside_action_rms_ratio": metrics[
                                "b_core_outside_vs_inside_action"
                            ]["outside_inside_rms_ratio"],
                        }
                    )
            core_max = {}
            for name, values in group_core.items():
                env_has_valid = np.any(valid, axis=0)
                valid_values = np.where(valid, values, -np.inf)
                per_environment_max = np.max(valid_values, axis=0)[env_has_valid]
                core_max[name] = {
                    "maximum": float(np.max(values[valid])) if np.any(valid) else None,
                    "per_environment_max": _signal_metrics(per_environment_max),
                }
            group_metrics[group_name] = {
                "mask": masks_np[group_name].astype(int).tolist(),
                "masked_joint_names": [
                    name for index, name in enumerate(joint_names) if masks_np[group_name][index]
                ],
                "redundant_control_group": group_name == "ankle_waist_frozen",
                "redundant_with": (
                    "hip_knee_only" if group_name == "ankle_waist_frozen" else None
                ),
                "by_joint": by_joint,
                "b_core_recovery": {
                    "transient_recovery_rate": recovery["transient_recovery_rate"],
                    "durable_recovery_rate": recovery["durable_recovery_rate"],
                    "final_1s_stable_rate": recovery["final_1s_stable_rate"],
                    "confirmed_recovery_time_s": recovery["recovery_time_s"],
                    "durable_confirmed_recovery_time_s": recovery[
                        "durable_recovery_time_s"
                    ],
                    "exit_cycle_count": recovery["exit_cycle_count"],
                },
                "termination_counts": termination_counts[group_name],
                "max_pelvis_root_core_state": core_max,
            }

        def ranked(metric: str) -> list[dict]:
            available = [row for row in ranking_rows if row[metric] is not None]
            return sorted(available, key=lambda row: row[metric], reverse=True)

        theoretical_two_sided = math.erfc(
            raw_clip / (args_cli.noise_std * math.sqrt(2.0))
        )
        all_sampled = stacked["sampled_raw_action"][:, GROUP_NAMES.index("all_base14")]
        distribution_validation = {
            "theoretical_two_sided_raw_clip_probability": theoretical_two_sided,
            "measured_all_base14": _signed_clip_fractions(all_sampled, raw_clip),
            "measured_all_base14_by_joint": {
                name: _signed_clip_fractions(all_sampled[..., index], raw_clip)
                for index, name in enumerate(joint_names)
            },
        }
        runtime_finite = bool(runtime_finite and completed_steps == args_cli.steps)
        result = {
            "schema_version": 1,
            "audit_id": "a3_base_recovery_joint_noise_physics_audit_v1",
            "task": "A3BaseStandRecoveryA-v0",
            "diagnostic_only": True,
            "simulation_only": True,
            "checkpoint_loaded": False,
            "training_run": False,
            "trace_path": str(trace_path),
            "trace_sha256": trace_sha,
            "trace_sha256_verified": True,
            "selected_clean_trace_index": selected_trace_index.tolist(),
            "selected_clean_trace_index_sha256": hashlib.sha256(
                selected_trace_index.tobytes()
            ).hexdigest(),
            "same_clean_trace_index_reused_by_all_groups": True,
            "envelope_decision_path": str(decision_path),
            "envelope_decision_sha256": decision_sha,
            "envelope_decision_sha256_verified": True,
            "approved_envelope_name": decision["approved_envelope"]["name"],
            "authorizes_ppo": False,
            "group_size": args_cli.group_size,
            "policy_steps": args_cli.steps,
            "noise_std": args_cli.noise_std,
            "generator_seed": args_cli.seed,
            "shared_base_normal_sample_across_groups": True,
            "independent_sample_each_policy_step": True,
            "zero_residual_baseline": True,
            "action_joint_order": joint_names,
            "joint_order_passed": joint_order_passed,
            "action_scale_rad": {
                name: float(scale[index].item()) for index, name in enumerate(joint_names)
            },
            "raw_clip_abs": raw_clip,
            "pd_and_physical_contract": _optional_joint_physics(
                robot, base_ids, joint_names
            ),
            "distribution_validation": distribution_validation,
            "groups": group_metrics,
            "mask_equivalence": {
                "ankle_waist_frozen_equals_hip_knee_only": bool(
                    np.array_equal(
                        masks_np["ankle_waist_frozen"], masks_np["hip_knee_only"]
                    )
                )
            },
            "sensitivity_ranking": {
                "scope": "all_base14",
                "association_not_causation": True,
                "by_applied_torque_rms": ranked("torque_rms_nm"),
                "by_tracking_error_rms": ranked("tracking_error_rms_rad"),
                "by_outside_inside_action_rms_ratio": ranked(
                    "outside_inside_action_rms_ratio"
                ),
            },
            "completed_policy_steps": completed_steps,
            "runtime_finite": runtime_finite,
            "audit_completed": runtime_finite,
            "recovery_training_approved": False,
            "ppo_approved": False,
            "deployment_approved": False,
        }
        output_text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args_cli.output.with_name(f".{args_cli.output.name}.tmp")
        temporary.write_text(output_text, encoding="utf-8")
        temporary.replace(args_cli.output)
        print(output_text, end="")
        return 0 if runtime_finite else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
