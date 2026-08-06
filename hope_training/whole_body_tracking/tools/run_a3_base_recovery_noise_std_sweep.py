#!/usr/bin/env python3
"""Diagnostic continuous-Gaussian Base14 std sweep on one fixed clean trace."""

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
parser.add_argument("--stds", default="0.15,0.10,0.075,0.05,0.025,0.0")
parser.add_argument("--seed", type=int, default=20260719)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _parse_stds(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as error:
        raise ValueError("--stds must be comma-separated numbers") from error
    if len(values) != 6:
        raise ValueError("--stds must contain exactly six groups")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("--stds must contain finite non-negative values")
    if len(set(values)) != len(values) or 0.0 not in values:
        raise ValueError("--stds must be unique and include 0.0")
    return values


try:
    STDS = _parse_stds(args_cli.stds)
except ValueError as error:
    parser.error(str(error))
if args_cli.group_size < 1:
    parser.error("--group-size must be positive")
if args_cli.steps < 1:
    parser.error("--steps must be positive")
for option, value in (
    ("--expected-trace-sha256", args_cli.expected_trace_sha256),
    ("--expected-envelope-decision-sha256", args_cli.expected_envelope_decision_sha256),
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


CORE_NAMES = (
    "abs_pelvis_roll_rad",
    "abs_pelvis_pitch_rad",
    "abs_root_angular_velocity_x_rad_s",
    "abs_root_angular_velocity_y_rad_s",
    "abs_root_linear_velocity_x_m_s",
    "abs_root_linear_velocity_y_m_s",
    "abs_base_height_error_m",
)
SIGNAL_NAMES = (
    "sampled_raw_action",
    "effective_clipped_action",
    "effective_residual",
    "action_rate",
    "q_target_residual",
    "q_actual_deviation_from_default",
    "tracking_error",
    "applied_torque",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rms_peak(values: np.ndarray) -> dict[str, float | None]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if not flattened.size:
        return {"rms": None, "peak_abs": None}
    return {
        "rms": float(np.sqrt(np.mean(np.square(flattened)))),
        "peak_abs": float(np.max(np.abs(flattened))),
    }


def _signed_fractions(values: np.ndarray, threshold: float) -> dict[str, float | int]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if not flattened.size:
        return {"count": 0, "positive": 0.0, "negative": 0.0, "two_sided": 0.0}
    positive = float(np.mean(flattened >= threshold))
    negative = float(np.mean(flattened <= -threshold))
    return {
        "count": int(flattened.size),
        "positive": positive,
        "negative": negative,
        "two_sided": positive + negative,
    }


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def _compatibility_heuristic(candidate: dict, baseline: dict) -> dict:
    """Non-gating diagnostic comparison against the std=0 passive-like group."""
    reasons = []
    checks = (
        ("transient_recovery_rate", 0.02),
        ("durable_recovery_rate", 0.05),
        ("final_1s_stable_rate", 0.05),
    )
    for name, allowance in checks:
        if candidate.get(name) is None or baseline.get(name) is None:
            reasons.append(f"{name}_unavailable")
        elif baseline[name] - candidate[name] > allowance + 1.0e-12:
            reasons.append(f"{name}_drop_exceeds_{allowance}")
    candidate_p90 = candidate.get("confirmed_recovery_p90_s")
    baseline_p90 = baseline.get("confirmed_recovery_p90_s")
    if candidate_p90 is None or baseline_p90 is None:
        reasons.append("confirmed_recovery_p90_unavailable")
    elif candidate_p90 - baseline_p90 > 1.0 + 1.0e-12:
        reasons.append("confirmed_recovery_p90_increase_exceeds_1s")
    if candidate.get("non_timeout_terminations", 0) > baseline.get("non_timeout_terminations", 0):
        reasons.append("new_non_timeout_termination")
    if candidate.get("runtime_finite") is not True:
        reasons.append("runtime_not_finite")
    return {"compatible": not reasons, "reasons": reasons}


def _monotonic_violations(rows: list[dict]) -> list[dict]:
    """Return review flags; these trends are deliberately not pass/fail gates."""
    ordered = sorted(rows, key=lambda row: row["std"], reverse=True)
    violations = []
    for high, low in zip(ordered, ordered[1:]):
        for metric in (
            "transient_recovery_rate",
            "durable_recovery_rate",
            "final_1s_stable_rate",
        ):
            if high[metric] is not None and low[metric] is not None and low[metric] + 1.0e-12 < high[metric]:
                violations.append(
                    {"metric": metric, "higher_std": high["std"], "lower_std": low["std"],
                     "expectation": "nondecreasing_as_std_decreases"}
                )
        for metric in ("overall_effective_action_rms", "overall_torque_rms_nm"):
            high_value, low_value = high[metric], low[metric]
            tolerance = max(1.0e-12, 0.05 * high_value)
            if low_value > high_value + tolerance:
                violations.append(
                    {"metric": metric, "higher_std": high["std"], "lower_std": low["std"],
                     "expectation": "broadly_decreasing_as_std_decreases", "relative_tolerance": 0.05}
                )
    return violations


def _core_state(robot) -> dict[str, torch.Tensor]:
    gravity = robot.data.projected_gravity_b
    return {
        "abs_pelvis_roll_rad": torch.abs(torch.atan2(-gravity[:, 1], -gravity[:, 2])),
        "abs_pelvis_pitch_rad": torch.abs(torch.asin(torch.clamp(gravity[:, 0], -1.0, 1.0))),
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


def _write_repeated_clean_trace(unwrapped, robot, trace, selected, group_count: int) -> None:
    pose = torch.as_tensor(
        np.concatenate([trace["roll_pitch_rad"][selected]] * group_count),
        device=unwrapped.device, dtype=torch.float32,
    )
    velocity = torch.as_tensor(
        np.concatenate([trace["angular_velocity_rad_s"][selected]] * group_count),
        device=unwrapped.device, dtype=torch.float32,
    )
    total = group_count * selected.size
    env_ids = torch.arange(total, device=unwrapped.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += unwrapped.scene.env_origins
    delta = math_utils.quat_from_euler_xyz(
        pose[:, 0], pose[:, 1], torch.zeros(total, device=unwrapped.device)
    )
    root_state[:, 3:7] = math_utils.quat_mul(root_state[:, 3:7], delta)
    root_state[:, 10:12] += velocity
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone(), env_ids=env_ids
    )
    unwrapped.recovery_disturbed_mask[:] = False
    unwrapped.recovery_initial_roll_pitch_rad[:] = pose
    unwrapped.recovery_initial_angular_velocity_rad_s[:] = velocity


def _grouped(value: torch.Tensor, group_count: int, group_size: int) -> torch.Tensor:
    return value.reshape(group_count, group_size, *value.shape[1:])


def _numpy(value: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(value.detach().to(device="cpu", dtype=torch.float32).numpy())


def _validate_inputs():
    trace_path = args_cli.trace.expanduser().resolve()
    decision_path = args_cli.envelope_decision.expanduser().resolve()
    trace_sha = _sha256(trace_path)
    decision_sha = _sha256(decision_path)
    if trace_sha.lower() != args_cli.expected_trace_sha256.lower():
        raise ValueError(f"Trace SHA-256 mismatch: expected={args_cli.expected_trace_sha256}, actual={trace_sha}")
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
        raise ValueError("Envelope decision is not the approved B_core_only no-PPO decision")
    with np.load(trace_path, allow_pickle=False) as payload:
        trace = {name: payload[name].copy() for name in payload.files}
    return trace_path, trace_sha, decision_path, decision_sha, decision, trace


def main() -> int:
    env = None
    try:
        trace_path, trace_sha, decision_path, decision_sha, decision, trace = _validate_inputs()
        group_count = len(STDS)
        selected = _select_clean_trace(trace, args_cli.group_size)
        selected_trace_index = trace["trace_index"][selected].astype(np.int32)
        joint_names = list(A3_BASE_ACTION_JOINTS)

        cfg = gym.spec("A3BaseStandRecoveryA-v0").kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = group_count * args_cli.group_size
        cfg.seed = args_cli.seed
        cfg.sim.device = args_cli.device
        env = gym.make("A3BaseStandRecoveryA-v0", cfg=cfg)
        env.reset(seed=args_cli.seed)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        action_term = unwrapped.action_manager.get_term("base")
        _write_repeated_clean_trace(unwrapped, robot, trace, selected, group_count)

        base_ids_list, resolved_names = robot.find_joints(joint_names, preserve_order=True)
        joint_order_passed = bool(
            resolved_names == joint_names
            and list(action_term.cfg.base_joint_names) == joint_names
            and list(action_term._base_joint_ids) == list(base_ids_list)
        )
        if not joint_order_passed:
            raise ValueError("A3 Base action joint order mismatch")
        base_ids = torch.tensor(base_ids_list, dtype=torch.long, device=unwrapped.device)
        raw_clip = float(action_term.cfg.raw_clip)
        scale = action_term._scale.detach().flatten()
        default_q = robot.data.default_joint_pos[:, base_ids]
        total_envs = group_count * args_cli.group_size
        std_tensor = torch.tensor(STDS, device=unwrapped.device, dtype=torch.float32)
        generator = torch.Generator(device=unwrapped.device)
        generator.manual_seed(args_cli.seed)

        records = {name: [] for name in SIGNAL_NAMES}
        core_records = {name: [] for name in CORE_NAMES}
        linear_velocity_records, angular_velocity_records, active_records = [], [], []
        active = torch.ones(total_envs, dtype=torch.bool, device=unwrapped.device)
        previous_effective = torch.zeros((total_envs, 14), device=unwrapped.device)
        termination_counts = [
            {name: 0 for name in unwrapped.termination_manager.active_terms}
            for _ in STDS
        ]
        runtime_finite = True
        runtime_finite_by_group = [True for _ in STDS]
        completed_steps = 0
        std0_index = STDS.index(0.0)
        std0_transport = {
            "sampled_command_exact_zero": True,
            "manager_raw_action_exact_zero": True,
            "scaled_residual_exact_zero": True,
        }

        for step in range(args_cli.steps):
            current_core = _core_state(robot)
            for name in CORE_NAMES:
                core_records[name].append(
                    _grouped(current_core[name], group_count, args_cli.group_size).clone()
                )
            linear_velocity_records.append(
                _grouped(robot.data.root_lin_vel_b, group_count, args_cli.group_size).clone()
            )
            angular_velocity_records.append(
                _grouped(robot.data.root_ang_vel_b, group_count, args_cli.group_size).clone()
            )
            active_records.append(_grouped(active, group_count, args_cli.group_size).clone())

            # Exactly one fresh standard-normal Base14 sample per trace row and step;
            # every std group receives that same sample multiplied by its own std.
            base_standard_normal_sample = torch.randn(
                (args_cli.group_size, 14), generator=generator, device=unwrapped.device
            )
            sampled = base_standard_normal_sample.unsqueeze(0) * std_tensor[:, None, None]
            sampled[std0_index].zero_()
            action = sampled.reshape(total_envs, 14)
            _obs, _reward, terminated, truncated, _extras = env.step(action)

            effective = action_term.raw_actions.detach().clone()
            target = action_term.processed_actions.detach().clone()
            effective_residual = effective * scale.unsqueeze(0)
            actual_q = robot.data.joint_pos[:, base_ids]
            current = {
                "sampled_raw_action": action,
                "effective_clipped_action": effective,
                "effective_residual": effective_residual,
                "action_rate": effective - previous_effective,
                "q_target_residual": target - default_q,
                "q_actual_deviation_from_default": actual_q - default_q,
                "tracking_error": target - actual_q,
                "applied_torque": robot.data.applied_torque[:, base_ids],
            }
            previous_effective.copy_(effective)
            for name, value in current.items():
                records[name].append(_grouped(value, group_count, args_cli.group_size).clone())

            std0_transport["sampled_command_exact_zero"] &= bool(
                torch.count_nonzero(sampled[std0_index]) == 0
            )
            std0_effective = _grouped(effective, group_count, args_cli.group_size)[std0_index]
            std0_residual = _grouped(effective_residual, group_count, args_cli.group_size)[std0_index]
            std0_transport["manager_raw_action_exact_zero"] &= bool(torch.count_nonzero(std0_effective) == 0)
            std0_transport["scaled_residual_exact_zero"] &= bool(torch.count_nonzero(std0_residual) == 0)

            done = (terminated | truncated) & active
            if done.any():
                grouped_done = _grouped(done, group_count, args_cli.group_size)
                for term_name in unwrapped.termination_manager.active_terms:
                    term = _grouped(
                        unwrapped.termination_manager.get_term(term_name) & done,
                        group_count, args_cli.group_size,
                    )
                    for group_index in range(group_count):
                        termination_counts[group_index][term_name] += int(term[group_index].sum().item())
                active[done] = False
            finite_per_env = torch.ones(total_envs, dtype=torch.bool, device=unwrapped.device)
            for value in (*current.values(), *current_core.values(), robot.data.root_state_w, robot.data.joint_vel):
                finite_per_env &= torch.isfinite(value).reshape(total_envs, -1).all(dim=1)
            grouped_finite = _grouped(
                finite_per_env, group_count, args_cli.group_size
            )
            for group_index in range(group_count):
                runtime_finite_by_group[group_index] &= bool(
                    grouped_finite[group_index].all()
                )
            runtime_finite &= bool(finite_per_env.all())
            completed_steps = step + 1
            if completed_steps == 1 or completed_steps % 25 == 0:
                print(
                    f"[noise-std-sweep] step {completed_steps}/{args_cli.steps} "
                    f"active={int(active.sum().item())} finite={runtime_finite}",
                    flush=True,
                )

        stacked = {name: _numpy(torch.stack(values)) for name, values in records.items()}
        core_np = {name: _numpy(torch.stack(values)) for name, values in core_records.items()}
        linear_velocity_np = _numpy(torch.stack(linear_velocity_records))
        angular_velocity_np = _numpy(torch.stack(angular_velocity_records))
        active_np = _numpy(torch.stack(active_records)).astype(bool)
        approved_channels = decision["approved_envelope"]["channels"]
        dwell_s = float(decision["approved_envelope"]["dwell_s"])

        groups = []
        comparison_rows = []
        for group_index, std in enumerate(STDS):
            valid = active_np[:, group_index]
            group_core = {name: values[:, group_index] for name, values in core_np.items()}
            non_timeout = sum(
                count for name, count in termination_counts[group_index].items() if name != "time_out"
            )
            events = episode_events(
                group_core, valid, approved_channels, float(unwrapped.step_dt), dwell_s=dwell_s
            )
            recovery = summarize_events(events, non_timeout)
            recovery_times = [event["recovery_time_s"] for event in events if event["recovered"]]
            durable_times = [
                event["durable_recovery_time_s"] for event in events if event["durable_recovery"]
            ]
            recovery_percentiles = _percentiles(recovery_times)
            durable_percentiles = _percentiles(durable_times)

            by_joint = {}
            for joint_index, joint_name in enumerate(joint_names):
                sampled_joint = stacked["sampled_raw_action"][:, group_index, :, joint_index][valid]
                effective_joint = stacked["effective_clipped_action"][:, group_index, :, joint_index][valid]
                by_joint[joint_name] = {
                    "sampled": _rms_peak(sampled_joint),
                    "sampled_clip_fraction": _signed_fractions(sampled_joint, raw_clip),
                    "effective": _rms_peak(effective_joint),
                    "effective_saturation_fraction": _signed_fractions(
                        effective_joint, raw_clip - 1.0e-6
                    ),
                    "residual": _rms_peak(
                        stacked["effective_residual"][:, group_index, :, joint_index][valid]
                    ),
                    "action_rate": _rms_peak(
                        stacked["action_rate"][:, group_index, :, joint_index][valid]
                    ),
                    "qtarget": _rms_peak(
                        stacked["q_target_residual"][:, group_index, :, joint_index][valid]
                    ),
                    "qactual": _rms_peak(
                        stacked["q_actual_deviation_from_default"][:, group_index, :, joint_index][valid]
                    ),
                    "tracking_error": _rms_peak(
                        stacked["tracking_error"][:, group_index, :, joint_index][valid]
                    ),
                    "torque_nm": _rms_peak(
                        stacked["applied_torque"][:, group_index, :, joint_index][valid]
                    ),
                }

            measured_clip = _signed_fractions(
                stacked["sampled_raw_action"][:, group_index][valid], raw_clip
            )
            theoretical_clip = (
                0.0 if std == 0.0 else math.erfc(raw_clip / (std * math.sqrt(2.0)))
            )
            overall_action = _rms_peak(
                stacked["effective_clipped_action"][:, group_index][valid]
            )
            overall_torque = _rms_peak(stacked["applied_torque"][:, group_index][valid])
            final_active_count = int(
                active.reshape(group_count, args_cli.group_size)[group_index]
                .sum()
                .item()
            )
            survived_count = (
                final_active_count
                + termination_counts[group_index].get("time_out", 0)
            )
            row = {
                "std": std,
                "survived_count": survived_count,
                "survival_rate": survived_count / args_cli.group_size,
                "termination_counts": termination_counts[group_index],
                "non_timeout_terminations": non_timeout,
                "b_core_recovery": {
                    "transient_recovery_rate": recovery["transient_recovery_rate"],
                    "durable_recovery_rate": recovery["durable_recovery_rate"],
                    "final_1s_stable_rate": recovery["final_1s_stable_rate"],
                    "confirmed_recovery_time_s": recovery_percentiles,
                    "durable_recovery_time_s": durable_percentiles,
                    "exit_cycle_count": recovery["exit_cycle_count"],
                },
                "max_pelvis_roll_rad": (
                    float(np.max(group_core["abs_pelvis_roll_rad"][valid])) if np.any(valid) else None
                ),
                "max_pelvis_pitch_rad": (
                    float(np.max(group_core["abs_pelvis_pitch_rad"][valid])) if np.any(valid) else None
                ),
                "root_linear_velocity_rms_m_s": _rms_peak(
                    linear_velocity_np[:, group_index][valid]
                )["rms"],
                "root_angular_velocity_rms_rad_s": _rms_peak(
                    angular_velocity_np[:, group_index][valid]
                )["rms"],
                "overall_effective_action_rms": overall_action["rms"],
                "overall_torque_rms_nm": overall_torque["rms"],
                "clip_validation": {
                    "theoretical_two_sided_probability_erfc": theoretical_clip,
                    "measured_overall": measured_clip,
                    "measured_by_joint": {
                        name: by_joint[name]["sampled_clip_fraction"] for name in joint_names
                    },
                },
                "by_joint": by_joint,
                "runtime_finite": runtime_finite_by_group[group_index],
            }
            groups.append(row)
            comparison_rows.append(
                {
                    "std": std,
                    "transient_recovery_rate": recovery["transient_recovery_rate"],
                    "durable_recovery_rate": recovery["durable_recovery_rate"],
                    "final_1s_stable_rate": recovery["final_1s_stable_rate"],
                    "confirmed_recovery_p90_s": recovery_percentiles["p90"],
                    "non_timeout_terminations": non_timeout,
                    "overall_effective_action_rms": overall_action["rms"],
                    "overall_torque_rms_nm": overall_torque["rms"],
                    "runtime_finite": runtime_finite_by_group[group_index],
                }
            )

        baseline = next(row for row in comparison_rows if row["std"] == 0.0)
        compatibility = []
        for row, group in zip(comparison_rows, groups):
            delta = {
                key: (
                    None if row[key] is None or baseline[key] is None else row[key] - baseline[key]
                )
                for key in (
                    "transient_recovery_rate",
                    "durable_recovery_rate",
                    "final_1s_stable_rate",
                    "confirmed_recovery_p90_s",
                    "non_timeout_terminations",
                    "overall_effective_action_rms",
                    "overall_torque_rms_nm",
                )
            }
            heuristic = _compatibility_heuristic(row, baseline)
            group["delta_vs_std0_passive_like"] = delta
            group["diagnostic_clean_compatibility_heuristic"] = heuristic
            compatibility.append((row["std"], heuristic["compatible"]))
        compatible_nonzero = [std for std, compatible in compatibility if std > 0.0 and compatible]
        violations = _monotonic_violations(comparison_rows)

        runtime_finite = bool(runtime_finite and completed_steps == args_cli.steps)
        std0_transport["passed"] = bool(all(std0_transport.values()))
        result = {
            "schema_version": 1,
            "audit_id": "a3_base_recovery_noise_std_sweep_v1",
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
            "same_clean_trace_index_reused_by_all_std_groups": True,
            "envelope_decision_path": str(decision_path),
            "envelope_decision_sha256": decision_sha,
            "envelope_decision_sha256_verified": True,
            "approved_envelope_name": "B_core_only",
            "authorizes_ppo": False,
            "group_size": args_cli.group_size,
            "policy_steps": args_cli.steps,
            "stds": list(STDS),
            "generator_seed": args_cli.seed,
            "shared_standard_normal_sample_across_std_groups": True,
            "independent_sample_each_policy_step": True,
            "continuous_gaussian_base14": True,
            "action_joint_order": joint_names,
            "joint_order_passed": joint_order_passed,
            "raw_clip_abs": raw_clip,
            "action_scale_rad": {
                name: float(scale[index].item()) for index, name in enumerate(joint_names)
            },
            "real_action_manager_clip_scale_pd": True,
            "std0_passive_like_transport": std0_transport,
            "groups": groups,
            "diagnostic_clean_compatibility_heuristic": {
                "is_gate": False,
                "thresholds": {
                    "transient_drop_max_pp": 2.0,
                    "durable_drop_max_pp": 5.0,
                    "final_1s_drop_max_pp": 5.0,
                    "confirmed_recovery_p90_increase_max_s": 1.0,
                    "new_non_timeout_terminations_allowed": 0,
                    "runtime_finite_required": True,
                },
                "highest_compatible_nonzero_std": max(compatible_nonzero) if compatible_nonzero else None,
                "does_not_approve_or_modify_training_std": True,
            },
            "monotonic_trend_review": {
                "is_gate": False,
                "expectations": [
                    "recovery rates nondecreasing as std decreases",
                    "effective action and torque RMS broadly decrease as std decreases",
                ],
                "violations": violations,
            },
            "completed_policy_steps": completed_steps,
            "runtime_finite": runtime_finite,
            "audit_completed": runtime_finite,
            "approval": False,
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
