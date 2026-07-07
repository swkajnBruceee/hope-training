"""Independent temporal repair layer for generic retarget initialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from analysis.mocap_cleaning.a3_metadata import A3_ACTIVE_JOINTS_FIRST_PASS, A3_POLICY_JOINT_ORDER


def _minimum_jerk(alpha: np.ndarray) -> np.ndarray:
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def _interpolate_segment(
    q0: np.ndarray,
    q1: np.ndarray,
    frames: int,
) -> np.ndarray:
    out = np.zeros((frames, q0.shape[0]), dtype=np.float64)
    if frames <= 1:
        out[0] = q0
        return out
    for i in range(frames):
        alpha = np.clip(i / float(frames - 1), 0.0, 1.0)
        alpha = _minimum_jerk(np.asarray(alpha))
        out[i] = (1.0 - alpha) * q0 + alpha * q1
    return out


def _interpolate_hermite_segment(
    q0: np.ndarray,
    q1: np.ndarray,
    frames: int,
    dt: float,
    v0: np.ndarray,
    v1: np.ndarray,
) -> np.ndarray:
    out = np.zeros((frames, q0.shape[0]), dtype=np.float64)
    if frames <= 1:
        out[0] = q0
        return out
    duration = max(float(frames - 1) * dt, 1e-9)
    for i in range(frames):
        s = np.clip(i / float(frames - 1), 0.0, 1.0)
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2
        out[i] = h00 * q0 + h10 * duration * v0 + h01 * q1 + h11 * duration * v1
    return out


def _joint_limits(active_joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    # Conservative dynamic-repair joint box; geometry initialization already
    # respected real URDF limits upstream.
    lower = []
    upper = []
    for name in active_joint_names:
        if name == "waist_yaw_joint":
            lower.append(-2.61799387799)
            upper.append(2.61799387799)
        elif name == "waist_roll_joint":
            lower.append(-0.34906585039)
            upper.append(0.34906585039)
        elif name == "waist_pitch_joint":
            lower.append(-0.48869219055)
            upper.append(0.41887902047)
        elif name == "right_shoulder_pitch_joint":
            lower.append(-2.87979326579)
            upper.append(2.87979326579)
        elif name == "right_shoulder_roll_joint":
            lower.append(-2.61799387799)
            upper.append(0.08726646259)
        elif name == "right_shoulder_yaw_joint":
            lower.append(-2.79252680319)
            upper.append(2.79252680319)
        elif name == "right_elbow_joint":
            lower.append(-0.08726646259)
            upper.append(2.87979326579)
        elif name == "right_wrist_roll_joint":
            lower.append(-1.91986217719)
            upper.append(1.91986217719)
        elif name == "right_wrist_pitch_joint":
            lower.append(-1.65806278939)
            upper.append(1.65806278939)
        elif name == "right_wrist_yaw_joint":
            lower.append(-2.26892802759)
            upper.append(2.26892802759)
        else:
            lower.append(-np.pi)
            upper.append(np.pi)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _joint_dynamic_caps(
    active_joint_names: list[str],
    duration_s: float,
) -> np.ndarray:
    caps = []
    for name in active_joint_names:
        vel_limit = 12.0
        acc_limit = 120.0
        if name == "waist_yaw_joint":
            vel_limit = 8.0
            acc_limit = 90.0
        elif name == "right_shoulder_pitch_joint":
            vel_limit = 10.0
            acc_limit = 80.0
        delta_q_vel_cap = vel_limit * duration_s / 1.875
        delta_q_acc_cap = acc_limit * duration_s * duration_s / 5.77
        caps.append(min(delta_q_vel_cap, delta_q_acc_cap))
    return np.asarray(caps, dtype=np.float64)


def _active_idx() -> list[int]:
    return [A3_POLICY_JOINT_ORDER.index(name) for name in A3_ACTIVE_JOINTS_FIRST_PASS]


def _compute_joint_vel_acc(q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    vel = np.gradient(q, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def _repair_post_segment(
    q_active: np.ndarray,
    anchor_frames: list[int],
    dt: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    # anchor layout from diagnostics:
    # [0]=boundary_start, [1]=pre_far, [2]=pre_near, [3]=hit, [4]=post_near, [5]=post_far, [6]=boundary_end
    post_near = int(anchor_frames[4])
    post_far = int(anchor_frames[5])
    end = int(anchor_frames[6])
    active_names = A3_ACTIVE_JOINTS_FIRST_PASS
    lower, upper = _joint_limits(active_names)

    q_post_near = q_active[post_near].copy()
    q_post_far_ref = q_active[post_far].copy()
    q_end_ref = q_active[end].copy()
    q_post_near = np.clip(q_post_near, lower, upper)
    q_post_far_ref = np.clip(q_post_far_ref, lower, upper)
    q_end_ref = np.clip(q_end_ref, lower, upper)

    # Variables: post_far and boundary_end active-joint targets.
    x0 = np.concatenate([q_post_far_ref, q_end_ref], axis=0)
    post_len = post_far - post_near + 1
    tail_len = end - post_far + 1
    post_duration_s = max((post_far - post_near) * dt, 1e-6)
    delta_cap = _joint_dynamic_caps(active_names, post_duration_s)
    tail_duration_s = max((end - post_far) * dt, 1e-6)
    tail_delta_cap = _joint_dynamic_caps(active_names, tail_duration_s)
    # boundary_end is a stabilization anchor, not a forced return-to-stand.
    # Bound it around post_far so the tail cannot create a large recovery snap.
    lb = np.concatenate([np.maximum(lower, q_post_near - delta_cap), np.maximum(lower, q_post_far_ref - tail_delta_cap)], axis=0)
    ub = np.concatenate([np.minimum(upper, q_post_near + delta_cap), np.minimum(upper, q_post_far_ref + tail_delta_cap)], axis=0)
    x0 = np.clip(x0, lb, ub)

    before_vel, before_acc = _compute_joint_vel_acc(q_active, dt)
    before_max_vel = float(np.max(np.abs(before_vel[post_near : end + 1])))
    before_max_acc = float(np.max(np.abs(before_acc[post_near : end + 1])))
    start_velocity = np.clip(before_vel[post_near], -12.0, 12.0)
    zero_velocity = np.zeros_like(start_velocity)

    def objective(x: np.ndarray) -> np.ndarray:
        q_post_far = x[: len(active_names)]
        q_end = x[len(active_names) :]
        seg1 = _interpolate_hermite_segment(q_post_near, q_post_far, post_len, dt, start_velocity, zero_velocity)
        seg2 = _interpolate_segment(q_post_far, q_end, tail_len)
        q_trial = q_active.copy()
        q_trial[post_near : post_far + 1] = seg1
        q_trial[post_far : end + 1] = seg2
        vel, acc = _compute_joint_vel_acc(q_trial, dt)
        jerk = np.gradient(acc, dt, axis=0)

        # Strong dynamic regularization on post segment only.
        seg_slice = slice(post_near, end + 1)
        vel_seg = vel[seg_slice]
        acc_seg = acc[seg_slice]
        jerk_seg = jerk[seg_slice]
        vel_limit = 12.0
        acc_limit = 120.0
        vel_excess = np.maximum(np.abs(vel_seg) - vel_limit, 0.0)
        acc_excess = np.maximum(np.abs(acc_seg) - acc_limit, 0.0)

        # Keep near-hit geometry stable by not moving post_near and softly
        # staying close to original post_far / end targets.
        res = [
            0.55 * (q_post_far - q_post_far_ref),
            0.55 * (q_end - q_post_far),
            0.05 * (q_end - q_end_ref),
            0.08 * vel_seg.ravel(),
            0.05 * acc_seg.ravel(),
            0.015 * jerk_seg.ravel(),
            0.30 * vel_excess.ravel(),
            0.40 * acc_excess.ravel(),
        ]
        return np.concatenate(res, axis=0)

    result = least_squares(objective, x0=x0, bounds=(lb, ub), max_nfev=48, verbose=0)
    q_post_far = result.x[: len(active_names)]
    q_end = result.x[len(active_names) :]
    seg1 = _interpolate_hermite_segment(q_post_near, q_post_far, post_len, dt, start_velocity, zero_velocity)
    seg2 = _interpolate_segment(q_post_far, q_end, tail_len)
    repaired = q_active.copy()
    repaired[post_near : post_far + 1] = seg1
    repaired[post_far : end + 1] = seg2
    vel, acc = _compute_joint_vel_acc(repaired, dt)
    candidate_max_vel = float(np.max(np.abs(vel[post_near : end + 1])))
    candidate_max_acc = float(np.max(np.abs(acc[post_near : end + 1])))
    before_score = max(before_max_vel / 12.0, before_max_acc / 120.0)
    candidate_score = max(candidate_max_vel / 12.0, candidate_max_acc / 120.0)

    hold_repaired = q_active.copy()
    hold_repaired[post_near : end + 1] = q_post_near[None, :]
    hold_vel, hold_acc = _compute_joint_vel_acc(hold_repaired, dt)
    hold_max_vel = float(np.max(np.abs(hold_vel[post_near : end + 1])))
    hold_max_acc = float(np.max(np.abs(hold_acc[post_near : end + 1])))
    hold_score = max(hold_max_vel / 12.0, hold_max_acc / 120.0)

    choice = min(
        [
            ("original", before_score, q_active, before_vel, before_acc, before_max_vel, before_max_acc),
            ("optimized", candidate_score, repaired, vel, acc, candidate_max_vel, candidate_max_acc),
            ("hold_post_near", hold_score, hold_repaired, hold_vel, hold_acc, hold_max_vel, hold_max_acc),
        ],
        key=lambda item: item[1],
    )
    repair_choice, _, repaired, vel, acc, after_max_vel, after_max_acc = choice
    kept_original = repair_choice == "original"
    return repaired, {
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "post_near_frame": post_near,
        "post_far_frame": post_far,
        "boundary_end_frame": end,
        "before_max_velocity_radps": before_max_vel,
        "before_max_acceleration_radps2": before_max_acc,
        "candidate_max_velocity_radps": candidate_max_vel,
        "candidate_max_acceleration_radps2": candidate_max_acc,
        "hold_post_near_max_velocity_radps": hold_max_vel,
        "hold_post_near_max_acceleration_radps2": hold_max_acc,
        "before_dynamic_score": before_score,
        "candidate_dynamic_score": candidate_score,
        "hold_post_near_dynamic_score": hold_score,
        "selected_repair": repair_choice,
        "max_velocity_radps": after_max_vel,
        "max_acceleration_radps2": after_max_acc,
        "kept_original_post_segment": kept_original,
        "delta_cap_rad": {name: float(cap) for name, cap in zip(active_names, delta_cap)},
        "tail_delta_cap_rad": {name: float(cap) for name, cap in zip(active_names, tail_delta_cap)},
    }


def repair_generic_init_csv(
    csv_path: str | Path,
    diagnostics_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    csv_path = Path(csv_path)
    diagnostics_path = Path(diagnostics_path)
    csv_data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    diagnostics = json.loads(diagnostics_path.read_text())
    dt = 0.005  # generic init follows the 200 Hz spec
    joint_block = csv_data[:, 7:].copy()
    active_idx = _active_idx()
    repaired_active, repair_report = _repair_post_segment(joint_block[:, active_idx], diagnostics["anchor_frames"], dt)
    repaired_csv = csv_data.copy()
    repaired_csv[:, 7 + np.asarray(active_idx, dtype=int)] = repaired_active
    return repaired_csv, {
        "source_csv": str(csv_path),
        "source_diagnostics": str(diagnostics_path),
        "post_temporal_repair": repair_report,
    }


def write_temporal_repair(
    csv_path: str | Path,
    diagnostics_path: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    repaired_csv, report = repair_generic_init_csv(csv_path, diagnostics_path)
    target_csv = Path(output_csv) if output_csv is not None else Path(csv_path)
    np.savetxt(target_csv, repaired_csv, delimiter=",", fmt="%.10f")
    report_path = target_csv.with_suffix(".temporal_repair.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return {
        "csv_path": str(target_csv),
        "report_path": str(report_path),
        **report,
    }
