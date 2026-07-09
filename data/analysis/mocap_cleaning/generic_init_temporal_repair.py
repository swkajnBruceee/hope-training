"""Independent temporal repair layer for generic retarget initialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from analysis.mocap_cleaning.a3_metadata import A3_ACTIVE_JOINTS_FIRST_PASS, A3_POLICY_JOINT_ORDER
from analysis.mocap_cleaning.a3_refinement_solver import _compute_racket_series


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


def _hit_metrics(csv_data: np.ndarray, spec: dict[str, Any], dt: float) -> dict[str, float]:
    hit = int(spec["hit_target"]["hit_index"])
    target_pos = np.asarray(spec["hit_target"]["racket_position_m"], dtype=np.float64)
    target_normal = np.asarray(spec["hit_target"]["racket_normal_w"], dtype=np.float64)
    target_vel_dir = np.asarray(spec["hit_target"]["racket_velocity_direction_w"], dtype=np.float64)
    racket_pos, racket_normal, _ = _compute_racket_series(csv_data, spec)
    racket_vel = np.gradient(racket_pos, dt, axis=0)
    vel_dir = racket_vel[hit] / max(float(np.linalg.norm(racket_vel[hit])), 1e-9)
    normal_cos = float(
        np.clip(
            np.dot(racket_normal[hit], target_normal)
            / (np.linalg.norm(racket_normal[hit]) * np.linalg.norm(target_normal) + 1e-9),
            -1.0,
            1.0,
        )
    )
    vel_cos = float(np.clip(np.dot(vel_dir, target_vel_dir) / (np.linalg.norm(target_vel_dir) + 1e-9), -1.0, 1.0))
    return {
        "position_error_m": float(np.linalg.norm(racket_pos[hit] - target_pos)),
        "orientation_error_deg": float(np.degrees(np.arccos(normal_cos))),
        "velocity_direction_error_deg": float(np.degrees(np.arccos(vel_cos))),
    }


def _dynamic_score(q_active: np.ndarray, dt: float) -> tuple[float, float, float]:
    vel, acc = _compute_joint_vel_acc(q_active, dt)
    max_vel = float(np.max(np.abs(vel))) if vel.size else 0.0
    max_acc = float(np.max(np.abs(acc))) if acc.size else 0.0
    return max(max_vel / 12.0, max_acc / 120.0), max_vel, max_acc


def _phase_for_frame(spec: dict[str, Any], frame_idx: int) -> str:
    windows = spec.get("windows", {})
    for phase_name in ("pre_hit", "hit", "post_hit"):
        block = windows.get(phase_name)
        if block is None:
            continue
        if int(block["frame_start"]) <= frame_idx <= int(block["frame_end"]):
            return phase_name
    return "boundary"


def _peak_cap_scale(anchor_frame: int, hit_frame: int, phase: str) -> float:
    distance = abs(int(anchor_frame) - int(hit_frame))
    if phase == "hit":
        if distance <= 2:
            return 0.42
        if distance <= 5:
            return 0.58
        return 0.75
    if phase == "pre_hit":
        return 0.48 if anchor_frame <= hit_frame else 0.78
    if phase == "post_hit":
        return 0.48 if anchor_frame >= hit_frame else 0.78
    if distance <= 2:
        return 0.55
    return 0.68


def _repair_hit_window(
    csv_data: np.ndarray,
    q_active: np.ndarray,
    spec: dict[str, Any],
    anchor_frames: list[int],
    active_idx: list[int],
    dt: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    # 7-anchor layout:
    # [0]=boundary_start, [1]=pre_far, [2]=pre_near, [3]=hit, [4]=post_near, [5]=post_far, [6]=boundary_end
    pre_far = int(anchor_frames[1])
    pre_near = int(anchor_frames[2])
    hit = int(anchor_frames[3])
    post_near = int(anchor_frames[4])
    post_far = int(anchor_frames[5])
    active_names = A3_ACTIVE_JOINTS_FIRST_PASS
    lower, upper = _joint_limits(active_names)

    q_pre_far = np.clip(q_active[pre_far].copy(), lower, upper)
    q_pre_ref = np.clip(q_active[pre_near].copy(), lower, upper)
    q_hit_ref = np.clip(q_active[hit].copy(), lower, upper)
    q_post_ref = np.clip(q_active[post_near].copy(), lower, upper)
    q_post_far = np.clip(q_active[post_far].copy(), lower, upper)

    x0 = np.concatenate([q_pre_ref, q_hit_ref, q_post_ref], axis=0)
    pre_cap = np.minimum(_joint_dynamic_caps(active_names, max((hit - pre_near) * dt, 1e-6)) * 5.0, 0.70)
    post_cap = np.minimum(_joint_dynamic_caps(active_names, max((post_near - hit) * dt, 1e-6)) * 5.0, 0.70)
    # Keep the solved hit posture fixed; this layer may reshape the approach
    # and exit corridor, but must not trade away the contact pose.
    hit_cap = np.ones_like(pre_cap) * 1e-6
    lb = np.concatenate(
        [
            np.maximum(lower, q_hit_ref - pre_cap),
            np.maximum(lower, q_hit_ref - hit_cap),
            np.maximum(lower, q_hit_ref - post_cap),
        ],
        axis=0,
    )
    ub = np.concatenate(
        [
            np.minimum(upper, q_hit_ref + pre_cap),
            np.minimum(upper, q_hit_ref + hit_cap),
            np.minimum(upper, q_hit_ref + post_cap),
        ],
        axis=0,
    )
    x0 = np.clip(x0, lb, ub)

    base_csv = csv_data.copy()
    before_vel_series, before_acc_series = _compute_joint_vel_acc(q_active, dt)
    before_acc_idx = np.unravel_index(int(np.argmax(np.abs(before_acc_series))), before_acc_series.shape)
    before_acc_frame = int(before_acc_idx[0])
    before_acc_joint = int(before_acc_idx[1])
    before_acc_phase = _phase_for_frame(spec, before_acc_frame)
    peak_acc_joint_name = active_names[before_acc_joint]
    peak_frame_weights = np.ones(q_active.shape[0], dtype=np.float64)
    if before_acc_phase == "hit":
        lo = max(0, hit_frame - 8)
        hi = min(q_active.shape[0], hit_frame + 9)
        peak_frame_weights[lo:hi] = 2.5
    elif before_acc_phase == "pre_hit":
        lo = max(0, hit_frame - 16)
        hi = min(q_active.shape[0], hit_frame + 1)
        peak_frame_weights[lo:hi] = 2.0
    elif before_acc_phase == "post_hit":
        lo = max(0, hit_frame - 1)
        hi = min(q_active.shape[0], hit_frame + 17)
        peak_frame_weights[lo:hi] = 2.0
    else:
        peak_frame_weights[: max(12, pre_far + 2)] = 1.6
        peak_frame_weights[max(post_far - 2, 0) :] = 1.6
    joint_weights = np.ones(len(active_names), dtype=np.float64)
    joint_weights += np.clip(np.max(np.abs(before_acc_series), axis=0) / 120.0, 0.0, 2.5)
    joint_weights[before_acc_joint] += 1.5
    pre_cap_mod = pre_cap.copy()
    post_cap_mod = post_cap.copy()
    if before_acc_phase in ("pre_hit", "hit", "boundary"):
        pre_cap_mod[before_acc_joint] *= 0.65
    if before_acc_phase in ("post_hit", "hit", "boundary"):
        post_cap_mod[before_acc_joint] *= 0.65
    lb = np.concatenate(
        [
            np.maximum(lower, q_hit_ref - pre_cap_mod),
            np.maximum(lower, q_hit_ref - hit_cap),
            np.maximum(lower, q_hit_ref - post_cap_mod),
        ],
        axis=0,
    )
    ub = np.concatenate(
        [
            np.minimum(upper, q_hit_ref + pre_cap_mod),
            np.minimum(upper, q_hit_ref + hit_cap),
            np.minimum(upper, q_hit_ref + post_cap_mod),
        ],
        axis=0,
    )
    x0 = np.clip(x0, lb, ub)

    frame_weights_window = peak_frame_weights[np.asarray([frame for frame in range(pre_far, post_far + 1)], dtype=int)]
    before_hit = _hit_metrics(base_csv, spec, dt)
    before_score, before_max_vel, before_max_acc = _dynamic_score(q_active, dt)
    target_pos = np.asarray(spec["hit_target"]["racket_position_m"], dtype=np.float64)
    target_normal = np.asarray(spec["hit_target"]["racket_normal_w"], dtype=np.float64)
    target_tangent = np.asarray(spec["hit_target"]["racket_tangent_w"], dtype=np.float64)
    target_vel_dir = np.asarray(spec["hit_target"]["racket_velocity_direction_w"], dtype=np.float64)

    def build_trial(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_pre = x[: len(active_names)]
        q_hit = x[len(active_names) : 2 * len(active_names)]
        q_post = x[2 * len(active_names) :]
        q_trial = q_active.copy()
        frames = np.asarray([pre_far, pre_near, hit, post_near, post_far], dtype=np.float64)
        values = np.stack([q_pre_far, q_pre, q_hit, q_post, q_post_far], axis=0)
        t = np.arange(pre_far, post_far + 1, dtype=np.float64)
        window_values = np.zeros((t.shape[0], len(active_names)), dtype=np.float64)
        for local_j in range(len(active_names)):
            spline = CubicSpline(frames, values[:, local_j], bc_type="natural")
            window_values[:, local_j] = spline(t)
        q_trial[pre_far : post_far + 1] = window_values
        trial_csv = base_csv.copy()
        trial_csv[:, 7 + np.asarray(active_idx, dtype=int)] = q_trial
        return q_trial, trial_csv

    def objective(x: np.ndarray) -> np.ndarray:
        q_trial, trial_csv = build_trial(x)
        racket_pos, racket_normal, racket_tangent = _compute_racket_series(trial_csv, spec)
        racket_vel = np.gradient(racket_pos, dt, axis=0)
        vel_dir = racket_vel[hit] / max(float(np.linalg.norm(racket_vel[hit])), 1e-9)
        vel, acc = _compute_joint_vel_acc(q_trial, dt)
        jerk = np.gradient(acc, dt, axis=0)
        window = slice(pre_far, post_far + 1)
        frame_weights = frame_weights_window[:, None]
        joint_weight_vec = joint_weights[None, :]
        vel_excess = np.maximum(np.abs(vel[window]) - 12.0, 0.0)
        acc_excess = np.maximum(np.abs(acc[window]) - 120.0, 0.0)
        q_pre = x[: len(active_names)]
        q_hit = x[len(active_names) : 2 * len(active_names)]
        q_post = x[2 * len(active_names) :]
        return np.concatenate(
            [
                90.0 * (racket_pos[hit] - target_pos),
                35.0 * (racket_normal[hit] - target_normal),
                12.0 * (racket_tangent[hit] - target_tangent),
                10.0 * (vel_dir - target_vel_dir),
                0.18 * (joint_weights * (q_pre - q_pre_ref)),
                0.40 * (joint_weights * (q_hit - q_hit_ref)),
                0.18 * (joint_weights * (q_post - q_post_ref)),
                0.015 * (frame_weights * joint_weight_vec * vel[window]).ravel(),
                0.0025 * (frame_weights * joint_weight_vec * acc[window]).ravel(),
                0.00015 * (frame_weights * joint_weight_vec * jerk[window]).ravel(),
                0.18 * (frame_weights * joint_weight_vec * vel_excess).ravel(),
                0.020 * (frame_weights * joint_weight_vec * acc_excess).ravel(),
            ],
            axis=0,
        )

    result = least_squares(objective, x0=x0, bounds=(lb, ub), max_nfev=36, verbose=0)
    candidate_active, candidate_csv = build_trial(result.x)
    candidate_hit = _hit_metrics(candidate_csv, spec, dt)
    candidate_score, candidate_max_vel, candidate_max_acc = _dynamic_score(candidate_active, dt)

    hit_ok = (
        candidate_hit["position_error_m"] <= max(0.02, before_hit["position_error_m"] + 0.01)
        and candidate_hit["orientation_error_deg"] <= max(2.0, before_hit["orientation_error_deg"] + 1.0)
        and candidate_hit["velocity_direction_error_deg"] <= max(18.0, before_hit["velocity_direction_error_deg"] + 5.0)
    )
    improved = candidate_score < before_score
    if hit_ok and improved:
        return candidate_active, {
            "selected_repair": "optimized",
            "optimizer_success": bool(result.success),
            "optimizer_nfev": int(result.nfev),
            "before_hit": before_hit,
            "candidate_hit": candidate_hit,
            "before_dynamic_score": before_score,
            "candidate_dynamic_score": candidate_score,
            "before_max_velocity_radps": before_max_vel,
            "before_max_acceleration_radps2": before_max_acc,
            "max_velocity_radps": candidate_max_vel,
            "max_acceleration_radps2": candidate_max_acc,
            "peak_acc_joint": peak_acc_joint_name,
            "peak_acc_frame": before_acc_frame,
            "peak_acc_phase": before_acc_phase,
        }
    return q_active.copy(), {
        "selected_repair": "original",
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "rollback_reason": "hit_constraint_or_dynamic_score",
        "before_hit": before_hit,
        "candidate_hit": candidate_hit,
        "before_dynamic_score": before_score,
        "candidate_dynamic_score": candidate_score,
        "before_max_velocity_radps": before_max_vel,
        "before_max_acceleration_radps2": before_max_acc,
        "candidate_max_velocity_radps": candidate_max_vel,
        "candidate_max_acceleration_radps2": candidate_max_acc,
        "max_velocity_radps": before_max_vel,
        "max_acceleration_radps2": before_max_acc,
        "peak_acc_joint": peak_acc_joint_name,
        "peak_acc_frame": before_acc_frame,
        "peak_acc_phase": before_acc_phase,
    }


def _repair_spline_control_points(
    csv_data: np.ndarray,
    q_active: np.ndarray,
    spec: dict[str, Any],
    anchor_frames: list[int],
    active_idx: list[int],
    dt: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    active_names = A3_ACTIVE_JOINTS_FIRST_PASS
    lower, upper = _joint_limits(active_names)
    frames = np.asarray(anchor_frames, dtype=np.float64)
    frame_idx = np.asarray(anchor_frames, dtype=int)
    hit_anchor_idx = 3
    hit_frame = int(frame_idx[hit_anchor_idx])
    opt_anchor_indices = [i for i in range(len(frame_idx)) if i != hit_anchor_idx]
    ref_anchor = np.clip(q_active[frame_idx].copy(), lower[None, :], upper[None, :])
    q_hit_ref = ref_anchor[hit_anchor_idx].copy()

    x0 = ref_anchor[opt_anchor_indices].ravel()
    cap = np.ones_like(ref_anchor[opt_anchor_indices]) * 0.22
    # Let preparation and far follow-through absorb more smoothing; keep
    # near-hit controls tighter so velocity direction does not drift.
    for row_i, anchor_i in enumerate(opt_anchor_indices):
        if anchor_i in (1, 5, 6):
            cap[row_i, :] = 0.45
        if anchor_i in (2, 4):
            cap[row_i, :] = 0.16
    lb = np.maximum(lower[None, :], ref_anchor[opt_anchor_indices] - cap).ravel()
    ub = np.minimum(upper[None, :], ref_anchor[opt_anchor_indices] + cap).ravel()
    x0 = np.clip(x0, lb, ub)

    base_csv = csv_data.copy()
    before_hit = _hit_metrics(base_csv, spec, dt)
    before_score, before_max_vel, before_max_acc = _dynamic_score(q_active, dt)
    target_vel_dir = np.asarray(spec["hit_target"]["racket_velocity_direction_w"], dtype=np.float64)

    def build_trial(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        controls = ref_anchor.copy()
        controls[hit_anchor_idx] = q_hit_ref
        controls[opt_anchor_indices] = x.reshape((len(opt_anchor_indices), len(active_names)))
        q_trial = q_active.copy()
        t = np.arange(q_active.shape[0], dtype=np.float64)
        for local_j in range(len(active_names)):
            spline = CubicSpline(frames, controls[:, local_j], bc_type="natural")
            q_trial[:, local_j] = spline(t)
        q_trial = np.clip(q_trial, lower[None, :], upper[None, :])
        trial_csv = base_csv.copy()
        trial_csv[:, 7 + np.asarray(active_idx, dtype=int)] = q_trial
        return q_trial, trial_csv

    def objective(x: np.ndarray) -> np.ndarray:
        q_trial, trial_csv = build_trial(x)
        vel, acc = _compute_joint_vel_acc(q_trial, dt)
        jerk = np.gradient(acc, dt, axis=0)
        racket_pos, _, _ = _compute_racket_series(trial_csv, spec)
        racket_vel = np.gradient(racket_pos, dt, axis=0)
        vel_dir = racket_vel[hit_frame] / max(float(np.linalg.norm(racket_vel[hit_frame])), 1e-9)
        vel_excess = np.maximum(np.abs(vel) - 12.0, 0.0)
        acc_excess = np.maximum(np.abs(acc) - 120.0, 0.0)
        controls = x.reshape((len(opt_anchor_indices), len(active_names)))
        ref_controls = ref_anchor[opt_anchor_indices]
        return np.concatenate(
            [
                10.0 * (vel_dir - target_vel_dir),
                0.35 * (controls - ref_controls).ravel(),
                0.020 * vel.ravel(),
                0.0030 * acc.ravel(),
                0.0002 * jerk.ravel(),
                0.20 * vel_excess.ravel(),
                0.030 * acc_excess.ravel(),
            ],
            axis=0,
        )

    result = least_squares(objective, x0=x0, bounds=(lb, ub), max_nfev=24, verbose=0)
    candidate_active, candidate_csv = build_trial(result.x)
    candidate_hit = _hit_metrics(candidate_csv, spec, dt)
    candidate_score, candidate_max_vel, candidate_max_acc = _dynamic_score(candidate_active, dt)
    hit_ok = (
        candidate_hit["position_error_m"] <= max(0.02, before_hit["position_error_m"] + 0.01)
        and candidate_hit["orientation_error_deg"] <= max(2.0, before_hit["orientation_error_deg"] + 1.0)
        and candidate_hit["velocity_direction_error_deg"] <= max(20.0, before_hit["velocity_direction_error_deg"] + 4.0)
    )
    improved = candidate_score < before_score
    if hit_ok and improved:
        return candidate_active, {
            "selected_repair": "optimized",
            "optimizer_success": bool(result.success),
            "optimizer_nfev": int(result.nfev),
            "before_hit": before_hit,
            "candidate_hit": candidate_hit,
            "before_dynamic_score": before_score,
            "candidate_dynamic_score": candidate_score,
            "before_max_velocity_radps": before_max_vel,
            "before_max_acceleration_radps2": before_max_acc,
            "max_velocity_radps": candidate_max_vel,
            "max_acceleration_radps2": candidate_max_acc,
        }
    return q_active.copy(), {
        "selected_repair": "original",
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "rollback_reason": "hit_constraint_or_dynamic_score",
        "before_hit": before_hit,
        "candidate_hit": candidate_hit,
        "before_dynamic_score": before_score,
        "candidate_dynamic_score": candidate_score,
        "before_max_velocity_radps": before_max_vel,
        "before_max_acceleration_radps2": before_max_acc,
        "candidate_max_velocity_radps": candidate_max_vel,
        "candidate_max_acceleration_radps2": candidate_max_acc,
        "max_velocity_radps": before_max_vel,
        "max_acceleration_radps2": before_max_acc,
    }


def _repair_post_segment(
    q_active: np.ndarray,
    anchor_frames: list[int],
    dt: float,
    spec: dict[str, Any] | None = None,
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
    before_acc_idx = np.unravel_index(int(np.argmax(np.abs(before_acc))), before_acc.shape)
    before_acc_frame = int(before_acc_idx[0])
    before_acc_joint = int(before_acc_idx[1])
    before_acc_phase = _phase_for_frame(spec, before_acc_frame) if spec is not None else "boundary"
    peak_acc_joint_name = active_names[before_acc_joint]
    before_max_vel = float(np.max(np.abs(before_vel[post_near : end + 1])))
    before_max_acc = float(np.max(np.abs(before_acc[post_near : end + 1])))
    start_velocity = np.clip(before_vel[post_near], -12.0, 12.0)
    zero_velocity = np.zeros_like(start_velocity)
    peak_cap_scale = np.ones_like(delta_cap)
    for joint_i, joint_name in enumerate(active_names):
        if joint_name != peak_acc_joint_name:
            continue
        peak_cap_scale[joint_i] = min(peak_cap_scale[joint_i], 0.55 if before_acc_phase == "boundary" else 0.70)
    delta_cap = delta_cap * peak_cap_scale
    tail_delta_cap = tail_delta_cap * peak_cap_scale

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
        "peak_acc_joint": peak_acc_joint_name,
        "peak_acc_frame": before_acc_frame,
        "peak_acc_phase": before_acc_phase,
    }


def _repair_local_peak_window(
    csv_data: np.ndarray,
    q_active: np.ndarray,
    spec: dict[str, Any],
    active_idx: list[int],
    dt: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    active_names = A3_ACTIVE_JOINTS_FIRST_PASS
    lower, upper = _joint_limits(active_names)
    before_vel, before_acc = _compute_joint_vel_acc(q_active, dt)
    before_max_acc = float(np.max(np.abs(before_acc))) if before_acc.size else 0.0
    before_max_vel = float(np.max(np.abs(before_vel))) if before_vel.size else 0.0
    if before_max_acc <= 120.0:
        return q_active.copy(), {"selected_repair": "skipped", "reason": "acceleration_within_limit"}

    peak_idx = np.unravel_index(int(np.argmax(np.abs(before_acc))), before_acc.shape)
    peak_frame = int(peak_idx[0])
    peak_joint = int(peak_idx[1])
    peak_joint_name = active_names[peak_joint]
    peak_phase = _phase_for_frame(spec, peak_frame)
    hit_frame = int(spec["hit_target"]["hit_index"])

    companion_joint_names: list[str] = []
    profile_key = f"{peak_joint_name}@{peak_phase}"
    if peak_joint_name == "right_shoulder_yaw_joint":
        companion_joint_names = ["waist_yaw_joint", "waist_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint"]
        if peak_phase == "pre_hit":
            companion_joint_names.append("right_wrist_yaw_joint")
    elif peak_joint_name == "right_shoulder_roll_joint" and peak_phase == "hit":
        companion_joint_names = ["waist_roll_joint", "waist_yaw_joint", "right_shoulder_yaw_joint", "right_elbow_joint"]
    companion_joint_indices = [active_names.index(name) for name in companion_joint_names if name in active_names]
    joint_indices = [peak_joint] + companion_joint_indices

    radius = 3 if peak_phase == "hit" else 4
    if peak_joint_name == "right_shoulder_yaw_joint" and peak_phase == "hit":
        radius = 5
    elif peak_joint_name == "right_shoulder_yaw_joint" and peak_phase == "pre_hit":
        radius = 5
    elif peak_joint_name == "right_shoulder_roll_joint" and peak_phase == "hit":
        radius = 4
    window_lo = max(0, peak_frame - radius)
    window_hi = min(q_active.shape[0], peak_frame + radius + 1)
    variable_frames = list(range(window_lo, window_hi))
    if peak_phase == "hit" and hit_frame in variable_frames:
        variable_frames.remove(hit_frame)
    if not variable_frames:
        return q_active.copy(), {
            "selected_repair": "skipped",
            "reason": "no_variable_frames",
            "peak_acc_joint": peak_joint_name,
            "peak_acc_frame": peak_frame,
            "peak_acc_phase": peak_phase,
        }

    base_csv = csv_data.copy()
    before_hit = _hit_metrics(base_csv, spec, dt)
    original_values = q_active[np.asarray(variable_frames, dtype=int)[:, None], np.asarray(joint_indices, dtype=int)[None, :]]
    x0 = original_values.ravel()

    cap = np.full_like(original_values, 0.06, dtype=np.float64)
    cap[:, 0] = 0.10 if peak_phase == "boundary" else 0.08
    if peak_joint_name == "right_shoulder_yaw_joint" and peak_phase == "hit":
        cap[:, 0] = 0.24
    elif peak_joint_name == "right_shoulder_yaw_joint" and peak_phase == "pre_hit":
        cap[:, 0] = 0.20
    elif peak_joint_name == "right_shoulder_roll_joint" and peak_phase == "hit":
        cap[:, 0] = 0.18
    if peak_joint_name == "waist_yaw_joint" and peak_phase == "hit":
        cap[:, 0] = 0.09
    if len(joint_indices) > 1:
        if peak_joint_name == "right_shoulder_yaw_joint":
            cap[:, 1:] = 0.09
        elif peak_joint_name == "right_shoulder_roll_joint" and peak_phase == "hit":
            cap[:, 1:] = 0.07
        else:
            cap[:, 1:] = 0.035
    ref_values = original_values.copy()
    stay_weights = np.full((len(variable_frames), len(joint_indices)), 0.65, dtype=np.float64)
    if peak_joint_name == "right_shoulder_yaw_joint" and peak_phase == "hit":
        stay_weights[:, 0] = 0.35
        if stay_weights.shape[1] > 1:
            stay_weights[:, 1:] = 0.25
    elif peak_joint_name == "right_shoulder_yaw_joint" and peak_phase == "pre_hit":
        stay_weights[:, 0] = 0.40
        if stay_weights.shape[1] > 1:
            stay_weights[:, 1:] = 0.28
    elif peak_joint_name == "right_shoulder_roll_joint" and peak_phase == "hit":
        stay_weights[:, 0] = 0.40
        if stay_weights.shape[1] > 1:
            stay_weights[:, 1:] = 0.30
    lower_sel = lower[np.asarray(joint_indices, dtype=int)][None, :]
    upper_sel = upper[np.asarray(joint_indices, dtype=int)][None, :]
    lb = np.maximum(lower_sel, ref_values - cap).ravel()
    ub = np.minimum(upper_sel, ref_values + cap).ravel()
    x0 = np.clip(x0, lb, ub)
    target_vel_dir = np.asarray(spec["hit_target"]["racket_velocity_direction_w"], dtype=np.float64)

    def build_trial(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_trial = q_active.copy()
        values = x.reshape((len(variable_frames), len(joint_indices)))
        for frame_local, frame_idx in enumerate(variable_frames):
            q_trial[frame_idx, joint_indices] = values[frame_local]
        q_trial[:, joint_indices] = np.clip(q_trial[:, joint_indices], lower[joint_indices][None, :], upper[joint_indices][None, :])
        trial_csv = base_csv.copy()
        trial_csv[:, 7 + np.asarray(active_idx, dtype=int)] = q_trial
        return q_trial, trial_csv

    def objective(x: np.ndarray) -> np.ndarray:
        q_trial, trial_csv = build_trial(x)
        vel, acc = _compute_joint_vel_acc(q_trial, dt)
        jerk = np.gradient(acc, dt, axis=0)
        local_slice = slice(max(0, window_lo - 1), min(q_trial.shape[0], window_hi + 1))
        values = x.reshape((len(variable_frames), len(joint_indices)))
        vel_local = vel[local_slice][:, joint_indices]
        acc_local = acc[local_slice][:, joint_indices]
        jerk_local = jerk[local_slice][:, joint_indices]
        acc_excess = np.maximum(np.abs(acc_local) - 120.0, 0.0)
        acc_penalty = 0.24
        if peak_joint_name == "right_shoulder_yaw_joint":
            acc_penalty = 0.32 if peak_phase == "hit" else 0.30
        elif peak_joint_name == "right_shoulder_roll_joint" and peak_phase == "hit":
            acc_penalty = 0.30
        res = [
            (stay_weights * (values - ref_values)).ravel(),
            0.030 * vel_local.ravel(),
            0.010 * acc_local.ravel(),
            0.0008 * jerk_local.ravel(),
            acc_penalty * acc_excess.ravel(),
        ]
        if peak_phase in ("hit", "pre_hit"):
            racket_pos, _, _ = _compute_racket_series(trial_csv, spec)
            racket_vel = np.gradient(racket_pos, dt, axis=0)
            vel_dir = racket_vel[hit_frame] / max(float(np.linalg.norm(racket_vel[hit_frame])), 1e-9)
            vel_dir_weight = 18.0 if peak_phase == "hit" else 12.0
            res.append(vel_dir_weight * (vel_dir - target_vel_dir))
        return np.concatenate(res, axis=0)

    result = least_squares(objective, x0=x0, bounds=(lb, ub), max_nfev=32, verbose=0)
    candidate_active, candidate_csv = build_trial(result.x)
    candidate_hit = _hit_metrics(candidate_csv, spec, dt)
    candidate_vel, candidate_acc = _compute_joint_vel_acc(candidate_active, dt)
    candidate_max_vel = float(np.max(np.abs(candidate_vel))) if candidate_vel.size else 0.0
    candidate_max_acc = float(np.max(np.abs(candidate_acc))) if candidate_acc.size else 0.0
    before_peak_acc = float(abs(before_acc[peak_frame, peak_joint]))
    candidate_peak_acc = float(abs(candidate_acc[peak_frame, peak_joint]))

    hit_ok = (
        candidate_hit["position_error_m"] <= max(0.005, before_hit["position_error_m"] + 0.002)
        and candidate_hit["orientation_error_deg"] <= max(0.5, before_hit["orientation_error_deg"] + 0.2)
        and candidate_hit["velocity_direction_error_deg"] <= max(3.0, before_hit["velocity_direction_error_deg"] + 2.5)
    )
    improved = candidate_peak_acc < before_peak_acc - 1e-3 and candidate_max_acc <= before_max_acc + 2.0
    vel_ok = candidate_max_vel <= before_max_vel + 0.2
    if hit_ok and improved and vel_ok:
        return candidate_active, {
            "selected_repair": "optimized",
            "optimizer_success": bool(result.success),
            "optimizer_nfev": int(result.nfev),
            "peak_acc_joint": peak_joint_name,
            "peak_acc_frame": peak_frame,
            "peak_acc_phase": peak_phase,
            "profile_key": profile_key,
            "companion_joints": companion_joint_names,
            "before_hit": before_hit,
            "candidate_hit": candidate_hit,
            "before_peak_acceleration_radps2": before_peak_acc,
            "candidate_peak_acceleration_radps2": candidate_peak_acc,
            "before_max_velocity_radps": before_max_vel,
            "before_max_acceleration_radps2": before_max_acc,
            "max_velocity_radps": candidate_max_vel,
            "max_acceleration_radps2": candidate_max_acc,
        }
    return q_active.copy(), {
        "selected_repair": "original",
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "rollback_reason": "hit_constraint_or_peak_not_improved",
        "peak_acc_joint": peak_joint_name,
        "peak_acc_frame": peak_frame,
        "peak_acc_phase": peak_phase,
        "profile_key": profile_key,
        "companion_joints": companion_joint_names,
        "before_hit": before_hit,
        "candidate_hit": candidate_hit,
        "before_peak_acceleration_radps2": before_peak_acc,
        "candidate_peak_acceleration_radps2": candidate_peak_acc,
        "before_max_velocity_radps": before_max_vel,
        "before_max_acceleration_radps2": before_max_acc,
        "candidate_max_velocity_radps": candidate_max_vel,
        "candidate_max_acceleration_radps2": candidate_max_acc,
        "max_velocity_radps": before_max_vel,
        "max_acceleration_radps2": before_max_acc,
    }


def repair_generic_init_csv(
    csv_path: str | Path,
    diagnostics_path: str | Path,
    spec: dict[str, Any] | None = None,
    enable_hit_window: bool = False,
    enable_spline_control: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    csv_path = Path(csv_path)
    diagnostics_path = Path(diagnostics_path)
    csv_data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    diagnostics = json.loads(diagnostics_path.read_text())
    dt = 0.005  # generic init follows the 200 Hz spec
    joint_block = csv_data[:, 7:].copy()
    active_idx = _active_idx()
    repaired_active = joint_block[:, active_idx].copy()
    if spec is not None and enable_hit_window:
        repaired_active, hit_window_report = _repair_hit_window(
            csv_data=csv_data,
            q_active=repaired_active,
            spec=spec,
            anchor_frames=diagnostics["anchor_frames"],
            active_idx=active_idx,
            dt=dt,
        )
        csv_data = csv_data.copy()
        csv_data[:, 7 + np.asarray(active_idx, dtype=int)] = repaired_active
    else:
        hit_window_report = {"skipped": True, "reason": "disabled_or_spec_not_provided"}
    if spec is not None and enable_spline_control:
        repaired_active, spline_control_report = _repair_spline_control_points(
            csv_data=csv_data,
            q_active=repaired_active,
            spec=spec,
            anchor_frames=diagnostics["anchor_frames"],
            active_idx=active_idx,
            dt=dt,
        )
        csv_data = csv_data.copy()
        csv_data[:, 7 + np.asarray(active_idx, dtype=int)] = repaired_active
    else:
        spline_control_report = {"skipped": True, "reason": "disabled_or_spec_not_provided"}
    repaired_active, repair_report = _repair_post_segment(
        repaired_active,
        diagnostics["anchor_frames"],
        dt,
        spec=spec,
    )
    repaired_csv = csv_data.copy()
    repaired_csv[:, 7 + np.asarray(active_idx, dtype=int)] = repaired_active
    if spec is not None:
        local_peak_passes: list[dict[str, Any]] = []
        for _ in range(2):
            repaired_active, pass_report = _repair_local_peak_window(
                repaired_csv,
                repaired_active,
                spec,
                active_idx,
                dt,
            )
            local_peak_passes.append(pass_report)
            repaired_csv = csv_data.copy()
            repaired_csv[:, 7 + np.asarray(active_idx, dtype=int)] = repaired_active
            if pass_report.get("selected_repair") != "optimized":
                break
            if float(pass_report.get("max_acceleration_radps2", 0.0)) <= 120.0:
                break
        final_pass = local_peak_passes[-1]
        any_optimized = any(item.get("selected_repair") == "optimized" for item in local_peak_passes)
        local_peak_report = {
            **final_pass,
            "selected_repair": "optimized" if any_optimized else final_pass.get("selected_repair"),
            "passes": local_peak_passes,
        }
    else:
        local_peak_report = {"skipped": True, "reason": "spec_not_provided"}
    repaired_csv = csv_data.copy()
    repaired_csv[:, 7 + np.asarray(active_idx, dtype=int)] = repaired_active
    return repaired_csv, {
        "source_csv": str(csv_path),
        "source_diagnostics": str(diagnostics_path),
        "hit_window_temporal_repair": hit_window_report,
        "spline_control_temporal_repair": spline_control_report,
        "post_temporal_repair": repair_report,
        "local_peak_temporal_repair": local_peak_report,
    }


def write_temporal_repair(
    csv_path: str | Path,
    diagnostics_path: str | Path,
    output_csv: str | Path | None = None,
    spec: dict[str, Any] | None = None,
    enable_hit_window: bool = False,
    enable_spline_control: bool = True,
) -> dict[str, Any]:
    repaired_csv, report = repair_generic_init_csv(
        csv_path,
        diagnostics_path,
        spec=spec,
        enable_hit_window=enable_hit_window,
        enable_spline_control=enable_spline_control,
    )
    target_csv = Path(output_csv) if output_csv is not None else Path(csv_path)
    np.savetxt(target_csv, repaired_csv, delimiter=",", fmt="%.10f")
    report_path = target_csv.with_suffix(".temporal_repair.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return {
        "csv_path": str(target_csv),
        "report_path": str(report_path),
        **report,
    }
