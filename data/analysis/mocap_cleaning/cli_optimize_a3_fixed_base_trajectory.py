#!/usr/bin/env python3
"""Optimize fixed-base A3 trajectories with hit-first constraints."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    del _ROOT

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from analysis.mocap_cleaning.a3_metadata import A3_POLICY_JOINT_ORDER
from analysis.mocap_cleaning.a3_refinement_solver import (
    _fk_racket_state,
    compute_joint_vel_acc,
    load_a3_joint_limits,
    load_retarget_csv,
    write_retarget_csv,
)
from analysis.mocap_cleaning.config import load_config


CONTROL_POINT_ORDER = (
    "boundary_start",
    "pre_far",
    "pre_near",
    "hit",
    "post_near",
    "post_far",
    "boundary_end",
)


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(norm, 1e-9))


def _config_float(mapping: dict[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    return float(value)


def _target_axes(quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = []
    tangents = []
    for quat in quat_xyzw:
        rot = _quat_xyzw_to_matrix(quat)
        normals.append(rot[:, 1])
        tangents.append(rot[:, 0])
    return np.asarray(normals, dtype=np.float64), np.asarray(tangents, dtype=np.float64)


def _racket_series(csv_data: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_index_by_name = {name: i for i, name in enumerate(A3_POLICY_JOINT_ORDER)}
    pos = []
    normal = []
    tangent = []
    for row in csv_data:
        p, rot = _fk_racket_state(base_pos, base_quat, row[7:], joint_index_by_name, {"a3_joint_order": A3_POLICY_JOINT_ORDER})
        pos.append(p)
        normal.append(rot[:, 1])
        tangent.append(rot[:, 0])
    return np.asarray(pos), np.asarray(normal), np.asarray(tangent)


def _racket_series_for_frames(
    csv_data: np.ndarray,
    frame_indices: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_index_by_name = {name: i for i, name in enumerate(A3_POLICY_JOINT_ORDER)}
    pos = []
    normal = []
    tangent = []
    for frame in frame_indices.astype(int):
        row = csv_data[frame]
        p, rot = _fk_racket_state(base_pos, base_quat, row[7:], joint_index_by_name, {"a3_joint_order": A3_POLICY_JOINT_ORDER})
        pos.append(p)
        normal.append(rot[:, 1])
        tangent.append(rot[:, 0])
    return np.asarray(pos), np.asarray(normal), np.asarray(tangent)


def _joint_jerk(joint_acc: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(joint_acc, dt, axis=0)


def _control_frames(hit_index: int, n_frames: int, config: dict[str, Any]) -> dict[str, int]:
    offsets = config["optimization_control_points"]
    frames = {
        "boundary_start": 0,
        "pre_far": int(np.clip(hit_index + int(offsets["pre_far_offset"]), 0, n_frames - 1)),
        "pre_near": int(np.clip(hit_index + int(offsets["pre_near_offset"]), 0, n_frames - 1)),
        "hit": int(np.clip(hit_index + int(offsets["hit_offset"]), 0, n_frames - 1)),
        "post_near": int(np.clip(hit_index + int(offsets["post_near_offset"]), 0, n_frames - 1)),
        "post_far": int(np.clip(hit_index + int(offsets["post_far_offset"]), 0, n_frames - 1)),
        "boundary_end": n_frames - 1,
    }
    last = -1
    for name in CONTROL_POINT_ORDER:
        frame = frames[name]
        if frame <= last:
            frame = min(n_frames - 1, last + 1)
            frames[name] = frame
        last = frame
    return frames


def _hit_and_corridor_masks(hit_index: int, n_frames: int, control_frames: dict[str, int], phase_windows: dict[str, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    hit_lo = max(0, hit_index + int(phase_windows["hit"][0]))
    hit_hi = min(n_frames - 1, hit_index + int(phase_windows["hit"][1]))
    hit_mask = np.zeros(n_frames, dtype=bool)
    hit_mask[hit_lo : hit_hi + 1] = True
    corridor_lo = int(control_frames["pre_near"])
    corridor_hi = int(control_frames["post_near"])
    corridor_mask = np.zeros(n_frames, dtype=bool)
    corridor_mask[corridor_lo : corridor_hi + 1] = True
    corridor_mask &= ~hit_mask
    return hit_mask, corridor_mask


def _weak_target_frames(control_frames: dict[str, int]) -> tuple[int, int]:
    return int(control_frames["pre_far"]), int(control_frames["post_far"])


def _geometry_support_frames(
    hit_mask: np.ndarray,
    corridor_mask: np.ndarray,
    weak_pre_frame: int,
    weak_post_frame: int,
    n_frames: int,
) -> np.ndarray:
    frames = set(np.flatnonzero(hit_mask).tolist())
    frames.update(np.flatnonzero(corridor_mask).tolist())
    frames.add(int(weak_pre_frame))
    frames.add(int(weak_post_frame))
    support = set()
    for frame in frames:
        support.add(frame)
        support.add(max(0, frame - 1))
        support.add(min(n_frames - 1, frame + 1))
    return np.asarray(sorted(support), dtype=np.int64)


def _spline_sequence(control_q: np.ndarray, control_frames: dict[str, int], n_frames: int) -> np.ndarray:
    x = np.asarray([control_frames[name] for name in CONTROL_POINT_ORDER], dtype=np.float64)
    spline = CubicSpline(x, control_q, axis=0, bc_type="natural")
    frames = np.arange(n_frames, dtype=np.float64)
    return spline(frames).astype(np.float64)


def _control_point_stay_weights(config: dict[str, Any]) -> np.ndarray:
    weights = config["optimization_stay_weights"]
    return np.asarray([float(weights[name]) for name in CONTROL_POINT_ORDER], dtype=np.float64)


def _schema_pass(csv_data: np.ndarray) -> bool:
    return bool(
        csv_data.ndim == 2
        and csv_data.shape[1] == 7 + len(A3_POLICY_JOINT_ORDER)
        and np.isfinite(csv_data).all()
    )


def _source_quality_ok(target_spec: dict[str, Any]) -> bool:
    flags = target_spec.get("quality_flags", {})
    required = (
        "has_finite_data",
        "cleaning_usable",
        "racket_quat_available",
        "table_transform_available",
        "coordinate_transform_available",
        "usable_for_training",
    )
    return all(bool(flags.get(key, False)) for key in required)


def _replay_precheck(
    csv_data: np.ndarray,
    config: dict[str, Any],
    active_names: list[str],
    geometry_pass: bool,
    dynamics_pass: bool,
    hit_index_valid: bool,
) -> dict[str, Any]:
    base_pos = np.asarray(config["robot_base"]["position_m"], dtype=np.float64)
    base_quat = np.asarray(config["robot_base"]["quat_xyzw"], dtype=np.float64)
    lower_body_names = [
        name
        for name in A3_POLICY_JOINT_ORDER
        if any(token in name for token in ("hip_", "knee_", "ankle_"))
    ]
    lower_body_idx = [A3_POLICY_JOINT_ORDER.index(name) for name in lower_body_names]
    finite_ok = bool(np.isfinite(csv_data).all())
    schema_pass = _schema_pass(csv_data)
    base_fixed_ok = bool(
        np.allclose(csv_data[:, :3], base_pos[None, :], atol=1e-8)
        and np.allclose(csv_data[:, 3:7], base_quat[None, :], atol=1e-8)
    )
    lower_body_static_ok = True
    if lower_body_idx:
        lower_body_motion = csv_data[:, 7:][:, lower_body_idx]
        lower_body_static_ok = bool(np.max(np.abs(lower_body_motion - lower_body_motion[:1])) <= 1e-6)
    precheck = {
        "finite_ok": finite_ok,
        "schema_pass": schema_pass,
        "base_fixed_ok": base_fixed_ok,
        "lower_body_static_ok": lower_body_static_ok,
        "racket_frame_ok": True,
        "hit_frame_ok": hit_index_valid,
        "joint_dynamics_ok": dynamics_pass,
        "geometry_pass": geometry_pass,
        "csv_to_npz_compatible": schema_pass and finite_ok,
    }
    precheck["replay_ready"] = all(precheck.values())
    precheck["active_joint_names"] = active_names
    return precheck


def _control_point_delta_penalty(control_q: np.ndarray, control_q_ref: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    stay_weights = _control_point_stay_weights(config)[:, None]
    return (stay_weights * float(config["optimization"]["control_point_delta_weight"]) * (control_q - control_q_ref)).ravel()


def _control_point_smooth_penalty(control_q: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if control_q.shape[0] <= 1:
        return np.zeros(0, dtype=np.float64)
    return (float(config["optimization"]["control_point_smooth_weight"]) * np.diff(control_q, axis=0)).ravel()


def _limit_margin_penalty(q_seq: np.ndarray, lower: np.ndarray, upper: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    mid = 0.5 * (lower + upper)
    half = np.maximum(0.5 * (upper - lower), 1e-6)
    normalized = np.abs((q_seq - mid[None, :]) / half[None, :])
    margin_excess = np.maximum(normalized - 0.85, 0.0)
    return (float(config["optimization"]["limit_margin_weight"]) * margin_excess).ravel()


def _optimize_one(
    *,
    csv_init: np.ndarray,
    target_npz: Path,
    target_spec: dict[str, Any],
    config: dict[str, Any],
    active_names: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.load(target_npz, allow_pickle=False)
    target_pos = target["racket_pos"].astype(np.float64)
    target_quat = target["racket_quat"].astype(np.float64)
    target_vel = target["racket_vel"].astype(np.float64)
    target_normal, target_tangent = _target_axes(target_quat)
    target_vel_dir = _normalize_rows(target_vel)
    hit_index = int(target["hit_index"])
    n_frames = int(csv_init.shape[0])
    dt = 1.0 / float(config["time"]["fps"])
    active_idx = [A3_POLICY_JOINT_ORDER.index(name) for name in active_names]
    q_ref = csv_init[:, 7:][:, active_idx].copy()
    control_frames = _control_frames(hit_index, n_frames, config)
    control_q_ref = np.asarray([q_ref[control_frames[name]] for name in CONTROL_POINT_ORDER], dtype=np.float64)
    hit_mask, corridor_mask = _hit_and_corridor_masks(hit_index, n_frames, control_frames, config["phase_windows"])
    weak_pre_frame, weak_post_frame = _weak_target_frames(control_frames)
    support_frames = _geometry_support_frames(hit_mask, corridor_mask, weak_pre_frame, weak_post_frame, n_frames)
    support_lookup = {int(frame): i for i, frame in enumerate(support_frames.tolist())}
    opt_cfg = config["optimization"]

    def residual(flat_control_q: np.ndarray) -> np.ndarray:
        control_q = flat_control_q.reshape(control_q_ref.shape)
        q_seq = _spline_sequence(control_q, control_frames, n_frames)
        trial = csv_init.copy()
        trial[:, 7:][:, active_idx] = q_seq
        joint_vel, joint_acc = compute_joint_vel_acc(q_seq, dt)
        joint_jerk = _joint_jerk(joint_acc, dt)
        support_pos, support_normal, support_tangent = _racket_series_for_frames(trial, support_frames, base_pos, base_quat)

        res = []
        if np.any(hit_mask):
            hit_frames = np.flatnonzero(hit_mask)
            hit_idx = np.asarray([support_lookup[int(frame)] for frame in hit_frames], dtype=np.int64)
            hit_prev = np.asarray([support_lookup[max(0, int(frame) - 1)] for frame in hit_frames], dtype=np.int64)
            hit_next = np.asarray([support_lookup[min(n_frames - 1, int(frame) + 1)] for frame in hit_frames], dtype=np.int64)
            hit_vel = (support_pos[hit_next] - support_pos[hit_prev]) / max(2.0 * dt, 1e-9)
            hit_vel_dir = _normalize_rows(hit_vel)
            hit_speed = np.linalg.norm(hit_vel, axis=1)
            target_speed = np.linalg.norm(target_vel[hit_frames], axis=1)
            velocity_scale = max(_config_float(opt_cfg, "velocity_magnitude_scale_mps", 2.0), 1e-9)
            res.append((float(opt_cfg["hit_position_weight"]) * (support_pos[hit_idx] - target_pos[hit_frames])).ravel())
            res.append((float(opt_cfg["hit_normal_weight"]) * (support_normal[hit_idx] - target_normal[hit_frames])).ravel())
            res.append((float(opt_cfg["hit_velocity_direction_weight"]) * (hit_vel_dir - target_vel_dir[hit_frames])).ravel())
            res.append((_config_float(opt_cfg, "hit_velocity_magnitude_weight", 0.0) * ((hit_speed - target_speed) / velocity_scale)).ravel())
            res.append((0.3 * float(opt_cfg["hit_normal_weight"]) * (support_tangent[hit_idx] - target_tangent[hit_frames])).ravel())
        if np.any(corridor_mask):
            corridor_frames = np.flatnonzero(corridor_mask)
            corridor_idx = np.asarray([support_lookup[int(frame)] for frame in corridor_frames], dtype=np.int64)
            corridor_prev = np.asarray([support_lookup[max(0, int(frame) - 1)] for frame in corridor_frames], dtype=np.int64)
            corridor_next = np.asarray([support_lookup[min(n_frames - 1, int(frame) + 1)] for frame in corridor_frames], dtype=np.int64)
            corridor_vel_dir = _normalize_rows((support_pos[corridor_next] - support_pos[corridor_prev]) / max(2.0 * dt, 1e-9))
            res.append((float(opt_cfg["near_corridor_position_weight"]) * (support_pos[corridor_idx] - target_pos[corridor_frames])).ravel())
            res.append((float(opt_cfg["near_corridor_normal_weight"]) * (support_normal[corridor_idx] - target_normal[corridor_frames])).ravel())
            res.append((float(opt_cfg["near_corridor_velocity_direction_weight"]) * (corridor_vel_dir - target_vel_dir[corridor_frames])).ravel())
        for frame in (weak_pre_frame, weak_post_frame):
            idx = support_lookup[int(frame)]
            prev_idx = support_lookup[max(0, int(frame) - 1)]
            next_idx = support_lookup[min(n_frames - 1, int(frame) + 1)]
            vel_dir = _normalize_rows(
                ((support_pos[next_idx] - support_pos[prev_idx]) / max(2.0 * dt, 1e-9))[None, :]
            )[0]
            res.append((float(opt_cfg["weak_position_weight"]) * (support_pos[idx] - target_pos[frame])).ravel())
            res.append((float(opt_cfg["weak_velocity_direction_weight"]) * (vel_dir - target_vel_dir[frame])).ravel())

        res.append((float(opt_cfg["regularization_weight"]) * (q_seq - q_ref)).ravel())
        res.append(_control_point_delta_penalty(control_q, control_q_ref, config))
        res.append(_control_point_smooth_penalty(control_q, config))
        res.append(_limit_margin_penalty(q_seq, lower, upper, config))
        res.append((float(opt_cfg["smoothness_weight"]) * joint_vel).ravel())
        res.append((float(opt_cfg["acceleration_weight"]) * joint_acc).ravel())
        res.append((float(opt_cfg["jerk_weight"]) * joint_jerk).ravel())

        vel_excess = np.maximum(np.abs(joint_vel) - float(config["quality_thresholds"]["max_joint_velocity_warning_radps"]), 0.0)
        acc_excess = np.maximum(np.abs(joint_acc) - float(config["quality_thresholds"]["max_joint_acceleration_warning_radps2"]), 0.0)
        jerk_excess = np.maximum(np.abs(joint_jerk) - float(config["quality_thresholds"]["max_joint_jerk_warning_radps3"]), 0.0)
        res.append((0.25 * vel_excess).ravel())
        res.append((0.10 * acc_excess).ravel())
        res.append((0.03 * jerk_excess).ravel())
        return np.concatenate(res)

    bounds_lo = np.tile(lower, len(CONTROL_POINT_ORDER))
    bounds_hi = np.tile(upper, len(CONTROL_POINT_ORDER))
    result = least_squares(
        residual,
        x0=np.clip(control_q_ref, lower[None, :], upper[None, :]).ravel(),
        bounds=(bounds_lo, bounds_hi),
        max_nfev=int(opt_cfg["max_nfev"]),
        verbose=0,
    )
    control_q_opt = result.x.reshape(control_q_ref.shape)
    q_opt = _spline_sequence(control_q_opt, control_frames, n_frames)
    q_opt = np.clip(q_opt, lower[None, :], upper[None, :])
    csv_opt = csv_init.copy()
    csv_opt[:, 7:][:, active_idx] = q_opt
    metrics = {
        "optimizer_cost": float(result.cost),
        "optimizer_nfev": int(result.nfev),
        "optimizer_success": bool(result.success),
        "control_frame_map": {name: int(control_frames[name]) for name in CONTROL_POINT_ORDER},
        "active_joint_count": int(len(active_idx)),
        "source_quality_ok": _source_quality_ok(target_spec),
    }
    return csv_opt, metrics


def _evaluate(
    csv_data: np.ndarray,
    target_npz: Path,
    config: dict[str, Any],
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    active_names: list[str],
) -> dict[str, Any]:
    target = np.load(target_npz, allow_pickle=False)
    hit_index = int(target["hit_index"])
    target_pos = target["racket_pos"].astype(np.float64)
    target_quat = target["racket_quat"].astype(np.float64)
    target_vel = target["racket_vel"].astype(np.float64)
    target_normal, _ = _target_axes(target_quat)
    target_vel_dir = _normalize_rows(target_vel)
    pos, normal, _ = _racket_series(csv_data, base_pos, base_quat)
    vel = np.gradient(pos, 1.0 / float(config["time"]["fps"]), axis=0)
    vel_dir = _normalize_rows(vel)
    speed = np.linalg.norm(vel, axis=1)
    target_speed = np.linalg.norm(target_vel, axis=1)
    speed_err = np.abs(speed - target_speed)
    q = csv_data[:, 7:]
    joint_vel, joint_acc = compute_joint_vel_acc(q, 1.0 / float(config["time"]["fps"]))
    joint_jerk = _joint_jerk(joint_acc, 1.0 / float(config["time"]["fps"]))
    active_idx = [A3_POLICY_JOINT_ORDER.index(name) for name in active_names]
    pos_err_all = np.linalg.norm(pos - target_pos, axis=1)
    normal_cos = np.sum(_normalize_rows(normal) * _normalize_rows(target_normal), axis=1)
    normal_err_deg = np.degrees(np.arccos(np.clip(normal_cos, -1.0, 1.0)))
    vel_cos = np.sum(_normalize_rows(vel_dir) * _normalize_rows(target_vel_dir), axis=1)
    vel_err_deg = np.degrees(np.arccos(np.clip(vel_cos, -1.0, 1.0)))
    return {
        "hit_index": hit_index,
        "sequence_length_frames": int(csv_data.shape[0]),
        "racket_position_error_at_hit_m": float(pos_err_all[hit_index]),
        "racket_position_error_p50_m": float(np.nanpercentile(pos_err_all, 50)),
        "racket_position_error_p90_m": float(np.nanpercentile(pos_err_all, 90)),
        "racket_orientation_error_at_hit_deg": float(normal_err_deg[hit_index]),
        "racket_orientation_error_p90_deg": float(np.nanpercentile(normal_err_deg, 90)),
        "racket_velocity_direction_error_at_hit_deg": float(vel_err_deg[hit_index]),
        "racket_velocity_direction_error_p90_deg": float(np.nanpercentile(vel_err_deg, 90)),
        "racket_speed_at_hit_mps": float(speed[hit_index]),
        "target_racket_speed_at_hit_mps": float(target_speed[hit_index]),
        "racket_speed_error_at_hit_mps": float(speed_err[hit_index]),
        "racket_speed_error_p90_mps": float(np.nanpercentile(speed_err, 90)),
        "max_active_joint_velocity_radps": float(np.max(np.abs(joint_vel[:, active_idx]))),
        "max_active_joint_acceleration_radps2": float(np.max(np.abs(joint_acc[:, active_idx]))),
        "max_active_joint_jerk_radps3": float(np.max(np.abs(joint_jerk[:, active_idx]))),
    }


def _quality_layers(metrics: dict[str, Any], csv_data: np.ndarray, config: dict[str, Any], active_names: list[str]) -> dict[str, Any]:
    thresholds = config["quality_thresholds"]
    hit_index_valid = 0 <= int(metrics["hit_index"]) < int(metrics["sequence_length_frames"])
    hit_position_pass = metrics["racket_position_error_at_hit_m"] <= float(thresholds["hit_position_reject_m"])
    hit_orientation_pass = metrics["racket_orientation_error_at_hit_deg"] <= float(thresholds["hit_orientation_reject_deg"])
    hit_velocity_direction_pass = metrics["racket_velocity_direction_error_at_hit_deg"] <= float(thresholds["velocity_direction_reject_deg"])
    hit_velocity_magnitude_pass = metrics["racket_speed_error_at_hit_mps"] <= _config_float(
        thresholds,
        "velocity_magnitude_reject_mps",
        float("inf"),
    )
    geometry_pass = bool(hit_position_pass and hit_orientation_pass and hit_velocity_direction_pass and hit_velocity_magnitude_pass)
    dynamics_pass = bool(
        metrics["max_active_joint_velocity_radps"] <= float(thresholds["max_joint_velocity_reject_radps"])
        and metrics["max_active_joint_acceleration_radps2"] <= float(thresholds["max_joint_acceleration_reject_radps2"])
        and metrics["max_active_joint_jerk_radps3"] <= float(thresholds["max_joint_jerk_reject_radps3"])
    )
    schema_pass = _schema_pass(csv_data)
    precheck = _replay_precheck(csv_data, config, active_names, geometry_pass, dynamics_pass, hit_index_valid)
    return {
        "geometry_pass": geometry_pass,
        "dynamics_pass": dynamics_pass,
        "schema_pass": schema_pass,
        "hit_position_pass": hit_position_pass,
        "hit_orientation_pass": hit_orientation_pass,
        "hit_velocity_direction_pass": hit_velocity_direction_pass,
        "hit_velocity_magnitude_pass": hit_velocity_magnitude_pass,
        "replay_precheck": precheck,
        "replay_ready": bool(precheck["replay_ready"]),
    }


def _classify_failure(item: dict[str, Any], target_spec: dict[str, Any], layers: dict[str, Any]) -> str:
    if not _source_quality_ok(target_spec):
        return "bad_source_data"
    if item.get("ik_status") != "pass":
        return "fixed_base_reach_fail"
    if not layers["geometry_pass"]:
        return "fixed_base_hit_pose_fail"
    if not layers["dynamics_pass"]:
        return "fixed_base_dynamic_fail"
    if not layers["schema_pass"]:
        return "schema_fail"
    return "fixed_base_pass"


def _reject_reasons(metrics: dict[str, Any], layers: dict[str, Any], config: dict[str, Any]) -> list[str]:
    thresholds = config["quality_thresholds"]
    reasons = []
    if not layers["hit_position_pass"]:
        reasons.append("hit_position_error")
    if not layers["hit_orientation_pass"]:
        reasons.append("hit_orientation_error")
    if not layers["hit_velocity_direction_pass"]:
        reasons.append("hit_velocity_direction_error")
    if not layers["hit_velocity_magnitude_pass"]:
        reasons.append("hit_velocity_magnitude_error")
    if metrics["max_active_joint_velocity_radps"] > float(thresholds["max_joint_velocity_reject_radps"]):
        reasons.append("joint_velocity")
    if metrics["max_active_joint_acceleration_radps2"] > float(thresholds["max_joint_acceleration_reject_radps2"]):
        reasons.append("joint_acceleration")
    if metrics["max_active_joint_jerk_radps3"] > float(thresholds["max_joint_jerk_reject_radps3"]):
        reasons.append("joint_jerk")
    if not layers["schema_pass"]:
        reasons.append("schema")
    if not layers["replay_precheck"]["base_fixed_ok"]:
        reasons.append("base_not_fixed")
    if not layers["replay_precheck"]["lower_body_static_ok"]:
        reasons.append("lower_body_motion")
    return reasons


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# A3 P2 Fixed-Base Trajectory Optimization v1",
        "",
        f"- processed: `{report['processed']}`",
        f"- replay ready: `{report['replay_ready']}`",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in sorted(report["status_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Fail Categories", ""])
    for key, value in sorted(report["fail_category_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- manifest: `{report['manifest']}`")
    lines.append(f"- quality dir: `{report['quality_dir']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(str(config["output_root"]))
    manifest_path = args.manifest or output_root / "ik_init_manifest.json"
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = [item for item in manifest["samples"] if item.get("ik_status") == "pass"]
    if args.limit is not None:
        samples = samples[: max(0, args.limit)]

    opt_dir = output_root / "optimized_csv"
    quality_dir = output_root / "optimized_quality_reports"
    opt_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    limits = load_a3_joint_limits()
    active_names = [str(x) for x in config["ik"]["active_joints"]]
    lower = np.asarray([limits[name][0] for name in active_names], dtype=np.float64)
    upper = np.asarray([limits[name][1] for name in active_names], dtype=np.float64)
    base_pos = np.asarray(config["robot_base"]["position_m"], dtype=np.float64)
    base_quat = np.asarray(config["robot_base"]["quat_xyzw"], dtype=np.float64)

    entries = []
    status_counts = Counter()
    fail_category_counts = Counter()
    replay_ready_count = 0
    for item in samples:
        episode_id = str(item["episode_id"])
        csv_init = load_retarget_csv(item["ik_init_csv"])
        target_npz = Path(item["target_npz"])
        target_spec = json.loads(Path(item["target_spec_json"]).read_text(encoding="utf-8"))
        csv_opt, opt_metrics = _optimize_one(
            csv_init=csv_init,
            target_npz=target_npz,
            target_spec=target_spec,
            config=config,
            active_names=active_names,
            lower=lower,
            upper=upper,
            base_pos=base_pos,
            base_quat=base_quat,
        )
        metrics = _evaluate(csv_opt, target_npz, config, base_pos, base_quat, active_names)
        metrics.update(opt_metrics)
        layers = _quality_layers(metrics, csv_opt, config, active_names)
        fail_category = _classify_failure(item, target_spec, layers)
        reject_reasons = _reject_reasons(metrics, layers, config)
        status = "pass" if layers["replay_ready"] else "reject"
        replay_ready_count += int(layers["replay_ready"])
        metrics.update(layers)
        metrics.update(
            {
                "episode_id": episode_id,
                "status": status,
                "reject_reasons": reject_reasons,
                "fail_category": fail_category,
                "ik_init_csv": item["ik_init_csv"],
                "target_spec_json": item["target_spec_json"],
            }
        )
        opt_path = opt_dir / f"{episode_id}.csv"
        quality_path = quality_dir / f"{episode_id}.json"
        write_retarget_csv(opt_path, csv_opt)
        quality_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        entries.append(
            {
                **item,
                "optimized_csv": str(opt_path),
                "optimized_quality_report": str(quality_path),
                "optimized_status": status,
                "replay_ready": bool(layers["replay_ready"]),
                "fail_category": fail_category,
            }
        )
        status_counts[status] += 1
        fail_category_counts[fail_category] += 1

    out_manifest = {
        **manifest,
        "stage": "a3_fixed_base_trajectory_optimized_v1",
        "processed_count": len(entries),
        "replay_ready_count": replay_ready_count,
        "samples": entries,
    }
    out_manifest_path = output_root / "optimized_manifest.json"
    out_manifest_path.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "processed": len(entries),
        "replay_ready": replay_ready_count,
        "status_counts": dict(status_counts),
        "fail_category_counts": dict(fail_category_counts),
        "manifest": str(out_manifest_path),
        "quality_dir": str(quality_dir),
    }
    report_path = output_root / "optimized_summary.md"
    _write_markdown(report, report_path)
    print(f"Processed {len(entries)} optimized targets")
    print(dict(status_counts))
    print(dict(fail_category_counts))
    print(f"replay_ready={replay_ready_count}")
    print(f"Wrote {out_manifest_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
