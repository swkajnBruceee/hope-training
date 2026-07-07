"""A3 refinement solver v0.2.

This module implements the first executable solver stage after the refinement
contract:

- validate generic retarget CSV schema
- load A3 joint limits from the prepared URDF
- optimize active joints over the refinement window with smoothness and
  stay-close-to-init regularization
- keep locked joints fixed

The solver now uses the A3 URDF-based racket FK with phase-weighted racket
tracking terms over the pre-hit / hit / post-hit window, plus torso/arm
regularization against the generic retarget initialization.
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from analysis.mocap_cleaning.a3_metadata import A3_POLICY_JOINT_ORDER


@dataclass
class SolverResult:
    status: str
    warnings: list[str]
    reject_reasons: list[str]
    metrics: dict[str, Any]


def _parse_vec(text: str | None, size: int) -> np.ndarray:
    if text is None:
        return np.zeros(size, dtype=np.float64)
    values = [float(x) for x in text.split()]
    if len(values) != size:
        raise ValueError(f"expected vector of size {size}, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    rr, pp, yy = rpy
    cr, sr = np.cos(rr), np.sin(rr)
    cp, sp = np.cos(pp), np.sin(pp)
    cy, sy = np.cos(yy), np.sin(yy)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.asarray(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
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


def _compose_transform(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


@dataclass(frozen=True)
class ChainJoint:
    name: str
    joint_type: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]


@lru_cache(maxsize=1)
def load_a3_racket_chain() -> list[ChainJoint]:
    urdf_path = _resolved_path("hope_training/whole_body_tracking/training/assets/agibot_a3/urdf/model.urdf")
    root = ET.parse(urdf_path).getroot()
    child_to_joint: dict[str, ChainJoint | tuple[str, str]] = {}
    child_to_parent: dict[str, str] = {}
    child_to_name: dict[str, str] = {}
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        origin = joint.find("origin")
        axis = joint.find("axis")
        child_to_joint[child] = ChainJoint(
            name=str(joint.get("name")),
            joint_type=str(joint.get("type")),
            origin_xyz=tuple(_parse_vec(origin.get("xyz") if origin is not None else None, 3)),
            origin_rpy=tuple(_parse_vec(origin.get("rpy") if origin is not None else None, 3)),
            axis_xyz=tuple(_parse_vec(axis.get("xyz") if axis is not None else "0 0 1", 3)),
        )
        child_to_parent[child] = parent
        child_to_name[child] = str(joint.get("name"))

    target = "pingpang_red_Link"
    chain: list[ChainJoint] = []
    cur = target
    while cur in child_to_joint:
        joint = child_to_joint[cur]
        chain.append(joint)
        cur = child_to_parent[cur]
    chain.reverse()
    return chain


def _resolved_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"path not found: {candidate}")


@lru_cache(maxsize=1)
def load_a3_joint_limits() -> dict[str, tuple[float, float]]:
    urdf_path = _resolved_path("hope_training/whole_body_tracking/training/assets/agibot_a3/urdf/model.urdf")
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = float(limit.get("lower", "-inf"))
        upper = float(limit.get("upper", "inf"))
        limits[str(joint.get("name"))] = (lower, upper)
    return limits


def load_retarget_csv(path: str | Path) -> np.ndarray:
    csv_path = _resolved_path(path)
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    return data


def load_clean_sample_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz_path = _resolved_path(path)
    data = np.load(npz_path, allow_pickle=False)
    return {name: data[name] for name in data.files}


def write_retarget_csv(path: str | Path, data: np.ndarray) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv_path, data, delimiter=",", fmt="%.10f")


def validate_retarget_csv_schema(data: np.ndarray, spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_cols = 7 + len(spec["a3_joint_order"])
    if data.ndim != 2:
        errors.append("csv: expected 2D array")
        return errors
    if data.shape[1] != expected_cols:
        errors.append(f"csv: expected {expected_cols} columns, got {data.shape[1]}")
    seq_len = int(spec["coordinate_contract"]["sequence_length_frames"])
    if data.shape[0] != seq_len:
        errors.append(f"csv: expected {seq_len} frames, got {data.shape[0]}")
    if not np.isfinite(data).all():
        errors.append("csv: non-finite values")
    return errors


def compute_joint_vel_acc(q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    vel = np.gradient(q, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def _window_slice(spec: dict[str, Any]) -> slice:
    hit_win = spec["windows"]["hit"]
    pre_win = spec["windows"]["pre_hit"]
    post_win = spec["windows"]["post_hit"]
    start = int(pre_win["frame_start"])
    end = int(post_win["frame_end"]) + 1
    hit_start = int(hit_win["frame_start"])
    hit_end = int(hit_win["frame_end"]) + 1
    return slice(start, end), slice(hit_start - start, hit_end - start)


def _phase_frame_masks(spec: dict[str, Any], window: slice) -> dict[str, np.ndarray]:
    start = int(window.start)
    end = int(window.stop)
    size = end - start
    masks: dict[str, np.ndarray] = {}
    for name in ("pre_hit", "hit", "post_hit"):
        block = spec["windows"][name]
        mask = np.zeros(size, dtype=bool)
        lo = max(0, int(block["frame_start"]) - start)
        hi = min(size, int(block["frame_end"]) - start + 1)
        if lo < hi:
            mask[lo:hi] = True
        masks[name] = mask
    return masks


def _active_joint_indices(spec: dict[str, Any]) -> list[int]:
    joint_order = spec["a3_joint_order"]
    return [joint_order.index(name) for name in spec["joint_masks"]["active_joints_first_pass"]]


def _locked_joint_indices(spec: dict[str, Any]) -> list[int]:
    joint_order = spec["a3_joint_order"]
    return [joint_order.index(name) for name in spec["joint_masks"]["locked_joints_first_pass"]]


def _fk_racket_state(
    base_pos_xyz: np.ndarray,
    base_quat_xyzw: np.ndarray,
    joint_angles: np.ndarray,
    joint_index_by_name: dict[str, int],
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    chain = load_a3_racket_chain()
    T = _compose_transform(_quat_xyzw_to_matrix(base_quat_xyzw), base_pos_xyz)
    for item in chain:
        T = T @ _compose_transform(_rpy_matrix(np.asarray(item.origin_rpy)), np.asarray(item.origin_xyz))
        if item.joint_type != "fixed":
            q = float(joint_angles[joint_index_by_name[item.name]])
            T = T @ _compose_transform(_axis_angle_matrix(np.asarray(item.axis_xyz), q), np.zeros(3, dtype=np.float64))
    rot = T[:3, :3]
    pos = T[:3, 3]
    return pos, rot


def _compute_racket_series(
    csv_data: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_index_by_name = {name: i for i, name in enumerate(spec["a3_joint_order"])}
    pos = []
    normal = []
    tangent = []
    normal_axis = int(spec["a3_bodies"]["racket_normal_axis"])
    tangent_axis = int(spec["a3_bodies"]["racket_tangent_axis"])
    normal_sign = float(spec["a3_bodies"]["racket_normal_sign"])
    for row in csv_data:
        p, r = _fk_racket_state(row[:3], row[3:7], row[7:], joint_index_by_name, spec)
        pos.append(p)
        normal.append(r[:, normal_axis] * normal_sign)
        tangent.append(r[:, tangent_axis])
    return np.asarray(pos), np.asarray(normal), np.asarray(tangent)


def _target_racket_series(spec: dict[str, Any], window: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample = load_clean_sample_npz(spec["inputs"]["source_sample_npz"])
    pos = sample["racket_pos"][window].astype(np.float64)
    quat = sample["racket_quat"][window].astype(np.float64)
    vel = sample["racket_vel"][window].astype(np.float64)
    normal_axis = int(spec["a3_bodies"]["racket_normal_axis"])
    tangent_axis = int(spec["a3_bodies"]["racket_tangent_axis"])
    normal_sign = float(spec["a3_bodies"]["racket_normal_sign"])
    normal = []
    tangent = []
    for q in quat:
        rot = _quat_xyzw_to_matrix(q)
        normal.append(rot[:, normal_axis] * normal_sign)
        tangent.append(rot[:, tangent_axis])
    vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)
    vel_dir = np.divide(vel, np.maximum(vel_norm, 1e-9))
    return pos, np.asarray(normal), np.asarray(tangent), vel_dir


def _joint_group_indices(spec: dict[str, Any], active_idx: list[int]) -> tuple[list[int], list[int]]:
    joint_order = spec["a3_joint_order"]
    torso_names = {"waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"}
    torso_local = []
    arm_local = []
    for local_i, joint_i in enumerate(active_idx):
        if joint_order[joint_i] in torso_names:
            torso_local.append(local_i)
        else:
            arm_local.append(local_i)
    return torso_local, arm_local


def _interpolate_anchor_values(anchor_values: np.ndarray, anchor_frames: np.ndarray, num_frames: int) -> np.ndarray:
    out = np.zeros((num_frames, anchor_values.shape[1]), dtype=np.float64)
    for seg in range(len(anchor_frames) - 1):
        start = int(anchor_frames[seg])
        end = int(anchor_frames[seg + 1])
        if end <= start:
            out[start] = anchor_values[seg]
            continue
        for t in range(start, end + 1):
            alpha = np.clip((t - start) / float(end - start), 0.0, 1.0)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            out[t] = (1.0 - alpha) * anchor_values[seg] + alpha * anchor_values[seg + 1]
    if int(anchor_frames[0]) > 0:
        out[: int(anchor_frames[0])] = anchor_values[0]
    if int(anchor_frames[-1]) < num_frames:
        out[int(anchor_frames[-1]) :] = anchor_values[-1]
    return out


def _build_residual(
    flat_anchor_deltas: np.ndarray,
    q_anchor_init: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    fixed_csv_window: np.ndarray,
    active_idx: list[int],
    target_window: dict[str, np.ndarray],
    phase_masks: dict[str, np.ndarray],
    torso_local_idx: list[int],
    arm_local_idx: list[int],
    anchor_frames: np.ndarray,
    q_ref_window: np.ndarray,
    spec: dict[str, Any],
) -> np.ndarray:
    res = []
    anchor_deltas = flat_anchor_deltas.reshape(q_anchor_init.shape)
    anchor_q = np.clip(q_anchor_init + anchor_deltas, lower[None, :], upper[None, :])
    q_hit = anchor_q[2]
    mid = 0.5 * (lower + upper)
    half = np.maximum(0.5 * (upper - lower), 1e-6)
    res.append(0.02 * anchor_deltas.ravel())
    res.append(0.015 * np.diff(anchor_deltas, axis=0).ravel())
    res.append(0.02 * ((q_hit - mid) / half).ravel())

    phase_cfg = spec["phase_weights"]
    q_window = _interpolate_anchor_values(anchor_q, anchor_frames, fixed_csv_window.shape[0])
    q_window = np.clip(q_window, lower[None, :], upper[None, :])

    trial_csv = fixed_csv_window.copy()
    trial_csv[:, 7:][:, active_idx] = q_window
    racket_pos, racket_normal, racket_tangent = _compute_racket_series(trial_csv, spec)
    dt = float(spec["coordinate_contract"]["dt"])
    vel = np.gradient(racket_pos, dt, axis=0)
    vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)
    vel_dir = np.divide(vel, np.maximum(vel_norm, 1e-9))

    for phase_name, mask in phase_masks.items():
        if not np.any(mask):
            continue
        weights = phase_cfg[phase_name]["weights"]
        res.append((weights["racket_pose"] * (racket_pos[mask] - target_window["pos"][mask])).ravel())
        res.append((0.6 * weights["racket_pose"] * (racket_normal[mask] - target_window["normal"][mask])).ravel())
        res.append((0.3 * weights["racket_pose"] * (racket_tangent[mask] - target_window["tangent"][mask])).ravel())
        res.append((weights["racket_velocity"] * (vel_dir[mask] - target_window["vel_dir"][mask])).ravel())

        if torso_local_idx:
            torso_q = q_window[mask][:, torso_local_idx]
            torso_ref = q_ref_window[mask][:, torso_local_idx]
            res.append((weights["torso_support"] * 0.15 * (torso_q - torso_ref)).ravel())
        if arm_local_idx:
            arm_q = q_window[mask][:, arm_local_idx]
            arm_ref = q_ref_window[mask][:, arm_local_idx]
            res.append((weights["arm_posture"] * 0.08 * (arm_q - arm_ref)).ravel())

    if q_window.shape[0] > 1:
        q_vel = np.diff(q_window, axis=0) / max(float(spec["coordinate_contract"]["dt"]), 1e-9)
        res.append((0.02 * q_vel).ravel())
        vel_limit_target = 8.0
        vel_excess = np.maximum(np.abs(q_vel) - vel_limit_target, 0.0)
        res.append((0.15 * vel_excess).ravel())
    if q_window.shape[0] > 2:
        q_acc = (q_window[2:] - 2.0 * q_window[1:-1] + q_window[:-2]) / max(float(spec["coordinate_contract"]["dt"]) ** 2, 1e-9)
        res.append((0.005 * q_acc).ravel())
        acc_limit_target = 60.0
        acc_excess = np.maximum(np.abs(q_acc) - acc_limit_target, 0.0)
        res.append((0.12 * acc_excess).ravel())
    return np.concatenate(res, axis=0)


def refine_csv_v0(spec: dict[str, Any], csv_data: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    joint_block = csv_data[:, 7:].copy()
    limits = load_a3_joint_limits()
    active_idx = _active_joint_indices(spec)
    locked_idx = _locked_joint_indices(spec)
    dt = float(spec["coordinate_contract"]["dt"])
    window, hit_local = _window_slice(spec)
    q_init_window = joint_block[window][:, active_idx]
    fixed_csv_window = csv_data[window].copy()
    hit_frame_local = int(np.clip(int(spec["hit_target"]["hit_index"]) - int(window.start), 0, q_init_window.shape[0] - 1))
    pre_anchor_local = max(0, int(0.5 * (spec["windows"]["pre_hit"]["frame_start"] + spec["windows"]["pre_hit"]["frame_end"])) - int(window.start))
    post_anchor_local = min(q_init_window.shape[0] - 1, int(0.5 * (spec["windows"]["post_hit"]["frame_start"] + spec["windows"]["post_hit"]["frame_end"])) - int(window.start))
    anchor_frames = np.asarray([0, pre_anchor_local, hit_frame_local, post_anchor_local, q_init_window.shape[0] - 1], dtype=int)
    q_anchor_init = q_init_window[anchor_frames].copy()
    phase_masks = _phase_frame_masks(spec, window)
    target_pos, target_normal, target_tangent, target_vel_dir = _target_racket_series(spec, window)
    target_window = {
        "pos": target_pos,
        "normal": target_normal,
        "tangent": target_tangent,
        "vel_dir": target_vel_dir,
    }
    torso_local_idx, arm_local_idx = _joint_group_indices(spec, active_idx)

    lower = np.asarray([limits[spec["a3_joint_order"][i]][0] for i in active_idx], dtype=np.float64)
    upper = np.asarray([limits[spec["a3_joint_order"][i]][1] for i in active_idx], dtype=np.float64)

    init_violation = np.maximum(lower - q_init_window, 0.0) + np.maximum(q_init_window - upper, 0.0)
    max_violation_before = float(np.max(init_violation)) if init_violation.size else 0.0

    q_anchor_init = np.clip(q_anchor_init, lower[None, :], upper[None, :])
    lb = (lower[None, :] - q_anchor_init).ravel()
    ub = (upper[None, :] - q_anchor_init).ravel()
    result = least_squares(
        fun=_build_residual,
        x0=np.zeros_like(q_anchor_init).ravel(),
        bounds=(lb, ub),
        args=(
            q_anchor_init,
            lower,
            upper,
            fixed_csv_window,
            active_idx,
            target_window,
            phase_masks,
            torso_local_idx,
            arm_local_idx,
            anchor_frames,
            q_init_window,
            spec,
        ),
        max_nfev=24,
        verbose=0,
    )

    anchor_deltas = result.x.reshape(q_anchor_init.shape)
    anchor_q = np.clip(q_anchor_init + anchor_deltas, lower[None, :], upper[None, :])
    refined_window = _interpolate_anchor_values(anchor_q, anchor_frames, q_init_window.shape[0])
    refined_window = np.clip(refined_window, lower[None, :], upper[None, :])
    joint_block_refined = joint_block.copy()
    joint_block_refined[window][:, active_idx] = refined_window
    # keep locked joints exactly at init in the window
    if locked_idx:
        joint_block_refined[window][:, locked_idx] = joint_block[window][:, locked_idx]

    refined_csv = csv_data.copy()
    refined_csv[:, 7:] = joint_block_refined

    thresholds = spec["quality_thresholds"]["warning"]
    init_vel, init_acc = compute_joint_vel_acc(joint_block, dt)
    init_racket_pos, init_racket_normal, _ = _compute_racket_series(csv_data, spec)
    init_racket_vel = np.gradient(init_racket_pos, dt, axis=0)
    vel, acc = compute_joint_vel_acc(joint_block_refined, dt)
    racket_pos, racket_normal, _ = _compute_racket_series(refined_csv, spec)
    racket_vel = np.gradient(racket_pos, dt, axis=0)
    target_pos = np.asarray(spec["hit_target"]["racket_position_m"], dtype=np.float64)
    target_normal = np.asarray(spec["hit_target"]["racket_normal_w"], dtype=np.float64)
    target_vel_dir = np.asarray(spec["hit_target"]["racket_velocity_direction_w"], dtype=np.float64)
    hit_index = int(spec["hit_target"]["hit_index"])
    init_pos_err = float(np.linalg.norm(init_racket_pos[hit_index] - target_pos))
    init_normal_cos = float(np.clip(np.dot(init_racket_normal[hit_index], target_normal) / (np.linalg.norm(init_racket_normal[hit_index]) * np.linalg.norm(target_normal) + 1e-9), -1.0, 1.0))
    init_normal_err_deg = float(np.degrees(np.arccos(init_normal_cos)))
    init_vel_dir = init_racket_vel[hit_index] / max(float(np.linalg.norm(init_racket_vel[hit_index])), 1e-9)
    init_vel_cos = float(np.clip(np.dot(init_vel_dir, target_vel_dir) / (np.linalg.norm(target_vel_dir) + 1e-9), -1.0, 1.0))
    init_vel_dir_err_deg = float(np.degrees(np.arccos(init_vel_cos)))
    pos_err = float(np.linalg.norm(racket_pos[hit_index] - target_pos))
    normal_cos = float(np.clip(np.dot(racket_normal[hit_index], target_normal) / (np.linalg.norm(racket_normal[hit_index]) * np.linalg.norm(target_normal) + 1e-9), -1.0, 1.0))
    normal_err_deg = float(np.degrees(np.arccos(normal_cos)))
    vel_dir = racket_vel[hit_index] / max(float(np.linalg.norm(racket_vel[hit_index])), 1e-9)
    vel_cos = float(np.clip(np.dot(vel_dir, target_vel_dir) / (np.linalg.norm(target_vel_dir) + 1e-9), -1.0, 1.0))
    vel_dir_err_deg = float(np.degrees(np.arccos(vel_cos)))

    init_max_vel = float(np.max(np.abs(init_vel))) if init_vel.size else 0.0
    init_max_acc = float(np.max(np.abs(init_acc))) if init_acc.size else 0.0
    refined_max_vel = float(np.max(np.abs(vel))) if vel.size else 0.0
    refined_max_acc = float(np.max(np.abs(acc))) if acc.size else 0.0

    def _score(pos_err_v: float, normal_err_v: float, vel_dir_err_v: float, max_vel_v: float, max_acc_v: float) -> float:
        vel_excess = max(0.0, max_vel_v - thresholds["max_joint_velocity_radps_max"]) / max(thresholds["max_joint_velocity_radps_max"], 1e-9)
        acc_excess = max(0.0, max_acc_v - thresholds["max_joint_acceleration_radps2_max"]) / max(thresholds["max_joint_acceleration_radps2_max"], 1e-9)
        return (
            pos_err_v
            + 0.02 * normal_err_v
            + 0.01 * vel_dir_err_v
            + 0.25 * vel_excess
            + 0.15 * acc_excess
        )

    init_score = _score(init_pos_err, init_normal_err_deg, init_vel_dir_err_deg, init_max_vel, init_max_acc)
    refined_score = _score(pos_err, normal_err_deg, vel_dir_err_deg, refined_max_vel, refined_max_acc)
    kept_baseline = False
    if refined_score > init_score:
        refined_csv = csv_data.copy()
        joint_block_refined = joint_block.copy()
        vel = init_vel
        acc = init_acc
        pos_err = init_pos_err
        normal_err_deg = init_normal_err_deg
        vel_dir_err_deg = init_vel_dir_err_deg
        refined_max_vel = init_max_vel
        refined_max_acc = init_max_acc
        kept_baseline = True

    def _joint_peak_info(values: np.ndarray, phase_masks_local: dict[str, np.ndarray]) -> tuple[float, int, str]:
        flat_idx = int(np.argmax(np.abs(values)))
        frame_idx, joint_local = np.unravel_index(flat_idx, values.shape)
        phase = "boundary"
        for name, mask in phase_masks_local.items():
            if 0 <= frame_idx < mask.shape[0] and mask[frame_idx]:
                phase = name
                break
        return float(np.abs(values[frame_idx, joint_local])), int(frame_idx), phase

    init_vel_peak, init_vel_frame, init_vel_phase = _joint_peak_info(init_vel[:, active_idx], phase_masks)
    init_acc_peak, init_acc_frame, init_acc_phase = _joint_peak_info(init_acc[:, active_idx], phase_masks)
    refined_vel_peak, refined_vel_frame, refined_vel_phase = _joint_peak_info(vel[:, active_idx], phase_masks)
    refined_acc_peak, refined_acc_frame, refined_acc_phase = _joint_peak_info(acc[:, active_idx], phase_masks)
    init_vel_joint_name = spec["joint_masks"]["active_joints_first_pass"][int(np.argmax(np.max(np.abs(init_vel[:, active_idx]), axis=0)))]
    init_acc_joint_name = spec["joint_masks"]["active_joints_first_pass"][int(np.argmax(np.max(np.abs(init_acc[:, active_idx]), axis=0)))]
    refined_vel_joint_name = spec["joint_masks"]["active_joints_first_pass"][int(np.argmax(np.max(np.abs(vel[:, active_idx]), axis=0)))]
    refined_acc_joint_name = spec["joint_masks"]["active_joints_first_pass"][int(np.argmax(np.max(np.abs(acc[:, active_idx]), axis=0)))]

    metrics = {
        "optimizer_cost": float(result.cost),
        "optimizer_nfev": int(result.nfev),
        "optimizer_status": int(result.status),
        "optimizer_success": bool(result.success),
        "kept_generic_init_baseline": kept_baseline,
        "baseline_score": init_score,
        "refined_score": refined_score,
        "window_frame_start": int(window.start),
        "window_frame_end": int(window.stop - 1),
        "active_joint_count": len(active_idx),
        "locked_joint_count": len(locked_idx),
        "max_joint_limit_violation_before_clamp_rad": max_violation_before,
        "max_joint_velocity_radps": refined_max_vel,
        "max_joint_acceleration_radps2": refined_max_acc,
        "max_velocity_joint": refined_vel_joint_name,
        "max_velocity_frame": refined_vel_frame,
        "max_velocity_phase": refined_vel_phase,
        "max_acceleration_joint": refined_acc_joint_name,
        "max_acceleration_frame": refined_acc_frame,
        "max_acceleration_phase": refined_acc_phase,
        "generic_init_max_vel": init_vel_peak,
        "generic_init_max_vel_joint": init_vel_joint_name,
        "generic_init_max_vel_frame": init_vel_frame,
        "generic_init_max_vel_phase": init_vel_phase,
        "generic_init_max_acc": init_acc_peak,
        "generic_init_max_acc_joint": init_acc_joint_name,
        "generic_init_max_acc_frame": init_acc_frame,
        "generic_init_max_acc_phase": init_acc_phase,
        "refined_max_vel": refined_vel_peak,
        "refined_max_vel_joint": refined_vel_joint_name,
        "refined_max_vel_frame": refined_vel_frame,
        "refined_max_vel_phase": refined_vel_phase,
        "refined_max_acc": refined_acc_peak,
        "refined_max_acc_joint": refined_acc_joint_name,
        "refined_max_acc_frame": refined_acc_frame,
        "refined_max_acc_phase": refined_acc_phase,
        "ik_residual_rms": float(np.sqrt(max(result.cost, 0.0) / max(result.fun.size, 1))),
        "racket_position_error_at_hit_m": pos_err,
        "racket_orientation_error_at_hit_deg": normal_err_deg,
        "racket_velocity_direction_error_at_hit_deg": vel_dir_err_deg,
        "csv_to_npz_passed": None,
    }
    return refined_csv, metrics


def run_refine_mode(spec: dict[str, Any], write_metrics: bool) -> SolverResult:
    warnings: list[str] = []
    reject_reasons: list[str] = []
    generic_csv_path = Path(spec["artifacts"]["generic_retarget_csv"])
    refined_csv_path = Path(spec["artifacts"]["refined_retarget_csv"])
    metrics: dict[str, Any] = {
        "job_id": spec["job_id"],
        "mode": "refine",
        "validation_status": None,
        "warnings": warnings,
        "reject_reasons": reject_reasons,
    }

    if not generic_csv_path.exists():
        reject_reasons.append("generic_retarget_csv_missing")
        metrics["status"] = "skipped_missing_generic_retarget_csv"
        return SolverResult(metrics["status"], warnings, reject_reasons, metrics)

    csv_data = load_retarget_csv(generic_csv_path)
    schema_errors = validate_retarget_csv_schema(csv_data, spec)
    if schema_errors:
        reject_reasons.extend(schema_errors)
        metrics["status"] = "failed_csv_schema_validation"
        return SolverResult(metrics["status"], warnings, reject_reasons, metrics)

    refined_csv, solve_metrics = refine_csv_v0(spec, csv_data)
    write_retarget_csv(refined_csv_path, refined_csv)
    metrics.update(solve_metrics)
    warnings.append("solver_v0_uses_racket_fk_objectives_only_in_hit_window")
    warnings.append("solver_v0_does_not_yet_track_full_wrist_elbow_torso_reference_terms")
    thresholds = spec["quality_thresholds"]
    warning_hits = []
    reject_hits = []
    metric_map = {
        "racket_position_error_at_hit_m_max": "racket_position_error_at_hit_m",
        "racket_orientation_error_at_hit_deg_max": "racket_orientation_error_at_hit_deg",
        "racket_velocity_direction_error_at_hit_deg_max": "racket_velocity_direction_error_at_hit_deg",
        "ik_residual_rms_max": "ik_residual_rms",
        "max_joint_limit_violation_before_clamp_rad_max": "max_joint_limit_violation_before_clamp_rad",
        "max_joint_velocity_radps_max": "max_joint_velocity_radps",
        "max_joint_acceleration_radps2_max": "max_joint_acceleration_radps2",
    }
    for threshold_key, metric_key in metric_map.items():
        value = metrics.get(metric_key)
        if value is None:
            continue
        if value > thresholds["warning"][threshold_key]:
            warning_hits.append(f"{metric_key}>{thresholds['warning'][threshold_key]}")
        if value > thresholds["reject"][threshold_key]:
            reject_hits.append(f"{metric_key}>{thresholds['reject'][threshold_key]}")
    if not bool(metrics.get("optimizer_success", False)):
        warning_hits.append("optimizer_nonconverged")
    metrics["validation_warnings"] = warning_hits
    metrics["validation_reject_reasons"] = reject_hits
    if reject_hits:
        metrics["status"] = "failed_quality_reject"
        metrics["validation_status"] = "reject"
        reject_reasons.extend(reject_hits)
    elif warning_hits:
        metrics["status"] = "passed_refine_v0_with_warnings"
        metrics["validation_status"] = "warning"
        warnings.extend(warning_hits)
    else:
        metrics["status"] = "passed_refine_v0"
        metrics["validation_status"] = "pass"
    if write_metrics:
        quality_report = Path(spec["artifacts"]["quality_report_json"])
        quality_report.parent.mkdir(parents=True, exist_ok=True)
        quality_report.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    return SolverResult(metrics["status"], warnings, reject_reasons, metrics)
