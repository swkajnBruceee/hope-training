#!/usr/bin/env python3
"""Generate a heuristic generic retarget CSV from source BVH windows."""

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
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from analysis.mocap_cleaning.a3_metadata import (
    A3_ACTIVE_JOINTS_FIRST_PASS,
    A3_DEFAULT_JOINT_POS,
    A3_POLICY_JOINT_ORDER,
)
from analysis.mocap_cleaning.bvh_motion import load_bvh, sample_joint_channels_at_times
from analysis.mocap_cleaning.refinement_spec import resolve_existing_path
from analysis.mocap_cleaning.a3_refinement_solver import _fk_racket_state, load_a3_joint_limits


def _deg_to_rad(values: np.ndarray) -> np.ndarray:
    return values * np.pi / 180.0


def _infer_bvh_position_scale(hips_pos_raw: np.ndarray) -> float:
    # Motive CSV was millimeters, but these exported BVH root translations read
    # like centimeters in practice: root height is ~90 raw units, which should
    # be ~0.9 m, not 0.09 m. Keep the heuristic narrow and explicit.
    median_abs_y = float(np.median(np.abs(hips_pos_raw[:, 1])))
    if 50.0 <= median_abs_y <= 150.0:
        return 0.01
    return 0.001


def _quat_from_euler_zxy_deg(zxy_deg: np.ndarray) -> np.ndarray:
    angles = _deg_to_rad(zxy_deg)
    z = angles[:, 0]
    x = angles[:, 1]
    y = angles[:, 2]
    cz, sz = np.cos(z * 0.5), np.sin(z * 0.5)
    cx, sx = np.cos(x * 0.5), np.sin(x * 0.5)
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    qw = cz * cx * cy - sz * sx * sy
    qx = cz * sx * cy - sz * cx * sy
    qy = cz * cx * sy + sz * sx * cy
    qz = sz * cx * cy + cz * sx * sy
    return np.stack([qx, qy, qz, qw], axis=-1)


def _episode_to_raw_bvh(spec: dict[str, Any]) -> Path:
    source_csv = spec["inputs"]["source_clean_npz"]
    metadata_source = json.loads(resolve_existing_path(spec["inputs"]["source_sample_npz"]).with_suffix(".json").read_text()) if False else None
    del metadata_source
    csv_rel = spec["inputs"].get("source_csv")
    if not csv_rel:
        csv_rel = spec["job_id"]
    sample_npz = resolve_existing_path(spec["inputs"]["source_sample_npz"])
    metadata_path = sample_npz.parent.parent / "metadata" / f"{spec['episode_id']}.json"
    metadata = json.loads(resolve_existing_path(metadata_path).read_text())
    source_csv_rel = metadata["source"]["source_csv"]
    m = re.search(r"Skeleton(\d+)$", spec["episode_id"])
    if not m:
        raise ValueError(f"cannot parse skeleton id from {spec['episode_id']}")
    skeleton_num = int(m.group(1))
    bvh_rel = source_csv_rel.replace("Csv/", "DATA260703/Bvh/").replace(".csv", f"_Skeleton {skeleton_num:03d}.bvh")
    return Path(bvh_rel)


def _joint_series(motion, joint: str, times: np.ndarray) -> np.ndarray:
    return sample_joint_channels_at_times(motion, joint, ("Zrotation", "Xrotation", "Yrotation"), times)


def _quat_to_frame_axes(quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, w = quat_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    normal = rot[:, 1]
    tangent = rot[:, 0]
    return normal, tangent


def _smoothstep(alpha: np.ndarray) -> np.ndarray:
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _minimum_jerk(alpha: np.ndarray) -> np.ndarray:
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def _interpolate_joint_anchors(anchor_values: np.ndarray, anchor_frames: np.ndarray, total_frames: int) -> np.ndarray:
    if len(anchor_frames) >= 4:
        t = np.arange(total_frames, dtype=np.float64)
        out = np.zeros((total_frames, anchor_values.shape[1]), dtype=np.float64)
        for joint_idx in range(anchor_values.shape[1]):
            spline = CubicSpline(anchor_frames.astype(np.float64), anchor_values[:, joint_idx], bc_type="natural")
            out[:, joint_idx] = spline(t)
        return out

    out = np.zeros((total_frames, anchor_values.shape[1]), dtype=np.float64)
    for seg in range(len(anchor_frames) - 1):
        start = int(anchor_frames[seg])
        end = int(anchor_frames[seg + 1])
        if end <= start:
            out[start] = anchor_values[seg]
            continue
        for t in range(start, end + 1):
            alpha = np.clip((t - start) / float(end - start), 0.0, 1.0)
            alpha = _minimum_jerk(np.asarray(alpha))
            out[t] = (1.0 - alpha) * anchor_values[seg] + alpha * anchor_values[seg + 1]
    if int(anchor_frames[0]) > 0:
        out[: int(anchor_frames[0])] = anchor_values[0]
    if int(anchor_frames[-1]) < total_frames:
        out[int(anchor_frames[-1]) :] = anchor_values[-1]
    return out


def _unwrap_joint_series(joint_series: np.ndarray, joint_order: list[str]) -> np.ndarray:
    out = joint_series.copy()
    unwrap_names = {
        "waist_yaw_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
    }
    for idx, name in enumerate(joint_order):
        if name in unwrap_names:
            out[:, idx] = np.unwrap(out[:, idx], axis=0)
    return out


def _segment_requirements(delta_q: float, duration_s: float, vel_limit: float, acc_limit: float) -> dict[str, float]:
    abs_dq = abs(float(delta_q))
    duration_s = max(float(duration_s), 1e-6)
    required_t_vel = 1.875 * abs_dq / max(float(vel_limit), 1e-6)
    required_t_acc = float(np.sqrt(5.77 * abs_dq / max(float(acc_limit), 1e-6)))
    return {
        "delta_q": abs_dq,
        "duration_s": duration_s,
        "required_t_for_vel_limit": required_t_vel,
        "required_t_for_acc_limit": required_t_acc,
        "feasible": bool(duration_s >= max(required_t_vel, required_t_acc)),
    }


def _stroke_profile(label: str) -> dict[str, float]:
    if label == "forehand":
        return {
            "pre_near_frames": 10.0,
            "post_near_frames": 12.0,
            "pre_far_frames": 48.0,
            "post_far_frames": 56.0,
            "pre_near_scale": 1.0,
            "post_near_scale": 1.25,
            "post_far_scale": 1.55,
            "post_far_alpha_schedule": [1.0, 0.8, 0.6, 0.4],
        }
    return {
        "pre_near_frames": 9.0,
        "post_near_frames": 10.0,
        "pre_far_frames": 42.0,
        "post_far_frames": 52.0,
        "pre_near_scale": 1.0,
        "post_near_scale": 1.0,
        "post_far_scale": 1.15,
        "post_far_alpha_schedule": [1.0, 0.8, 0.6, 0.4, 0.25],
    }


def _joint_dynamic_caps(
    active_joint_names: list[str],
    duration_s: float,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    caps = []
    report: dict[str, dict[str, float]] = {}
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
        delta_q_cap = min(delta_q_vel_cap, delta_q_acc_cap)
        caps.append(delta_q_cap)
        report[name] = {
            "vel_limit": vel_limit,
            "acc_limit": acc_limit,
            "delta_q_vel_cap": delta_q_vel_cap,
            "delta_q_acc_cap": delta_q_acc_cap,
            "delta_q_cap": delta_q_cap,
        }
    return np.asarray(caps, dtype=np.float64), report


def _boundary_feasibility_repair(
    anchor_q: np.ndarray,
    anchor_frames: np.ndarray,
    joint_order: list[str],
    active_joint_names: list[str],
    dt: float,
    vel_limit: float = 12.0,
    acc_limit: float = 120.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    repaired = anchor_q.copy()
    active_idx = [joint_order.index(name) for name in active_joint_names]
    reports: dict[str, Any] = {}
    # boundary_start -> pre_far
    t_pre = max(float(anchor_frames[1] - anchor_frames[0]) * dt, 1e-6)
    # post_far -> boundary_end
    t_post = max(float(anchor_frames[-1] - anchor_frames[-2]) * dt, 1e-6)
    for joint_idx in active_idx:
        name = joint_order[joint_idx]
        pre_delta = repaired[1, joint_idx] - repaired[0, joint_idx]
        pre_req = _segment_requirements(pre_delta, t_pre, vel_limit, acc_limit)
        if pre_req["delta_q"] > 1e-9 and not pre_req["feasible"]:
            allowed = min(vel_limit * t_pre / 1.875, acc_limit * t_pre * t_pre / 5.77)
            scale = float(np.clip(allowed / pre_req["delta_q"], 0.0, 1.0))
            repaired[1, joint_idx] = repaired[0, joint_idx] + scale * pre_delta
            pre_req["repair_scale"] = scale
        else:
            pre_req["repair_scale"] = 1.0

        post_delta = repaired[-2, joint_idx] - repaired[-1, joint_idx]
        post_req = _segment_requirements(post_delta, t_post, vel_limit, acc_limit)
        if post_req["delta_q"] > 1e-9 and not post_req["feasible"]:
            allowed = min(vel_limit * t_post / 1.875, acc_limit * t_post * t_post / 5.77)
            scale = float(np.clip(allowed / post_req["delta_q"], 0.0, 1.0))
            repaired[-2, joint_idx] = repaired[-1, joint_idx] + scale * post_delta
            post_req["repair_scale"] = scale
        else:
            post_req["repair_scale"] = 1.0

        reports[name] = {
            "boundary_start_to_pre_far": pre_req,
            "post_far_to_boundary_end": post_req,
        }
    return repaired, reports


def _joint_diagnostics(joint_series: np.ndarray, dt: float, joint_order: list[str]) -> dict[str, Any]:
    vel = np.gradient(joint_series, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    vel_idx = np.unravel_index(int(np.argmax(np.abs(vel))), vel.shape)
    acc_idx = np.unravel_index(int(np.argmax(np.abs(acc))), acc.shape)
    def _neighbor(values: np.ndarray, frame_idx: int, joint_idx: int) -> list[float]:
        lo = max(0, frame_idx - 3)
        hi = min(values.shape[0], frame_idx + 4)
        return [float(x) for x in values[lo:hi, joint_idx]]
    return {
        "max_velocity_radps": float(np.max(np.abs(vel))),
        "max_velocity_joint": joint_order[int(vel_idx[1])],
        "max_velocity_frame_index": int(vel_idx[0]),
        "max_acceleration_radps2": float(np.max(np.abs(acc))),
        "max_acceleration_joint": joint_order[int(acc_idx[1])],
        "max_acceleration_frame_index": int(acc_idx[0]),
        "max_acceleration_neighbor_values": {
            "q": _neighbor(joint_series, int(acc_idx[0]), int(acc_idx[1])),
            "qdot": _neighbor(vel, int(acc_idx[0]), int(acc_idx[1])),
            "qddot": _neighbor(acc, int(acc_idx[0]), int(acc_idx[1])),
        },
    }


def _active_segment_report(
    anchor_q: np.ndarray,
    anchor_frames: np.ndarray,
    dt: float,
    joint_order: list[str],
    active_joint_names: list[str],
    vel_limit: float = 12.0,
    acc_limit: float = 120.0,
) -> dict[str, Any]:
    active_idx = [joint_order.index(name) for name in active_joint_names]
    names = [
        "boundary_start_to_pre_far",
        "pre_far_to_pre_near",
        "pre_near_to_hit",
        "hit_to_post_near",
        "post_near_to_post_far",
        "post_far_to_boundary_end",
    ]
    seg_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    out: dict[str, Any] = {}
    for joint_idx in active_idx:
        joint_name = joint_order[joint_idx]
        joint_report = {}
        for seg_name, (i0, i1) in zip(names, seg_pairs):
            delta_q = float(anchor_q[i1, joint_idx] - anchor_q[i0, joint_idx])
            duration_s = float(anchor_frames[i1] - anchor_frames[i0]) * dt
            req = _segment_requirements(delta_q, duration_s, vel_limit, acc_limit)
            joint_report[seg_name] = {
                "frame_start": int(anchor_frames[i0]),
                "frame_end": int(anchor_frames[i1]),
                "q_start": float(anchor_q[i0, joint_idx]),
                "q_end": float(anchor_q[i1, joint_idx]),
                **req,
            }
        out[joint_name] = joint_report
    return out


def _adjust_anchor_frames_from_feasibility(
    anchor_q: np.ndarray,
    anchor_frames: np.ndarray,
    dt: float,
    joint_order: list[str],
    active_joint_names: list[str],
    total_frames: int,
    vel_limit: float = 12.0,
    acc_limit: float = 120.0,
) -> np.ndarray:
    report = _active_segment_report(anchor_q, anchor_frames, dt, joint_order, active_joint_names, vel_limit, acc_limit)
    req_pre = 0.0
    req_post_near = 0.0
    req_post_far = 0.0
    for joint_name, joint_report in report.items():
        req_pre = max(
            req_pre,
            joint_report["pre_near_to_hit"]["required_t_for_vel_limit"],
            joint_report["pre_near_to_hit"]["required_t_for_acc_limit"],
        )
        req_post_near = max(
            req_post_near,
            joint_report["hit_to_post_near"]["required_t_for_vel_limit"],
            joint_report["hit_to_post_near"]["required_t_for_acc_limit"],
        )
        req_post_far = max(
            req_post_far,
            joint_report["post_near_to_post_far"]["required_t_for_vel_limit"],
            joint_report["post_near_to_post_far"]["required_t_for_acc_limit"],
        )
    req_pre_frames = int(np.ceil(req_pre / max(dt, 1e-6)))
    req_post_near_frames = int(np.ceil(req_post_near / max(dt, 1e-6)))
    req_post_far_frames = int(np.ceil(req_post_far / max(dt, 1e-6)))
    adjusted = anchor_frames.copy()
    original = anchor_frames.copy()
    # 7-anchor layout:
    # [0]=boundary_start, [1]=pre_far, [2]=pre_near, [3]=hit, [4]=post_near, [5]=post_far, [6]=boundary_end
    adjusted[2] = max(original[1] + 4, min(original[2], adjusted[3] - max(req_pre_frames, 1)))
    adjusted[4] = min(original[5] - 4, max(original[4], adjusted[3] + max(req_post_near_frames, 1)))
    adjusted[1] = max(1, original[1])
    adjusted[5] = min(total_frames - 2, max(original[5], adjusted[4] + max(req_post_far_frames, 4)))
    # keep strict monotonic ordering
    adjusted[2] = max(adjusted[1] + 1, adjusted[2])
    adjusted[3] = max(adjusted[2] + 1, adjusted[3])
    adjusted[4] = max(adjusted[3] + 1, adjusted[4])
    adjusted[5] = max(adjusted[4] + 1, adjusted[5])
    return adjusted


def _adjust_hit_corridor_anchor_frames(
    anchor_q: np.ndarray,
    anchor_frames: np.ndarray,
    dt: float,
    joint_order: list[str],
    active_joint_names: list[str],
    total_frames: int,
    safety_scale: float = 1.10,
) -> tuple[np.ndarray, dict[str, Any]]:
    report = _active_segment_report(anchor_q, anchor_frames, dt, joint_order, active_joint_names)
    req_pre = 0.0
    req_post = 0.0
    worst: dict[str, Any] = {
        "pre_near_to_hit": {"ratio": 0.0},
        "hit_to_post_near": {"ratio": 0.0},
    }
    for joint_name, joint_report in report.items():
        for seg_name, key in (("pre_near_to_hit", "pre_near_to_hit"), ("hit_to_post_near", "hit_to_post_near")):
            seg = joint_report[seg_name]
            req = max(seg["required_t_for_vel_limit"], seg["required_t_for_acc_limit"])
            ratio = req / max(seg["duration_s"], 1e-9)
            if ratio > worst[key]["ratio"]:
                worst[key] = {
                    "ratio": float(ratio),
                    "joint": joint_name,
                    "required_time_s": float(req),
                    "duration_s": float(seg["duration_s"]),
                    "delta_q": float(seg["delta_q"]),
                }
            if seg_name == "pre_near_to_hit":
                req_pre = max(req_pre, req)
            else:
                req_post = max(req_post, req)

    adjusted = anchor_frames.copy()
    original = anchor_frames.copy()
    hit = int(original[3])
    req_pre_frames = int(np.ceil(safety_scale * req_pre / max(dt, 1e-9)))
    req_post_frames = int(np.ceil(safety_scale * req_post / max(dt, 1e-9)))
    if req_pre_frames > 0:
        adjusted[2] = max(int(original[1]) + 4, min(int(original[2]), hit - req_pre_frames))
    if req_post_frames > 0:
        adjusted[4] = min(int(original[5]) - 4, max(int(original[4]), hit + req_post_frames))

    adjusted[2] = min(adjusted[2], hit - 1)
    adjusted[4] = max(adjusted[4], hit + 1)
    adjusted[1] = min(adjusted[1], adjusted[2] - 1)
    adjusted[5] = max(adjusted[5], adjusted[4] + 1)
    adjusted[5] = min(adjusted[5], total_frames - 2)
    adjusted[6] = total_frames - 1
    return adjusted, {
        "original_anchor_frames": [int(x) for x in original.tolist()],
        "adjusted_anchor_frames": [int(x) for x in adjusted.tolist()],
        "changed": bool(not np.array_equal(original, adjusted)),
        "required_pre_frames": int(req_pre_frames),
        "required_post_frames": int(req_post_frames),
        "worst_segments": worst,
    }


def _solve_anchor_task_space(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    q_init: np.ndarray,
    target_pos: np.ndarray,
    target_normal: np.ndarray,
    target_tangent: np.ndarray,
    joint_order: list[str],
    active_joint_names: list[str],
    target_weights: tuple[float, float, float] = (4.0, 2.0, 1.0),
    delta_cap: np.ndarray | None = None,
) -> np.ndarray:
    limits = load_a3_joint_limits()
    active_idx = [joint_order.index(name) for name in active_joint_names]
    joint_index_by_name = {name: i for i, name in enumerate(joint_order)}
    lower = np.asarray([limits[joint_order[i]][0] for i in active_idx], dtype=np.float64)
    upper = np.asarray([limits[joint_order[i]][1] for i in active_idx], dtype=np.float64)
    q_init = q_init.copy()
    q_init[active_idx] = np.clip(q_init[active_idx], lower, upper)
    delta_weight = np.ones(len(active_idx), dtype=np.float64) * 0.05
    for local_i, joint_i in enumerate(active_idx):
        name = joint_order[joint_i]
        if name == "right_shoulder_pitch_joint":
            delta_weight[local_i] = 0.16
        elif name == "waist_yaw_joint":
            delta_weight[local_i] = 0.03
        elif name in ("waist_roll_joint", "waist_pitch_joint", "right_shoulder_yaw_joint", "right_elbow_joint"):
            delta_weight[local_i] = 0.04

    prev_weight = np.ones(len(active_idx), dtype=np.float64) * 0.04
    for local_i, joint_i in enumerate(active_idx):
        if joint_order[joint_i] == "right_shoulder_pitch_joint":
            prev_weight[local_i] = 0.10

    def residual(delta: np.ndarray) -> np.ndarray:
        q = q_init.copy()
        q_active = np.clip(q_init[active_idx] + delta, lower, upper)
        q[active_idx] = q_active
        pos, rot = _fk_racket_state(base_pos, base_quat, q, joint_index_by_name, {})
        normal = rot[:, 1]
        tangent = rot[:, 0]
        pos_w, normal_w, tangent_w = target_weights
        res = [
            pos_w * (pos - target_pos),
            normal_w * (normal - target_normal),
            tangent_w * (tangent - target_tangent),
            delta_weight * delta,
            prev_weight * (q_active - q_init[active_idx]),
        ]
        return np.concatenate(res, axis=0)

    lb = lower - q_init[active_idx]
    ub = upper - q_init[active_idx]
    if delta_cap is not None:
        delta_cap = np.asarray(delta_cap, dtype=np.float64)
        lb = np.maximum(lb, -delta_cap)
        ub = np.minimum(ub, delta_cap)

    result = least_squares(
        residual,
        x0=np.zeros(len(active_idx), dtype=np.float64),
        bounds=(lb, ub),
        max_nfev=64,
        verbose=0,
    )
    q_out = q_init.copy()
    q_out[active_idx] = np.clip(q_init[active_idx] + np.clip(result.x, lb, ub), lower, upper)
    return q_out


def _pick_post_far_anchor(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    q_post_near: np.ndarray,
    post_near_target_pos: np.ndarray,
    raw_post_far_target_pos: np.ndarray,
    hit_target_normal: np.ndarray,
    hit_target_tangent: np.ndarray,
    profile: dict[str, float],
    dt: float,
    post_near_frame: int,
    post_far_frame: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    seg_duration_s = max(float(post_far_frame - post_near_frame) * dt, 1e-6)
    delta_cap, cap_report = _joint_dynamic_caps(A3_ACTIVE_JOINTS_FIRST_PASS, seg_duration_s)
    best_q = q_post_near.copy()
    best_report: dict[str, Any] | None = None
    for alpha in profile["post_far_alpha_schedule"]:
        target_pos = post_near_target_pos + float(alpha) * (raw_post_far_target_pos - post_near_target_pos)
        q_candidate = _solve_anchor_task_space(
            base_pos=base_pos,
            base_quat=base_quat,
            q_init=q_post_near,
            target_pos=target_pos,
            target_normal=hit_target_normal,
            target_tangent=hit_target_tangent,
            joint_order=A3_POLICY_JOINT_ORDER,
            active_joint_names=A3_ACTIVE_JOINTS_FIRST_PASS,
            target_weights=(1.2, 0.4, 0.2),
            delta_cap=delta_cap,
        )
        ratios = {}
        feasible = True
        for name in A3_ACTIVE_JOINTS_FIRST_PASS:
            idx = A3_POLICY_JOINT_ORDER.index(name)
            delta_q = float(abs(q_candidate[idx] - q_post_near[idx]))
            cap = float(cap_report[name]["delta_q_cap"])
            ratio = delta_q / max(cap, 1e-9)
            ratios[name] = ratio
            feasible = feasible and ratio <= 1.0 + 1e-6
        report = {
            "alpha": float(alpha),
            "target_pos": [float(x) for x in target_pos.tolist()],
            "ratios_to_cap": ratios,
            "feasible": feasible,
        }
        if feasible:
            return q_candidate, report
        if best_report is None or max(ratios.values()) < max(best_report["ratios_to_cap"].values()):
            best_q = q_candidate
            best_report = report
    return best_q, best_report if best_report is not None else {"alpha": None, "target_pos": None, "ratios_to_cap": {}, "feasible": False}


def build_generic_init_csv(spec: dict[str, Any], enable_timing_repair: bool = False) -> np.ndarray:
    sample_npz_path = resolve_existing_path(spec["inputs"]["source_sample_npz"])
    sample_npz = np.load(sample_npz_path, allow_pickle=False)
    metadata_path = sample_npz_path.parent.parent / "metadata" / f"{spec['episode_id']}.json"
    metadata = json.loads(resolve_existing_path(metadata_path).read_text())
    source_csv_rel = metadata["source"]["source_csv"]
    source_hit_time = float(metadata["source"]["hit_metadata"]["hit_time"])
    target_times = source_hit_time + sample_npz["time_rel"]

    raw_bvh = resolve_existing_path(_episode_to_raw_bvh(spec))
    motion = load_bvh(raw_bvh)

    out = np.zeros((target_times.shape[0], 7 + len(A3_POLICY_JOINT_ORDER)), dtype=np.float64)
    hips_pos_mm = sample_joint_channels_at_times(motion, "Hips", ("Xposition", "Yposition", "Zposition"), target_times)
    hips_rot_deg = sample_joint_channels_at_times(motion, "Hips", ("Zrotation", "Xrotation", "Yrotation"), target_times)
    pos_scale = _infer_bvh_position_scale(hips_pos_mm)
    base_pos_series = hips_pos_mm * pos_scale
    base_quat_series = _quat_from_euler_zxy_deg(hips_rot_deg)
    hit_index = int(spec["hit_target"]["hit_index"])
    out[:, 0:3] = np.tile(base_pos_series[hit_index][None, :], (target_times.shape[0], 1))
    out[:, 3:7] = np.tile(base_quat_series[hit_index][None, :], (target_times.shape[0], 1))

    joint_map = {name: idx for idx, name in enumerate(A3_POLICY_JOINT_ORDER)}
    defaults = np.asarray([A3_DEFAULT_JOINT_POS[name] for name in A3_POLICY_JOINT_ORDER], dtype=np.float64)
    out[:, 7:] = defaults[None, :]

    spine = _joint_series(motion, "Spine", target_times)
    spine1 = _joint_series(motion, "Spine1", target_times)
    r_shoulder = _joint_series(motion, "RightShoulder", target_times)
    r_arm = _joint_series(motion, "RightArm", target_times)
    r_forearm = _joint_series(motion, "RightForeArm", target_times)
    r_hand = _joint_series(motion, "RightHand", target_times)

    out[:, 7 + joint_map["waist_yaw_joint"]] = _deg_to_rad(0.35 * spine[:, 2] + 0.65 * spine1[:, 2])
    out[:, 7 + joint_map["waist_roll_joint"]] = _deg_to_rad(0.5 * spine[:, 0] + 0.5 * spine1[:, 0])
    out[:, 7 + joint_map["waist_pitch_joint"]] = _deg_to_rad(0.4 * spine[:, 1] + 0.6 * spine1[:, 1])
    out[:, 7 + joint_map["right_shoulder_pitch_joint"]] = _deg_to_rad(r_arm[:, 1])
    out[:, 7 + joint_map["right_shoulder_roll_joint"]] = _deg_to_rad(-r_shoulder[:, 0])
    out[:, 7 + joint_map["right_shoulder_yaw_joint"]] = _deg_to_rad(r_arm[:, 2])
    out[:, 7 + joint_map["right_elbow_joint"]] = np.clip(_deg_to_rad(np.abs(r_forearm[:, 1])), 0.0, 2.4)
    out[:, 7 + joint_map["right_wrist_roll_joint"]] = _deg_to_rad(r_hand[:, 0])
    out[:, 7 + joint_map["right_wrist_pitch_joint"]] = _deg_to_rad(r_hand[:, 1])
    out[:, 7 + joint_map["right_wrist_yaw_joint"]] = _deg_to_rad(r_hand[:, 2])

    profile = _stroke_profile(str(spec["label"]))
    pre_far = max(1, hit_index - int(profile["pre_far_frames"]))
    pre_near = max(pre_far + 1, hit_index - int(profile["pre_near_frames"]))
    post_near = min(target_times.shape[0] - 2, hit_index + int(profile["post_near_frames"]))
    post_far = min(target_times.shape[0] - 2, hit_index + int(profile["post_far_frames"]))
    anchor_frames = np.asarray(
        [
            0,
            pre_far,
            pre_near,
            int(hit_index),
            post_near,
            post_far,
            target_times.shape[0] - 1,
        ],
        dtype=int,
    )
    hit_target_pos = np.asarray(spec["hit_target"]["racket_position_m"], dtype=np.float64)
    hit_target_vel_dir = np.asarray(spec["hit_target"]["racket_velocity_direction_w"], dtype=np.float64)
    hit_target_normal = np.asarray(spec["hit_target"]["racket_normal_w"], dtype=np.float64)
    hit_target_tangent = np.asarray(spec["hit_target"]["racket_tangent_w"], dtype=np.float64)
    base_pre_near_dist = float(
        np.linalg.norm(sample_npz["racket_pos"][hit_index].astype(np.float64) - sample_npz["racket_pos"][pre_near].astype(np.float64))
    )
    base_post_near_dist = float(
        np.linalg.norm(sample_npz["racket_pos"][post_near].astype(np.float64) - sample_npz["racket_pos"][hit_index].astype(np.float64))
    )
    base_post_far_dist = float(
        np.linalg.norm(sample_npz["racket_pos"][post_far].astype(np.float64) - sample_npz["racket_pos"][hit_index].astype(np.float64))
    )

    def _swing_targets_for_frames(frames: np.ndarray) -> dict[str, np.ndarray]:
        # Timing repair moves anchor timestamps, not the near-hit spatial
        # corridor. Recomputing distances from farther source frames would
        # make the target farther exactly when we are trying to buy time.
        del frames
        return {
            "pre_near": hit_target_pos - profile["pre_near_scale"] * base_pre_near_dist * hit_target_vel_dir,
            "post_near": hit_target_pos + profile["post_near_scale"] * base_post_near_dist * hit_target_vel_dir,
            "post_far": hit_target_pos + profile["post_far_scale"] * base_post_far_dist * hit_target_vel_dir,
        }

    def _racket_task_score(q: np.ndarray, target_pos: np.ndarray, target_normal: np.ndarray, target_tangent: np.ndarray) -> float:
        pos, rot = _fk_racket_state(out[hit_index, 0:3], out[hit_index, 3:7], q, joint_map, spec)
        normal = rot[:, int(spec["a3_bodies"]["racket_normal_axis"])] * float(spec["a3_bodies"]["racket_normal_sign"])
        tangent = rot[:, int(spec["a3_bodies"]["racket_tangent_axis"])]
        normal_err = 1.0 - float(np.clip(np.dot(normal, target_normal) / (np.linalg.norm(normal) * np.linalg.norm(target_normal) + 1e-9), -1.0, 1.0))
        tangent_err = 1.0 - float(np.clip(np.dot(tangent, target_tangent) / (np.linalg.norm(tangent) * np.linalg.norm(target_tangent) + 1e-9), -1.0, 1.0))
        return float(np.linalg.norm(pos - target_pos) + 0.20 * normal_err + 0.08 * tangent_err)

    def _solve_best_anchor(
        frame: int,
        q_seeds: list[np.ndarray],
        target_pos: np.ndarray,
        target_normal: np.ndarray,
        target_tangent: np.ndarray,
        target_weights: tuple[float, float, float],
    ) -> np.ndarray:
        candidates = []
        for seed in q_seeds:
            candidates.append(
                _solve_anchor_task_space(
                    base_pos=out[frame, 0:3],
                    base_quat=out[frame, 3:7],
                    q_init=seed,
                    target_pos=target_pos,
                    target_normal=target_normal,
                    target_tangent=target_tangent,
                    joint_order=A3_POLICY_JOINT_ORDER,
                    active_joint_names=A3_ACTIVE_JOINTS_FIRST_PASS,
                    target_weights=target_weights,
                )
            )
        return min(candidates, key=lambda q: _racket_task_score(q, target_pos, target_normal, target_tangent))

    def _solve_anchor_bundle(frames: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        swing_targets = _swing_targets_for_frames(frames)
        post_far_repair = {"alpha": None, "target_pos": None, "ratios_to_cap": {}, "feasible": False}
        q_start = out[frames[0], 7:].copy()
        q_default = defaults.copy()

        pre_far_pos = sample_npz["racket_pos"][frames[1]].astype(np.float64)
        pre_far_normal, pre_far_tangent = _quat_to_frame_axes(sample_npz["racket_quat"][frames[1]].astype(np.float64))
        q_pre_far = _solve_anchor_task_space(
            base_pos=out[frames[1], 0:3],
            base_quat=out[frames[1], 3:7],
            q_init=q_start,
            target_pos=pre_far_pos,
            target_normal=pre_far_normal,
            target_tangent=pre_far_tangent,
            joint_order=A3_POLICY_JOINT_ORDER,
            active_joint_names=A3_ACTIVE_JOINTS_FIRST_PASS,
            target_weights=(2.0, 0.8, 0.4),
        )

        # The hit pose is the contract anchor. Solve it from multiple seeds so
        # a bad early-preparation branch cannot poison the whole swing.
        q_hit = _solve_best_anchor(
            frame=int(frames[3]),
            q_seeds=[q_pre_far, q_default, out[frames[3], 7:].copy()],
            target_pos=hit_target_pos,
            target_normal=hit_target_normal,
            target_tangent=hit_target_tangent,
            target_weights=(24.0, 10.0, 5.0),
        )
        near_cap_scale = 3.0
        pre_near_cap, _ = _joint_dynamic_caps(
            A3_ACTIVE_JOINTS_FIRST_PASS,
            max(float(frames[3] - frames[2]) * float(spec["coordinate_contract"]["dt"]), 1e-6),
        )
        post_near_cap, _ = _joint_dynamic_caps(
            A3_ACTIVE_JOINTS_FIRST_PASS,
            max(float(frames[4] - frames[3]) * float(spec["coordinate_contract"]["dt"]), 1e-6),
        )
        q_pre_near = _solve_anchor_task_space(
            base_pos=out[frames[2], 0:3],
            base_quat=out[frames[2], 3:7],
            q_init=q_hit,
            target_pos=swing_targets["pre_near"],
            target_normal=hit_target_normal,
            target_tangent=hit_target_tangent,
            joint_order=A3_POLICY_JOINT_ORDER,
            active_joint_names=A3_ACTIVE_JOINTS_FIRST_PASS,
            target_weights=(8.0, 4.0, 1.5),
            delta_cap=near_cap_scale * pre_near_cap,
        )
        q_post_near = _solve_anchor_task_space(
            base_pos=out[frames[4], 0:3],
            base_quat=out[frames[4], 3:7],
            q_init=q_hit,
            target_pos=swing_targets["post_near"],
            target_normal=hit_target_normal,
            target_tangent=hit_target_tangent,
            joint_order=A3_POLICY_JOINT_ORDER,
            active_joint_names=A3_ACTIVE_JOINTS_FIRST_PASS,
            target_weights=(8.0, 4.0, 1.5),
            delta_cap=near_cap_scale * post_near_cap,
        )
        q_post_far, post_far_repair = _pick_post_far_anchor(
            base_pos=out[frames[5], 0:3],
            base_quat=out[frames[5], 3:7],
            q_post_near=q_post_near,
            post_near_target_pos=swing_targets["post_near"],
            raw_post_far_target_pos=swing_targets["post_far"],
            hit_target_normal=hit_target_normal,
            hit_target_tangent=hit_target_tangent,
            profile=profile,
            dt=float(spec["coordinate_contract"]["dt"]),
            post_near_frame=int(frames[4]),
            post_far_frame=int(frames[5]),
        )
        q_end = out[frames[-1], 7:].copy()
        bundle = [q_start, q_pre_far, q_pre_near, q_hit, q_post_near, q_post_far, q_end]
        return np.asarray(bundle, dtype=np.float64), post_far_repair

    anchor_q, post_far_repair = _solve_anchor_bundle(anchor_frames)
    if enable_timing_repair:
        anchor_frames, timing_repair = _adjust_hit_corridor_anchor_frames(
            anchor_q=anchor_q,
            anchor_frames=anchor_frames,
            dt=float(spec["coordinate_contract"]["dt"]),
            joint_order=A3_POLICY_JOINT_ORDER,
            active_joint_names=A3_ACTIVE_JOINTS_FIRST_PASS,
            total_frames=target_times.shape[0],
        )
        if timing_repair["changed"]:
            anchor_q, post_far_repair = _solve_anchor_bundle(anchor_frames)
    else:
        timing_repair = {"skipped": True, "reason": "disabled_by_default"}
    swing_targets = _swing_targets_for_frames(anchor_frames)
    boundary_repair = {"skipped": True, "reason": "generic_init_preserves_hit_corridor; temporal_repair_handles_boundary_dynamics"}
    joint_series = _interpolate_joint_anchors(anchor_q, anchor_frames, target_times.shape[0])
    joint_series = _unwrap_joint_series(joint_series, A3_POLICY_JOINT_ORDER)
    limits = load_a3_joint_limits()
    for idx_joint, joint_name in enumerate(A3_POLICY_JOINT_ORDER):
        if joint_name in limits:
            lo, hi = limits[joint_name]
            joint_series[:, idx_joint] = np.clip(joint_series[:, idx_joint], lo, hi)
    out[:, 7:] = joint_series
    diagnostics = _joint_diagnostics(out[:, 7:], float(spec["coordinate_contract"]["dt"]), A3_POLICY_JOINT_ORDER)
    diagnostics["anchor_frames"] = [int(x) for x in anchor_frames.tolist()]
    diagnostics["stroke_profile"] = profile
    diagnostics["swing_direction_targets"] = {
        "hit_target_velocity_direction_w": [float(x) for x in hit_target_vel_dir.tolist()],
        "pre_near_target_pos": [float(x) for x in swing_targets["pre_near"].tolist()],
        "hit_target_pos": [float(x) for x in hit_target_pos.tolist()],
        "post_near_target_pos": [float(x) for x in swing_targets["post_near"].tolist()],
        "post_far_target_pos": [float(x) for x in swing_targets["post_far"].tolist()],
    }
    diagnostics["anchor_joint_values"] = {
        name: [float(anchor_q[i, A3_POLICY_JOINT_ORDER.index(name)]) for i in range(anchor_q.shape[0])]
        for name in ("waist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_elbow_joint")
    }
    diagnostics["boundary_feasibility_repair"] = boundary_repair
    diagnostics["hit_corridor_timing_repair"] = timing_repair
    diagnostics["post_followthrough_repair"] = post_far_repair
    diagnostics["active_segment_report"] = _active_segment_report(
        anchor_q=anchor_q,
        anchor_frames=anchor_frames,
        dt=float(spec["coordinate_contract"]["dt"]),
        joint_order=A3_POLICY_JOINT_ORDER,
        active_joint_names=A3_ACTIVE_JOINTS_FIRST_PASS,
    )
    return out, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--enable-timing-repair", action="store_true")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    out, diagnostics = build_generic_init_csv(spec, enable_timing_repair=bool(args.enable_timing_repair))
    output = args.output if args.output is not None else Path(spec["artifacts"]["generic_retarget_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, out, delimiter=",", fmt="%.10f")
    diag_path = output.with_suffix(".diagnostics.json")
    diag_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {output}")
    print(f"shape {out.shape}")
    print(f"Wrote {diag_path}")


if __name__ == "__main__":
    main()
