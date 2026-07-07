"""Helpers for building A3 racket-first refinement specs from clean samples."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from analysis.mocap_cleaning.a3_metadata import (
    A3_ACTIVE_JOINTS_FIRST_PASS,
    A3_LOCKED_JOINTS_FIRST_PASS,
    A3_MOUNT_NORMAL_AXIS,
    A3_MOUNT_NORMAL_SIGN,
    A3_MOUNT_OFFSET_M,
    A3_MOUNT_QUAT_XYZW,
    A3_POLICY_JOINT_ORDER,
    A3_RACKET_BODY,
    A3_RACKET_TANGENT_AXIS,
    A3_RACKET_UP_AXIS,
    A3_WEAK_TRACK_JOINTS_FIRST_PASS,
    A3_WRIST_BODY,
)


def resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    text = str(candidate)
    replacements = (
        ("data/analysis/mocap_cleaning_outputs/", "analysis/mocap_cleaning_outputs/"),
        ("analysis/mocap_cleaning_outputs/", "data/analysis/mocap_cleaning_outputs/"),
    )
    for src, dst in replacements:
        if text.startswith(src):
            alt = Path(dst + text[len(src):])
            if alt.exists():
                return alt
    raise FileNotFoundError(f"cannot resolve path: {candidate}")


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
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


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.full_like(vec, np.nan, dtype=np.float64)
    return vec / norm


def load_clean_sample(path: str | Path) -> dict[str, np.ndarray]:
    sample_path = resolve_existing_path(path)
    data = np.load(sample_path, allow_pickle=False)
    return {name: data[name] for name in data.files}


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = resolve_existing_path(path)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def optional_file_fingerprint(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.exists():
        return file_fingerprint(candidate)
    text = str(candidate)
    replacements = (
        ("data/analysis/mocap_cleaning_outputs/", "analysis/mocap_cleaning_outputs/"),
        ("analysis/mocap_cleaning_outputs/", "data/analysis/mocap_cleaning_outputs/"),
    )
    for src, dst in replacements:
        if text.startswith(src):
            alt = Path(dst + text[len(src):])
            if alt.exists():
                return file_fingerprint(alt)
    return {
        "path": str(path),
        "exists": False,
    }


def build_refinement_spec(job: dict[str, Any]) -> dict[str, Any]:
    sample = load_clean_sample(job["source_sample_npz"])
    hit_index = int(sample["hit_index"])
    time_rel = sample["time_rel"].astype(np.float64)
    racket_pos = sample["racket_pos"].astype(np.float64)
    racket_quat = sample["racket_quat"].astype(np.float64)
    racket_vel = sample["racket_vel"].astype(np.float64)
    body_center = sample["body_center"].astype(np.float64)
    body_right_axis = sample["body_right_axis"].astype(np.float64)

    quat_hit = sample["racket_pose_at_hit"][3:7].astype(np.float64)
    rot_hit = quat_xyzw_to_matrix(quat_hit)
    normal_hit = normalize(rot_hit[:, A3_MOUNT_NORMAL_AXIS] * A3_MOUNT_NORMAL_SIGN)
    tangent_hit = normalize(rot_hit[:, A3_RACKET_TANGENT_AXIS])
    up_hit = normalize(rot_hit[:, A3_RACKET_UP_AXIS])
    vel_hit = sample["racket_vel_at_hit"].astype(np.float64)
    vel_dir_hit = normalize(vel_hit)
    body_axis_hit = normalize(body_right_axis[hit_index])
    dt = float(np.nanmedian(np.diff(time_rel)))
    sequence_length_frames = int(time_rel.shape[0])

    def _window_bounds(offset_a: int, offset_b: int) -> dict[str, Any]:
        lo = max(0, hit_index + offset_a)
        hi = min(len(time_rel) - 1, hit_index + offset_b)
        return {
            "frame_start": int(lo),
            "frame_end": int(hi),
            "time_rel_start_s": float(time_rel[lo]),
            "time_rel_end_s": float(time_rel[hi]),
        }

    quality_thresholds = {
        "warning": {
            "racket_position_error_at_hit_m_max": 0.035,
            "racket_orientation_error_at_hit_deg_max": 10.0,
            "racket_velocity_direction_error_at_hit_deg_max": 12.0,
            "ik_residual_rms_max": 0.02,
            "max_joint_limit_violation_before_clamp_rad_max": 0.04,
            "max_joint_velocity_radps_max": 10.0,
            "max_joint_acceleration_radps2_max": 90.0,
        },
        "reject": {
            "racket_position_error_at_hit_m_max": 0.05,
            "racket_orientation_error_at_hit_deg_max": 15.0,
            "racket_velocity_direction_error_at_hit_deg_max": 20.0,
            "ik_residual_rms_max": 0.03,
            "max_joint_limit_violation_before_clamp_rad_max": 0.10,
            "max_joint_velocity_radps_max": 12.0,
            "max_joint_acceleration_radps2_max": 120.0,
        },
    }

    return {
        "spec_version": "1.1.0",
        "contract_version": "a3_refinement_contract_v1",
        "job_id": job["job_id"],
        "episode_id": job["episode_id"],
        "label": job["label"],
        "confidence": job["confidence"],
        "robot": job["robot"],
        "coordinate_contract": {
            "position_frame": "motive_global_m",
            "orientation_frame": "motive_global_m",
            "position_unit": "m",
            "angle_unit": "rad",
            "time_unit": "s",
            "quat_order": "xyzw",
            "fps": float(job["fps"]),
            "dt": dt,
            "hit_index": hit_index,
            "hit_timestamp_rel_s": float(time_rel[hit_index]),
            "sequence_length_frames": sequence_length_frames,
        },
        "a3_joint_order": A3_POLICY_JOINT_ORDER,
        "joint_masks": {
            "active_joints_first_pass": A3_ACTIVE_JOINTS_FIRST_PASS,
            "locked_joints_first_pass": A3_LOCKED_JOINTS_FIRST_PASS,
            "weak_track_joints_first_pass": A3_WEAK_TRACK_JOINTS_FIRST_PASS,
        },
        "a3_bodies": {
            "wrist_body": A3_WRIST_BODY,
            "racket_body": A3_RACKET_BODY,
            "wrist_to_racket_pos_m": list(A3_MOUNT_OFFSET_M),
            "wrist_to_racket_quat_xyzw": list(A3_MOUNT_QUAT_XYZW),
            "racket_center_body": A3_RACKET_BODY,
            "racket_normal_axis": A3_MOUNT_NORMAL_AXIS,
            "racket_normal_sign": A3_MOUNT_NORMAL_SIGN,
            "racket_tangent_axis": A3_RACKET_TANGENT_AXIS,
            "racket_up_axis": A3_RACKET_UP_AXIS,
        },
        "inputs": {
            "source_motion_id": job["episode_id"],
            "source_bvh": job["source_bvh"],
            "source_sample_npz": job["source_sample_npz"],
            "source_clean_npz": job["source_clean_npz"],
            "source_debug_npz": job["source_debug_npz"],
            "generic_retarget_csv": job["generic_retarget_csv"],
            "expected_refined_retarget_csv": job["retarget_csv"],
        },
        "input_fingerprints": {
            "source_sample_npz": file_fingerprint(job["source_sample_npz"]),
            "source_clean_npz": file_fingerprint(job["source_clean_npz"]),
            "source_debug_npz": file_fingerprint(job["source_debug_npz"]),
            "source_bvh": optional_file_fingerprint(job["source_bvh"]),
            "generic_retarget_csv": optional_file_fingerprint(job["generic_retarget_csv"]),
        },
        "hit_target": {
            "hit_index": hit_index,
            "time_rel_s": float(time_rel[hit_index]),
            "racket_position_m": sample["racket_pose_at_hit"][:3].astype(np.float64).tolist(),
            "racket_quat_xyzw": quat_hit.tolist(),
            "racket_normal_w": normal_hit.tolist(),
            "racket_tangent_w": tangent_hit.tolist(),
            "racket_up_w": up_hit.tolist(),
            "racket_velocity_mps": vel_hit.tolist(),
            "racket_velocity_direction_w": vel_dir_hit.tolist(),
            "body_center_w": body_center[hit_index].tolist(),
            "body_right_axis_w": body_axis_hit.tolist(),
        },
        "windows": {
            "pre_hit": _window_bounds(-24, -4),
            "hit": _window_bounds(-3, 3),
            "post_hit": _window_bounds(4, 20),
        },
        "refinement_policy": job["constraint_profile"]["refinement_policy"],
        "phase_weights": job["constraint_profile"]["phase_weights"],
        "constraints": job["constraint_profile"]["constraints"],
        "quality_template": {
            **job["quality_template"],
            **{key: None for key in quality_thresholds["reject"]},
            "validation_status": None,
            "validation_warnings": [],
            "validation_reject_reasons": [],
        },
        "quality_thresholds": quality_thresholds,
        "artifacts": {
            "generic_retarget_csv": job["generic_retarget_csv"],
            "refined_retarget_csv": job["retarget_csv"],
            "retarget_csv": job["retarget_csv"],
            "motion_npz": job["motion_npz"],
            "quality_report_json": job["refinement_outputs"]["quality_report_json"],
            "refinement_spec_json": job["refinement_outputs"]["quality_report_json"].replace(
                "/quality_reports/", "/refinement_specs/"
            ),
        },
        "notes": [
            "Use generic retarget output as the initialization for refinement.",
            "Prioritize racket pose/normal/velocity at hit over human arm shape fidelity.",
            "Allow torso participation; keep legs fixed in the first pass unless reachability fails.",
            "Calibrate human_wrist_to_racket_center and a3_ee_to_racket_center before trusting hit-point errors.",
            "Solver should read active/locked/weak-track joint masks from this spec, not hard-code them internally.",
        ],
    }


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
