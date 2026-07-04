"""CleanSample export helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from analysis.mocap_cleaning.derivative import (
    compute_angular_velocity,
    compute_velocity,
)
from analysis.mocap_cleaning.labeling import judge_success
from analysis.mocap_cleaning.labeling import classify_stroke_type
from analysis.mocap_cleaning.resampling import (
    make_hit_centered_time,
    resample_array,
    resample_quat_xyzw,
)


@dataclass
class SampleExportResult:
    episode_id: str
    sample_path: str
    metadata_path: str
    source_csv: str
    source_bvh: str
    hit_index: int
    hit_time: float
    hit_time_rel: float
    frames: int
    usable_for_training: bool
    quality_flags: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_mean_time_window(values: np.ndarray, time_rel: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    mask = (time_rel >= start_s) & (time_rel <= end_s)
    if not np.any(mask):
        return np.full(values.shape[1], np.nan)
    return np.nanmean(values[mask], axis=0)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = np.isfinite(vectors).all(axis=1, keepdims=True) & (norm > 1e-12)
    return np.divide(vectors, norm, out=np.full_like(vectors, np.nan), where=valid)


def export_clean_sample(
    *,
    episode_id: str,
    source_npz: str,
    debug_npz: str,
    source_csv: str,
    source_bvh: str,
    racket: str,
    candidate: str,
    hit_metadata: dict[str, Any],
    cleaning_usable: bool,
    output_npz: str,
    output_metadata: str,
    pre_hit_s: float,
    post_hit_s: float,
    target_fps: float,
    max_ball_speed_mps: float,
    max_racket_speed_mps: float,
    max_racket_omega_radps: float,
    table_config: dict[str, Any] | None = None,
    handedness: str = "right",
) -> SampleExportResult:
    src = np.load(source_npz, allow_pickle=True)
    dbg = np.load(debug_npz, allow_pickle=True)

    time = src["time"]
    hit_index_global = int(hit_metadata["hit_index"])
    hit_time = float(hit_metadata["hit_time"])
    time_new = make_hit_centered_time(hit_time, pre_hit_s, post_hit_s, target_fps)
    time_rel = time_new - hit_time
    hit_index = int(np.argmin(np.abs(time_new - hit_time)))

    ball_pos = resample_array(time, src["ball_pos_clean"], time_new)
    raw_ball_vel = resample_array(time, src["ball_vel"], time_new)
    racket_pos = resample_array(time, src["racket_pos"], time_new)
    body_center = resample_array(time, src["body_center"], time_new) if "body_center" in src else None
    body_right_axis = resample_array(time, src["body_right_axis"], time_new) if "body_right_axis" in src else None
    ball_vel = compute_velocity(ball_pos, time_new)
    racket_vel = compute_velocity(racket_pos, time_new)
    if "racket_quat" in src:
        racket_quat = resample_quat_xyzw(time, src["racket_quat"], time_new)
    else:
        racket_quat = np.full((len(ball_pos), 4), np.nan)
    if body_center is None:
        body_center_out = np.full((len(ball_pos), 3), np.nan)
    else:
        body_center_out = body_center
    if body_right_axis is None:
        body_right_axis_out = np.full((len(ball_pos), 3), np.nan)
    else:
        body_right_axis_out = _normalize_vectors(body_right_axis)
    racket_omega = compute_angular_velocity(racket_quat, time_new)
    valid_mask = np.isfinite(ball_pos).all(axis=1) & np.isfinite(racket_pos).all(axis=1)
    hit_pos = ball_pos[hit_index]
    racket_pose_at_hit = np.concatenate([racket_pos[hit_index], racket_quat[hit_index]])
    racket_vel_at_hit = racket_vel[hit_index]
    ball_in_vel = _safe_mean_time_window(ball_vel, time_rel, -0.08, -0.03)
    ball_out_vel = _safe_mean_time_window(ball_vel, time_rel, 0.03, 0.08)
    success_label = judge_success(
        ball_pos,
        ball_vel,
        hit_index=hit_index,
        table_config=table_config,
    )
    stroke_label = classify_stroke_type(
        racket_pos=racket_pos,
        ball_vel=ball_vel,
        hit_index=hit_index,
        body_center=body_center,
        body_right_axis=body_right_axis_out,
        handedness=handedness,
    )

    racket_quat_available = bool(np.isfinite(racket_quat).all())
    racket_omega_available = bool(np.isfinite(racket_omega).all())
    ball_speed = np.linalg.norm(ball_vel, axis=1)
    racket_speed = np.linalg.norm(racket_vel, axis=1)
    racket_omega_norm = np.linalg.norm(racket_omega, axis=1)
    ball_speed_reasonable = bool(np.nanmax(ball_speed) < max_ball_speed_mps)
    racket_speed_reasonable = bool(np.nanmax(racket_speed) < max_racket_speed_mps)
    racket_omega_reasonable = bool(np.nanmax(racket_omega_norm) < max_racket_omega_radps)
    quality_flags = {
        **hit_metadata["quality_flags"],
        "cleaning_usable": bool(cleaning_usable),
        "missing_near_hit": bool(not np.all(valid_mask[max(0, hit_index - 10): min(len(valid_mask), hit_index + 11)])),
        "racket_quat_available": racket_quat_available,
        "racket_omega_available": racket_omega_available,
        "ball_speed_reasonable": ball_speed_reasonable,
        "racket_speed_reasonable": racket_speed_reasonable,
        "racket_omega_reasonable": racket_omega_reasonable,
        "max_ball_speed_mps": float(np.nanmax(ball_speed)),
        "max_racket_speed_mps": float(np.nanmax(racket_speed)),
        "max_racket_omega_radps": float(np.nanmax(racket_omega_norm)),
        "coordinate_transform_available": False,
        "coordinate_frame": "motive_global_m",
        **success_label.flags,
        "stroke_type": stroke_label.stroke_type,
        **stroke_label.flags,
    }
    quality_flags["usable_for_training"] = bool(
        cleaning_usable
        and hit_metadata["valid_hit"]
        and not quality_flags["missing_near_hit"]
        and ball_speed_reasonable
        and racket_speed_reasonable
        and racket_omega_reasonable
    )

    source = {
        "source_npz": source_npz,
        "debug_npz": debug_npz,
        "source_csv": source_csv,
        "source_bvh": source_bvh,
        "racket": racket,
        "candidate": candidate,
        "handedness": handedness,
        "hit_metadata": hit_metadata,
        "pre_hit_s": pre_hit_s,
        "post_hit_s": post_hit_s,
        "target_fps": target_fps,
    }

    np.savez(
        output_npz,
        episode_id=np.asarray(episode_id),
        time=time_new,
        time_rel=time_rel,
        valid_mask=valid_mask.astype(np.int8),
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        ball_vel_resampled_from_source=raw_ball_vel,
        racket_pos=racket_pos,
        racket_quat=racket_quat,
        racket_vel=racket_vel,
        racket_omega=racket_omega,
        body_center=body_center_out,
        body_right_axis=body_right_axis_out,
        hit_index=np.asarray(hit_index, dtype=np.int64),
        hit_time=np.asarray(hit_time, dtype=float),
        hit_pos=hit_pos,
        racket_pose_at_hit=racket_pose_at_hit,
        racket_vel_at_hit=racket_vel_at_hit,
        ball_in_vel=ball_in_vel,
        ball_out_vel=ball_out_vel,
        landing_pos=success_label.landing_pos,
        success=np.asarray(success_label.success, dtype=np.int8),
        stroke_type=np.asarray(stroke_label.stroke_type),
        quality_flags_json=np.asarray(json.dumps(quality_flags, ensure_ascii=False)),
        source_json=np.asarray(json.dumps(source, ensure_ascii=False)),
        dist=resample_array(time, dbg["dist"][:, None], time_new)[:, 0],
        ball_dv=resample_array(time, dbg["ball_dv"][:, None], time_new)[:, 0],
        score=resample_array(time, dbg["score"][:, None], time_new)[:, 0],
    )

    metadata = {
        "episode_id": episode_id,
        "sample_path": output_npz,
        "source": source,
        "frames": int(len(time_new)),
        "hit_index": int(hit_index),
        "hit_time": hit_time,
        "hit_time_rel": float(hit_metadata["hit_time_rel"]),
        "quality_flags": quality_flags,
        "fields": {
            "quat_order": "xyzw",
            "position_unit": "m",
            "coordinate_frame": "motive_global_m",
            "sample_fps": target_fps,
            "success_encoding": "-1 unknown, 0 false, 1 true",
        },
    }
    with open(output_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return SampleExportResult(
        episode_id=episode_id,
        sample_path=output_npz,
        metadata_path=output_metadata,
        source_csv=source_csv,
        source_bvh=source_bvh,
        hit_index=int(hit_index),
        hit_time=hit_time,
        hit_time_rel=float(hit_metadata["hit_time_rel"]),
        frames=int(len(time_new)),
        usable_for_training=quality_flags["usable_for_training"],
        quality_flags=quality_flags,
    )
