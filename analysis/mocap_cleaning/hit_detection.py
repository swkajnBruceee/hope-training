"""Hit detection from cleaned ball and racket trajectories."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from analysis.mocap_cleaning.derivative import compute_velocity


@dataclass
class HitDetectionResult:
    hit_index: int | None
    hit_time: float | None
    hit_time_rel: float | None
    score_at_hit: float | None
    dist_at_hit_m: float | None
    racket_speed_at_hit_mps: float | None
    ball_dv_at_hit_mps: float | None
    valid_hit: bool
    quality_flags: dict[str, bool]
    reason: str
    debug: dict[str, np.ndarray]

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("debug", None)
        return data


def _safe_normalized(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values, dtype=float)
    denom = float(np.nanmax(finite))
    if denom <= 1.0e-8:
        return np.zeros_like(values, dtype=float)
    return values / denom


def detect_hit_index(
    time: np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    racket_pos: np.ndarray,
    max_distance_m: float,
    distance_ok_m: float,
    min_racket_speed_mps: float,
    min_ball_dv_mps: float,
    weights: dict[str, float],
) -> HitDetectionResult:
    finite = np.isfinite(ball_pos).all(axis=1) & np.isfinite(racket_pos).all(axis=1)
    if len(time) == 0 or not np.any(finite):
        return HitDetectionResult(
            hit_index=None,
            hit_time=None,
            hit_time_rel=None,
            score_at_hit=None,
            dist_at_hit_m=None,
            racket_speed_at_hit_mps=None,
            ball_dv_at_hit_mps=None,
            valid_hit=False,
            quality_flags={"has_finite_data": False},
            reason="no finite ball/racket data",
            debug={},
        )

    racket_vel = compute_velocity(racket_pos, time)
    dist = np.linalg.norm(ball_pos - racket_pos, axis=1)
    racket_speed = np.linalg.norm(racket_vel, axis=1)
    ball_dv = np.zeros(len(time), dtype=float)
    if len(time) > 2:
        ball_dv[1:-1] = np.linalg.norm(ball_vel[2:] - ball_vel[:-2], axis=1)

    dist_score = np.exp(-dist / 0.08)
    dv_score = _safe_normalized(ball_dv)
    racket_score = _safe_normalized(racket_speed)
    score = (
        float(weights.get("distance", 0.5)) * dist_score
        + float(weights.get("ball_dv", 0.3)) * dv_score
        + float(weights.get("racket_speed", 0.2)) * racket_score
    )
    valid_candidate = finite & (dist < max_distance_m)
    score_masked = score.copy()
    score_masked[~valid_candidate] = -np.inf

    if not np.any(valid_candidate):
        min_idx = int(np.nanargmin(dist))
        return HitDetectionResult(
            hit_index=None,
            hit_time=None,
            hit_time_rel=None,
            score_at_hit=None,
            dist_at_hit_m=float(dist[min_idx]),
            racket_speed_at_hit_mps=float(racket_speed[min_idx]),
            ball_dv_at_hit_mps=float(ball_dv[min_idx]),
            valid_hit=False,
            quality_flags={
                "has_finite_data": True,
                "hit_distance_ok": False,
                "racket_speed_ok": bool(racket_speed[min_idx] > min_racket_speed_mps),
                "ball_velocity_change_ok": bool(ball_dv[min_idx] > min_ball_dv_mps),
            },
            reason=f"no candidate frame with distance < {max_distance_m:.3f} m",
            debug={
                "dist": dist,
                "ball_dv": ball_dv,
                "racket_speed": racket_speed,
                "dist_score": dist_score,
                "dv_score": dv_score,
                "racket_score": racket_score,
                "score": score,
                "valid_candidate": valid_candidate.astype(np.int8),
            },
        )

    hit_index = int(np.argmax(score_masked))
    flags = {
        "has_finite_data": True,
        "hit_distance_ok": bool(dist[hit_index] < distance_ok_m),
        "racket_speed_ok": bool(racket_speed[hit_index] > min_racket_speed_mps),
        "ball_velocity_change_ok": bool(ball_dv[hit_index] > min_ball_dv_mps),
    }
    valid_hit = all(flags.values())
    reason = "valid hit" if valid_hit else "failed quality flags: " + ", ".join(k for k, v in flags.items() if not v)
    return HitDetectionResult(
        hit_index=hit_index,
        hit_time=float(time[hit_index]),
        hit_time_rel=float(time[hit_index] - time[0]),
        score_at_hit=float(score[hit_index]),
        dist_at_hit_m=float(dist[hit_index]),
        racket_speed_at_hit_mps=float(racket_speed[hit_index]),
        ball_dv_at_hit_mps=float(ball_dv[hit_index]),
        valid_hit=valid_hit,
        quality_flags=flags,
        reason=reason,
        debug={
            "dist": dist,
            "ball_dv": ball_dv,
            "racket_speed": racket_speed,
            "dist_score": dist_score,
            "dv_score": dv_score,
            "racket_score": racket_score,
            "score": score,
            "valid_candidate": valid_candidate.astype(np.int8),
        },
    )

