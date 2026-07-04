"""Ball candidate validation for Motive rigid bodies."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from analysis.mocap_cleaning.derivative import compute_velocity


@dataclass
class BallCandidateReport:
    name: str
    frames: int
    valid_ratio: float
    position_min_m: list[float]
    position_max_m: list[float]
    position_range_m: list[float]
    median_height_m: float
    height_range_m: float
    median_speed_mps: float
    p95_speed_mps: float
    max_speed_mps: float
    robust_p95_speed_mps: float
    speed_outlier_ratio: float
    max_frame_jump_m: float
    static_ratio: float
    high_speed_ratio: float
    near_racket_events: dict[str, int]
    decision: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_rows(pos: np.ndarray) -> np.ndarray:
    return np.isfinite(pos).all(axis=1)


def analyze_ball_candidate(
    name: str,
    pos_m: np.ndarray,
    time: np.ndarray,
    racket_positions_m: dict[str, np.ndarray] | None = None,
) -> BallCandidateReport:
    valid = _finite_rows(pos_m)
    valid_ratio = float(np.mean(valid)) if len(valid) else 0.0
    finite_pos = pos_m[valid]

    if len(finite_pos) == 0:
        return BallCandidateReport(
            name=name,
            frames=len(pos_m),
            valid_ratio=0.0,
            position_min_m=[float("nan")] * 3,
            position_max_m=[float("nan")] * 3,
            position_range_m=[float("nan")] * 3,
            median_height_m=float("nan"),
            height_range_m=float("nan"),
            median_speed_mps=float("nan"),
            p95_speed_mps=float("nan"),
            max_speed_mps=float("nan"),
            robust_p95_speed_mps=float("nan"),
            speed_outlier_ratio=0.0,
            max_frame_jump_m=float("nan"),
            static_ratio=1.0,
            high_speed_ratio=0.0,
            near_racket_events={},
            decision="invalid",
            reasons=["no finite positions"],
        )

    vel = compute_velocity(pos_m, time)
    speed = np.linalg.norm(vel, axis=1)
    finite_speed = speed[np.isfinite(speed)]
    pos_min = np.nanmin(pos_m, axis=0)
    pos_max = np.nanmax(pos_m, axis=0)
    pos_range = pos_max - pos_min

    near_racket_events: dict[str, int] = {}
    if racket_positions_m:
        for racket, racket_pos in racket_positions_m.items():
            finite_pair = _finite_rows(pos_m) & _finite_rows(racket_pos)
            if not np.any(finite_pair):
                near_racket_events[racket] = 0
                continue
            dist = np.linalg.norm(pos_m[finite_pair] - racket_pos[finite_pair], axis=1)
            near_racket_events[racket] = int(np.sum(dist < 0.20))

    median_speed = float(np.nanmedian(finite_speed)) if len(finite_speed) else float("nan")
    p95_speed = float(np.nanpercentile(finite_speed, 95)) if len(finite_speed) else float("nan")
    max_speed = float(np.nanmax(finite_speed)) if len(finite_speed) else float("nan")
    robust_speed = finite_speed[finite_speed < 80.0] if len(finite_speed) else finite_speed
    robust_p95_speed = float(np.nanpercentile(robust_speed, 95)) if len(robust_speed) else float("nan")
    speed_outlier_ratio = float(np.mean(finite_speed >= 80.0)) if len(finite_speed) else 0.0
    frame_jumps = np.linalg.norm(np.diff(pos_m, axis=0), axis=1)
    max_frame_jump = float(np.nanmax(frame_jumps)) if len(frame_jumps) else 0.0
    static_ratio = float(np.mean(finite_speed < 0.05)) if len(finite_speed) else 1.0
    high_speed_ratio = float(np.mean(finite_speed > 1.0)) if len(finite_speed) else 0.0
    height_range = float(pos_range[2])

    reasons: list[str] = []
    if valid_ratio < 0.95:
        reasons.append(f"valid ratio is low ({valid_ratio:.3f})")
    if static_ratio > 0.80:
        reasons.append(f"mostly static ({static_ratio:.3f} below 0.05 m/s)")
    if height_range < 0.05:
        reasons.append(f"height range is too small ({height_range:.3f} m)")
    if p95_speed < 0.5:
        reasons.append(f"p95 speed is too low for a ball ({p95_speed:.3f} m/s)")
    if speed_outlier_ratio > 0.001:
        reasons.append(f"contains tracking jumps ({speed_outlier_ratio:.3%} speeds >= 80 m/s)")
    elif max_speed > 80.0:
        reasons.append(f"has isolated implausible speed ({max_speed:.3f} m/s)")

    if not reasons and high_speed_ratio > 0.01:
        decision = "valid"
        reasons.append("dynamic rigid body with plausible height and speed variation")
    elif static_ratio < 0.80 and height_range >= 0.05 and p95_speed >= 0.5:
        decision = "uncertain"
        reasons.append("dynamic enough to inspect, but not enough evidence for valid ball")
    else:
        decision = "invalid"
        if not reasons:
            reasons.append("does not show ball-like motion")

    return BallCandidateReport(
        name=name,
        frames=len(pos_m),
        valid_ratio=valid_ratio,
        position_min_m=[float(x) for x in pos_min],
        position_max_m=[float(x) for x in pos_max],
        position_range_m=[float(x) for x in pos_range],
        median_height_m=float(np.nanmedian(pos_m[:, 2])),
        height_range_m=height_range,
        median_speed_mps=median_speed,
        p95_speed_mps=p95_speed,
        max_speed_mps=max_speed,
        robust_p95_speed_mps=robust_p95_speed,
        speed_outlier_ratio=speed_outlier_ratio,
        max_frame_jump_m=max_frame_jump,
        static_ratio=static_ratio,
        high_speed_ratio=high_speed_ratio,
        near_racket_events=near_racket_events,
        decision=decision,
        reasons=reasons,
    )
