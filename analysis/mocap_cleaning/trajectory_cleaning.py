"""Position trajectory cleaning utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from analysis.mocap_cleaning.derivative import compute_velocity


@dataclass
class GapSegment:
    start: int
    end: int
    duration_s: float
    filled: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["filled"] = bool(data["filled"])
        return data


@dataclass
class TrajectoryCleaningReport:
    frames: int
    raw_valid_ratio: float
    cleaned_valid_ratio: float
    outlier_frames: int
    short_gaps_filled: int
    long_gaps: int
    max_gap_s: float
    raw_max_speed_mps: float
    cleaned_max_speed_mps: float
    cleaned_p95_speed_mps: float
    usable: bool
    reasons: list[str]
    gaps: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finite_rows(pos: np.ndarray) -> np.ndarray:
    return np.isfinite(pos).all(axis=1)


def find_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) segments where mask is True."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def mark_speed_outliers(pos: np.ndarray, time: np.ndarray, max_speed_mps: float) -> tuple[np.ndarray, np.ndarray]:
    """Mark frames after implausible jumps as NaN.

    A speed outlier between frame i and i+1 usually means the newly observed
    frame has jumped to another marker/rigid body. Mark i+1 so short isolated
    jumps can be filled by interpolation.
    """
    cleaned = pos.copy()
    speed = np.linalg.norm(compute_velocity(pos, time), axis=1)
    outlier = np.isfinite(speed) & (speed > max_speed_mps)
    outlier[0] = False
    cleaned[outlier] = np.nan
    return cleaned, outlier


def fill_short_gaps(pos: np.ndarray, time: np.ndarray, max_gap_s: float) -> tuple[np.ndarray, list[GapSegment]]:
    filled = pos.copy()
    invalid = ~finite_rows(filled)
    segments: list[GapSegment] = []
    valid_rows = finite_rows(filled)

    for start, end in find_true_segments(invalid):
        left = start - 1
        right = end
        if start == 0 or right >= len(filled):
            duration = float(time[end - 1] - time[start]) if end > start else 0.0
            segments.append(GapSegment(start=start, end=end, duration_s=duration, filled=False))
            continue

        duration = float(time[end - 1] - time[start]) if end > start else 0.0
        can_fill = duration <= max_gap_s and valid_rows[left] and valid_rows[right]
        if can_fill:
            for dim in range(filled.shape[1]):
                filled[start:end, dim] = np.interp(time[start:end], [time[left], time[right]], [filled[left, dim], filled[right, dim]])
            valid_rows[start:end] = True
        segments.append(GapSegment(start=start, end=end, duration_s=duration, filled=can_fill))

    return filled, segments


def clean_position_trajectory(
    pos: np.ndarray,
    time: np.ndarray,
    max_speed_mps: float,
    max_gap_s: float,
    min_valid_ratio: float = 0.95,
) -> tuple[np.ndarray, TrajectoryCleaningReport]:
    raw_valid = finite_rows(pos)
    raw_speed = np.linalg.norm(compute_velocity(pos, time), axis=1) if len(time) else np.asarray([])
    outlier_marked, outlier_mask = mark_speed_outliers(pos, time, max_speed_mps=max_speed_mps)
    filled, gaps = fill_short_gaps(outlier_marked, time, max_gap_s=max_gap_s)
    cleaned_valid = finite_rows(filled)
    cleaned_speed = np.linalg.norm(compute_velocity(filled, time), axis=1) if len(time) else np.asarray([])
    finite_cleaned_speed = cleaned_speed[np.isfinite(cleaned_speed)]

    long_gaps = [gap for gap in gaps if not gap.filled]
    reasons: list[str] = []
    cleaned_valid_ratio = float(np.mean(cleaned_valid)) if len(cleaned_valid) else 0.0
    if cleaned_valid_ratio < min_valid_ratio:
        reasons.append(f"cleaned valid ratio below threshold ({cleaned_valid_ratio:.3f} < {min_valid_ratio:.3f})")
    if long_gaps:
        reasons.append(f"{len(long_gaps)} unfilled gaps remain")
    if len(finite_cleaned_speed) and float(np.nanmax(finite_cleaned_speed)) > max_speed_mps:
        reasons.append(f"cleaned trajectory still exceeds {max_speed_mps:.1f} m/s")

    report = TrajectoryCleaningReport(
        frames=len(pos),
        raw_valid_ratio=float(np.mean(raw_valid)) if len(raw_valid) else 0.0,
        cleaned_valid_ratio=cleaned_valid_ratio,
        outlier_frames=int(np.sum(outlier_mask)),
        short_gaps_filled=sum(1 for gap in gaps if gap.filled),
        long_gaps=len(long_gaps),
        max_gap_s=max((gap.duration_s for gap in gaps), default=0.0),
        raw_max_speed_mps=float(np.nanmax(raw_speed)) if len(raw_speed) else 0.0,
        cleaned_max_speed_mps=float(np.nanmax(finite_cleaned_speed)) if len(finite_cleaned_speed) else 0.0,
        cleaned_p95_speed_mps=float(np.nanpercentile(finite_cleaned_speed, 95)) if len(finite_cleaned_speed) else 0.0,
        usable=not reasons,
        reasons=reasons or ["trajectory cleaned successfully"],
        gaps=[gap.to_dict() for gap in gaps],
    )
    return filled, report
