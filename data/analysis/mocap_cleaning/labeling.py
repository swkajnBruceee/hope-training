"""Landing and success labeling utilities for cleaned mocap samples."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SuccessLabel:
    success: int
    landing_pos: np.ndarray
    flags: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["landing_pos"] = self.landing_pos.tolist()
        return out


@dataclass(frozen=True)
class StrokeLabel:
    stroke_type: str
    flags: dict[str, Any]


def _table_enabled(table_config: dict[str, Any] | None) -> bool:
    if not table_config:
        return False
    return bool(table_config.get("enabled", False))


def _required_table_values_available(table_config: dict[str, Any]) -> bool:
    required = (
        "table_z_m",
        "z_tolerance_m",
        "opponent_x_min_m",
        "opponent_x_max_m",
        "y_min_m",
        "y_max_m",
    )
    return all(table_config.get(k) is not None for k in required)


def detect_landing(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    *,
    start_index: int,
    table_z_m: float,
    z_tolerance_m: float,
) -> tuple[int | None, np.ndarray]:
    """Detect the first post-hit table bounce from z proximity and vz sign flip."""
    if len(ball_pos) != len(ball_vel):
        raise ValueError("ball_pos and ball_vel length mismatch")
    start = max(1, int(start_index) + 1)
    for i in range(start, len(ball_pos)):
        if not (np.isfinite(ball_pos[i]).all() and np.isfinite(ball_vel[i]).all() and np.isfinite(ball_vel[i - 1]).all()):
            continue
        near_table = abs(float(ball_pos[i, 2]) - table_z_m) <= z_tolerance_m
        vertical_bounce = ball_vel[i - 1, 2] < 0.0 and ball_vel[i, 2] > 0.0
        if near_table and vertical_bounce:
            return i, ball_pos[i].copy()
    return None, np.full(3, np.nan)


def judge_success(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    *,
    hit_index: int,
    table_config: dict[str, Any] | None,
) -> SuccessLabel:
    """Return reliable success labels only when table calibration is configured."""
    base_flags: dict[str, Any] = {
        "table_config_available": _table_enabled(table_config),
        "landing_detected": False,
        "success_label_reliable": False,
    }
    if not _table_enabled(table_config):
        base_flags["success_label_reason"] = "missing_table_calibration"
        return SuccessLabel(success=-1, landing_pos=np.full(3, np.nan), flags=base_flags)

    assert table_config is not None
    if not _required_table_values_available(table_config):
        base_flags["success_label_reason"] = "incomplete_table_calibration"
        return SuccessLabel(success=-1, landing_pos=np.full(3, np.nan), flags=base_flags)

    landing_index, landing_pos = detect_landing(
        ball_pos,
        ball_vel,
        start_index=hit_index,
        table_z_m=float(table_config["table_z_m"]),
        z_tolerance_m=float(table_config["z_tolerance_m"]),
    )
    if landing_index is None:
        base_flags["success_label_reason"] = "landing_not_detected_in_episode"
        return SuccessLabel(success=-1, landing_pos=landing_pos, flags=base_flags)

    x, y, _ = landing_pos
    in_opponent_table = (
        float(table_config["opponent_x_min_m"]) <= x <= float(table_config["opponent_x_max_m"])
        and float(table_config["y_min_m"]) <= y <= float(table_config["y_max_m"])
    )
    flags = {
        **base_flags,
        "landing_detected": True,
        "landing_index": int(landing_index),
        "success_label_reliable": True,
        "success_label_reason": "table_landing_rule",
    }
    return SuccessLabel(success=int(in_opponent_table), landing_pos=landing_pos, flags=flags)


def classify_stroke_type(
    *,
    racket_pos: np.ndarray,
    ball_vel: np.ndarray,
    hit_index: int,
    body_center: np.ndarray | None = None,
    body_right_axis: np.ndarray | None = None,
    handedness: str = "right",
    min_lateral_offset_m: float = 0.08,
) -> StrokeLabel:
    """Classify forehand/backhand from racket side relative to the player's torso."""
    flags: dict[str, Any] = {
        "stroke_label_source": "unknown",
        "stroke_label_confidence": 0.0,
        "stroke_label_reason": "missing_body_reference",
    }
    if body_center is None or body_right_axis is None:
        return StrokeLabel(stroke_type="unknown", flags=flags)
    if not (0 <= hit_index < len(racket_pos)):
        flags["stroke_label_reason"] = "invalid_hit_index"
        return StrokeLabel(stroke_type="unknown", flags=flags)
    if not (
        np.isfinite(racket_pos[hit_index]).all()
        and np.isfinite(body_center[hit_index]).all()
        and np.isfinite(body_right_axis[hit_index]).all()
    ):
        flags["stroke_label_reason"] = "nonfinite_body_reference_at_hit"
        return StrokeLabel(stroke_type="unknown", flags=flags)

    right_axis = body_right_axis[hit_index]
    axis_norm = float(np.linalg.norm(right_axis))
    if axis_norm < 1e-6:
        flags["stroke_label_reason"] = "degenerate_body_right_axis"
        return StrokeLabel(stroke_type="unknown", flags=flags)
    right_axis = right_axis / axis_norm

    rel = racket_pos[hit_index] - body_center[hit_index]
    lateral_offset = float(np.dot(rel, right_axis))
    confidence = min(1.0, abs(lateral_offset) / max(min_lateral_offset_m * 3.0, 1e-6))
    flags = {
        "stroke_label_source": "rule",
        "stroke_label_confidence": float(confidence),
        "stroke_label_reason": "body_lateral_offset_rule",
        "stroke_lateral_offset_m": lateral_offset,
        "stroke_handedness": handedness,
    }
    if abs(lateral_offset) < min_lateral_offset_m:
        flags["stroke_label_reason"] = "low_lateral_separation"
        return StrokeLabel(stroke_type="unknown", flags=flags)

    if handedness == "left":
        stroke_type = "forehand" if lateral_offset < 0.0 else "backhand"
    else:
        stroke_type = "forehand" if lateral_offset > 0.0 else "backhand"

    # A weak serve heuristic is intentionally not emitted as a reliable label here:
    # DATA260703 windows start near rallies, and no toss/hand-release signal is validated.
    if len(ball_vel):
        pre_start = max(0, hit_index - 29)
        pre_end = max(pre_start, hit_index - 10)
        if pre_end > pre_start:
            flags["stroke_pre_hit_ball_speed_mps"] = float(np.nanmean(np.linalg.norm(ball_vel[pre_start:pre_end], axis=1)))
    return StrokeLabel(stroke_type=stroke_type, flags=flags)
