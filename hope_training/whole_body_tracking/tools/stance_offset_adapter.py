#!/usr/bin/env python3
"""Pure geometry contract for world hit points and pre-hit base placement.

This module deliberately does not choose a robot comfort region. A calibrated
``canonical_reach_offset_b`` must be supplied by the caller. The adapter keeps
the world hit point unchanged and computes the horizontal base placement that
would put that point at the calibrated base-frame reach offset.

This is a pre-positioning calculation, not a walking controller. It does not
move feet during the swing and it does not modify target velocity or normal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


def _vec3(value: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    if len(value) != 3 or not all(math.isfinite(float(x)) for x in value):
        raise ValueError(f"{name} must be a finite 3-vector")
    return tuple(float(x) for x in value)


@dataclass(frozen=True)
class BasePose:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(x)) for x in (self.x, self.y, self.z, self.yaw)):
            raise ValueError("base pose must be finite")


@dataclass(frozen=True)
class HorizontalRegion:
    """Optional calibrated base-frame region for no-step upper-body execution."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, point_b: tuple[float, float, float]) -> bool:
        return (
            self.x_min <= point_b[0] <= self.x_max
            and self.y_min <= point_b[1] <= self.y_max
        )


@dataclass(frozen=True)
class StanceOffsetResult:
    hit_point_w: tuple[float, float, float]
    current_target_b: tuple[float, float, float]
    target_base_pose_w: BasePose
    relocated_target_b: tuple[float, float, float]
    required_base_xy: tuple[float, float]
    required_horizontal_offset: tuple[float, float]
    status: str


def _rotate_z(yaw: float, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    x, y, z = vector
    return (c * x - s * y, s * x + c * y, z)


def _inverse_rotate_z(yaw: float, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return _rotate_z(-yaw, vector)


def adapt_world_hit_point(
    *,
    hit_point_w: tuple[float, float, float],
    current_base_w: BasePose,
    canonical_reach_offset_b: tuple[float, float, float],
    comfort_region_b: Optional[HorizontalRegion] = None,
    target_yaw: Optional[float] = None,
) -> StanceOffsetResult:
    """Compute a stationary pre-positioning target without changing hit geometry.

    ``canonical_reach_offset_b`` is the desired hit point relative to the
    relocated base. The target base position is therefore:

        base_target_w = hit_point_w - R_wb(yaw) * reach_offset_b

    The current base-relative target is reported for deciding whether a step
    is needed. If no comfort region is provided, status is ``unclassified``;
    the function never invents a comfort threshold.
    """

    hit = _vec3(hit_point_w, "hit_point_w")
    reach = _vec3(canonical_reach_offset_b, "canonical_reach_offset_b")
    yaw = current_base_w.yaw if target_yaw is None else float(target_yaw)
    if not math.isfinite(yaw):
        raise ValueError("target_yaw must be finite")

    current_delta = (
        hit[0] - current_base_w.x,
        hit[1] - current_base_w.y,
        hit[2] - current_base_w.z,
    )
    current_target_b = _inverse_rotate_z(current_base_w.yaw, current_delta)
    rotated_reach_w = _rotate_z(yaw, reach)
    target_base = BasePose(
        hit[0] - rotated_reach_w[0],
        hit[1] - rotated_reach_w[1],
        current_base_w.z,
        yaw,
    )
    relocated_delta = (
        hit[0] - target_base.x,
        hit[1] - target_base.y,
        hit[2] - target_base.z,
    )
    relocated_target_b = _inverse_rotate_z(yaw, relocated_delta)
    required_offset = (target_base.x - current_base_w.x, target_base.y - current_base_w.y)

    if comfort_region_b is None:
        status = "unclassified"
    elif comfort_region_b.contains(current_target_b):
        status = "inside_upper_body_region"
    else:
        status = "requires_stance_offset"

    return StanceOffsetResult(
        hit_point_w=hit,
        current_target_b=current_target_b,
        target_base_pose_w=target_base,
        relocated_target_b=relocated_target_b,
        required_base_xy=(target_base.x, target_base.y),
        required_horizontal_offset=required_offset,
        status=status,
    )
