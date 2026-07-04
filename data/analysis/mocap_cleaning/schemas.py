"""Shared data structures for mocap cleaning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PoseSeries:
    """Rigid pose time series.

    Positions remain in the source unit until an explicit conversion step.
    Quaternions use xyzw order, matching Motive CSV Rotation X/Y/Z/W.
    """

    pos: np.ndarray
    quat_xyzw: np.ndarray | None = None


@dataclass
class RawTrial:
    source_path: str
    take_name: str
    fps: float
    time: np.ndarray
    position_unit: str
    coordinate_space: str
    rigid_bodies: dict[str, PoseSeries] = field(default_factory=dict)
    bones: dict[str, PoseSeries] = field(default_factory=dict)
    markers: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityColumns:
    kind: str
    name: str
    pos: tuple[int, int, int] | None = None
    quat_xyzw: tuple[int, int, int, int] | None = None

