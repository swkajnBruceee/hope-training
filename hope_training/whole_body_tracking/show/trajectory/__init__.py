"""Python trajectory-draw package.

Renders externally-computed (C++ ROS workspace) table-tennis trajectories in
the Isaac Sim viewport. This module is the *display* half of the pipeline;
trajectory estimation/prediction live in the C++ workspace
(`hope_ws/src/trajectory`).
"""

from .overlay import (
    IsaacTrajectoryOverlay,
    TrajectoryOverlayConfig,
)

__all__ = [
    "IsaacTrajectoryOverlay",
    "TrajectoryOverlayConfig",
]
