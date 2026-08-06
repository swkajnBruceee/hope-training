"""Isaac hit-state-draw package.

Renders the externally-computed (C++ ROS workspace) hit plan in the Isaac Sim
viewport via ``debug_draw``. Mirrors :mod:`show.trajectory` 1:1:

* Wire format is defined by ``hit_state_udp_bridge.cpp`` (magic ``"HITS"``,
  header + 6 vec3s); see ``show/trajectory/overlay.py`` for the matching
  trajectory protocol.
* Only the ``debug_draw`` backend is supported; it survives
  ``--rendering_mode performance`` and never touches USD prims.

This module is *display only*; the actual solver / planning logic lives in
``hope_ws/src/solver``.
"""

from .overlay import (
    IsaacHitOverlay,
    HitOverlayConfig,
)

__all__ = [
    "IsaacHitOverlay",
    "HitOverlayConfig",
]
