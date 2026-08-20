"""Stage5-parity post-contact return metrics for the deterministic V2-B2 smoke."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

NET_X = 1.37
NET_HEIGHT = 0.1525
BALL_RADIUS = 0.02
OPPONENT_X_BOUNDS = (1.40, 2.71)
OPPONENT_Y_BOUNDS = (-1.495, -0.03)
TABLE_SURFACE_BALL_Z_BOUNDS = (-0.005, 0.060)
RETURNED_FORWARD_VX_MIN = 0.20
UPWARD_AFTER_BOUNCE_VZ_MIN = 0.05


@dataclass(frozen=True)
class NetCrossing:
    position: np.ndarray
    alpha: float
    clearance: float
    clears_net: bool


def detect_outgoing_net_crossing(prev_position, position, velocity, *, contact_seen: bool, net_seen: bool = False):
    """Reproduce Stage5 ball-centre interpolation and radius-aware net gate."""
    p0 = np.asarray(prev_position, dtype=np.float64)
    p1 = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    if not contact_seen or net_seen or not (p0[0] < NET_X <= p1[0]) or not (v[0] > 0.0):
        return None
    alpha = float(np.clip((NET_X - p0[0]) / max(p1[0] - p0[0], 1e-9), 0.0, 1.0))
    crossing = p0 + alpha * (p1 - p0)
    crossing[0] = NET_X
    clearance = float(crossing[2] - (NET_HEIGHT + BALL_RADIUS))
    return NetCrossing(crossing, alpha, clearance, clearance > 0.0)


def detect_opponent_table_bounce(position, velocity, *, contact_seen: bool, bounce_seen: bool = False) -> bool:
    """Reproduce Stage5's first valid opponent-side post-contact bounce gate."""
    p = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    return bool(
        contact_seen and not bounce_seen
        and OPPONENT_X_BOUNDS[0] <= p[0] <= OPPONENT_X_BOUNDS[1]
        and OPPONENT_Y_BOUNDS[0] <= p[1] <= OPPONENT_Y_BOUNDS[1]
        and TABLE_SURFACE_BALL_Z_BOUNDS[0] <= p[2] <= TABLE_SURFACE_BALL_Z_BOUNDS[1]
        and v[0] > RETURNED_FORWARD_VX_MIN
        and v[2] > UPWARD_AFTER_BOUNCE_VZ_MIN
    )


def legal_return(contact: bool, cross_net: bool, opponent_table_landing: bool) -> bool:
    return bool(contact and cross_net and opponent_table_landing)

