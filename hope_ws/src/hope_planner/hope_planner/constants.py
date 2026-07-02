"""Physical constants and tuning parameters for the HOPE planner.

Values follow the HOPE canonical world frame (origin at P1 near-side left
corner, X toward P2, Y left, Z up) and ITTF regulation table dimensions.
See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 2.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TableParams:
    """ITTF regulation table dimensions in the HOPE canonical frame."""

    length: float = 2.74          # m, along X
    width: float = 1.525          # m, along -Y
    height: float = 0.76          # m, table surface above floor
    net_x: float = 1.37           # m, net position along X
    net_height: float = 0.1525    # m, net height above table surface
    net_overhang: float = 0.15    # m, net extends past each table edge in Y


@dataclass
class BallPhysics:
    """Aerodynamic and restitution parameters.

    Calibrated from recorded ball trajectories by fitting observed
    accelerations and bounce velocity ratios (see calibrate_ball_physics).
    The HITTER paper uses 15 trajectories; more data improves robustness.
    """

    k: float = 0.09375           # quadratic drag coefficient (1/m): a_drag = -k|v|v
    C_h: float = 0.649           # table tangential velocity retention beta_table
    C_v: float = 0.906           # table normal restitution e_table
    g: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    radius: float = 0.02         # ball radius, 40 mm diameter
    mass: float = 0.0027         # 2.7 g


@dataclass
class PlannerConfig:
    """Planner tuning parameters."""

    # State estimation
    poly_order: int = 2           # polynomial fit order
    fit_window: int = 31          # number of position samples for velocity fit
    mocap_hz: float = 360.0       # motion capture frame rate

    # Trajectory prediction
    dt_integrate: float = 0.001   # integration time step (s)
    max_predict_time: float = 2.0  # max forward prediction horizon (s)
    bounce_z_tol: float = 0.005   # z threshold for bounce detection (m)

    # Racket planning
    x_hit: float = 0.0            # virtual hitting plane X coordinate (m)
    target_land: np.ndarray = field(
        default_factory=lambda: np.array([2.055, -0.7625, 0.0])
    )                             # center of opponent's half
    delta_t_flight: float = 0.5   # desired post-strike flight time (s)
    C_r: float = 0.842            # ball-racket normal restitution e_racket
    racket_radius: float = 0.075  # 7.5 cm paddle radius
    racket_marker_plane_gap: float = 0.0365  # ball center to racket marker-plane contact gap (m)
