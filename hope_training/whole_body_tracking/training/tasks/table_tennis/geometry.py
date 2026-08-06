"""Canonical HOPE table-tennis geometry, expressed in the HOPE world frame.

The HOPE world frame (ROS 2 REP-103, identical to the planner / mocap reference docs):

* **Origin** — the near-side **left** corner of the table *surface*, from Player One's (P1) perspective.
* **X** — forward, toward Player Two (P2), along the table length:  ``x in [0, 2.74] m``.
* **Y** — left, from P1's perspective, along the table width:        ``y in [-1.525, 0] m``.
* **Z** — up; **z = 0 is the table surface** (the floor is therefore at ``z = -0.76 m``).

Key landmarks (see ``data/mocap/HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md`` §2.1
and ``hope_ws/.../hope_world_frame.yaml``)::

    Origin (P1 near-side left corner)   ( 0.0  ,  0.0   ,  0.0 )
    Net center line                     ( 1.37 , -0.7625,  0.0 )
    P1 half center                      ( 0.685, -0.7625,  0.0 )
    P2 half center                      ( 2.055, -0.7625,  0.0 )
    Floor directly below origin         ( 0.0  ,  0.0   , -0.76)

This module is the **single source of truth** for the geometry used to build the Isaac Lab scene.
It is pure Python (no Isaac / torch imports) so it can be imported anywhere and unit-tested without a
simulator. The numbers mirror ``hope_ws/src/hope_planner/hope_planner/constants.py`` (ITTF regulation
table) so the simulator and the model-based planner share one consistent world.

The simulation world frame is set **equal to** this HOPE frame: each environment's local origin sits
at the table-surface height (HOPE z = 0), so an asset's environment-local position *is* its HOPE-frame
position. With multiple environments, every environment is an independent court with its own HOPE
origin anchored at that environment's origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

##
# ITTF regulation table (meters). Mirrors hope_planner.constants.TableParams.
##
TABLE_LENGTH: float = 2.74          # along +X
TABLE_WIDTH: float = 1.525          # along -Y
TABLE_HEIGHT: float = 0.76          # table surface above the floor
TABLE_THICKNESS: float = 0.05       # visual/collision slab thickness of the table top

NET_X: float = 1.37                 # net plane position along X (== TABLE_LENGTH / 2)
NET_HEIGHT: float = 0.1525          # net height above the table surface
NET_OVERHANG: float = 0.15          # net extends this far past each table edge in Y
NET_THICKNESS: float = 0.01         # net slab thickness along X

LINE_WIDTH: float = 0.02            # painted boundary / center line width
LINE_THICKNESS: float = 0.001       # painted line slab thickness (sits just above the surface)

##
# Ball (40 mm ITTF). Mirrors hope_planner.constants.BallPhysics.
##
BALL_RADIUS: float = 0.02           # 40 mm diameter
BALL_MASS: float = 0.0027           # 2.7 g

##
# Coordinate landmarks in the HOPE frame.
##
ORIGIN: tuple[float, float, float] = (0.0, 0.0, 0.0)
TABLE_CENTER: tuple[float, float, float] = (TABLE_LENGTH / 2.0, -TABLE_WIDTH / 2.0, 0.0)
NET_CENTER: tuple[float, float, float] = (NET_X, -TABLE_WIDTH / 2.0, 0.0)
P1_HALF_CENTER: tuple[float, float, float] = (TABLE_LENGTH / 4.0, -TABLE_WIDTH / 2.0, 0.0)
P2_HALF_CENTER: tuple[float, float, float] = (3.0 * TABLE_LENGTH / 4.0, -TABLE_WIDTH / 2.0, 0.0)
FLOOR_Z: float = -TABLE_HEIGHT      # the floor surface, in HOPE z

##
# Robot nominal standing pose, P1 side (HOPE frame).
# From HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md: "Robot nominal standing position
# ~ (-0.5, -0.7625, floor)". X < 0 puts the robot behind the near table end; Y centers it on the
# table width; the pelvis height is added on top of FLOOR_Z by the robot-specific config.
##
P1_STAND_X: float = -0.5
P1_STAND_Y: float = -TABLE_WIDTH / 2.0   # -0.7625, centered on table width

# Robot nominal standing pose, P2 side (HOPE frame), mirrored across the far
# table end and facing P1.  This is used by the first fixed-base P2 retargeting
# pass before full-body balance is introduced.
P2_STAND_X: float = TABLE_LENGTH + 0.5
P2_STAND_Y: float = -TABLE_WIDTH / 2.0


def table_top_center() -> tuple[float, float, float]:
    """Center of the table-top collision/visual slab (its top face is at HOPE z = 0)."""
    return (TABLE_CENTER[0], TABLE_CENTER[1], -TABLE_THICKNESS / 2.0)


def net_center() -> tuple[float, float, float]:
    """Center of the net slab (spans z in [0, NET_HEIGHT])."""
    return (NET_X, -TABLE_WIDTH / 2.0, NET_HEIGHT / 2.0)


def net_size() -> tuple[float, float, float]:
    """Full extents (x, y, z) of the net slab, including the lateral overhang past the table edges."""
    return (NET_THICKNESS, TABLE_WIDTH + 2.0 * NET_OVERHANG, NET_HEIGHT)


def table_top_size() -> tuple[float, float, float]:
    """Full extents (x, y, z) of the table-top slab."""
    return (TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS)


@dataclass
class ServeConfig:
    """Parameters for the scripted ball serve used to reset / visualize ball flight.

    The ball is spawned over one half of the table and launched toward the other side with a slight
    downward component, so a reset shows a full flight -> table bounce -> rising arc. All positions are
    in the HOPE frame; velocities are in m/s in the HOPE frame. Ranges are sampled uniformly per reset.
    """

    # Default visualization serve: one fixed ball line toward the robot's forehand-middle hitting zone.
    # Lower toss height and slightly faster incoming speed make the trajectory easier to inspect.
    pos_x_range: tuple[float, float] = (2.22, 2.22)
    pos_y_range: tuple[float, float] = (-0.9625, -0.5625)
    pos_z_range: tuple[float, float] = (0.43, 0.43)
    # Fixed launch velocity toward P1 (-X) with zero lateral drift and a small upward component so the
    # first bounce lands consistently on the forehand-middle lane.
    vel_x_range: tuple[float, float] = (-6.5, -5.5)
    vel_y_range: tuple[float, float] = (-0.5, 0.5)
    vel_z_range: tuple[float, float] = (0.08, 0.08)
    # Optional initial spin (rad/s) about each HOPE axis; only matters if Magnus is enabled.
    spin_range: tuple[float, float] = (0.0, 0.0)
    spin_x_range: tuple[float, float] | None = None
    spin_y_range: tuple[float, float] | None = None
    spin_z_range: tuple[float, float] | None = None


@dataclass
class BounceMaterials:
    """PhysX contact-material parameters for the ball and the static surfaces.

    NOTE on restitution: PhysX applies a single (normal) restitution per contact pair, combined from
    the two materials' coefficients. With ``restitution_combine_mode="multiply"`` everywhere, the
    effective normal restitution of a ball<->surface bounce is ``ball.restitution * surface.restitution``.
    We therefore set the *ball* restitution to the table normal-restitution target ``C_v = 0.906`` and the
    *table* restitution to 1.0, so a ball<->table bounce reproduces C_v, while the net (0.1) kills the
    ball and the floor (0.4) bounces it modestly.

    The planner's tangential retention ``C_h = 0.649`` and ball<->racket ``C_r = 0.842`` are *not*
    reproduced exactly: tangential (horizontal) velocity loss is governed by PhysX friction, not by a
    restitution coefficient, and the racket inherits the robot's contact material. Matching C_h / C_r
    exactly is a calibration TODO (a sim analogue of ``hope_planner.calibrate_ball_physics``); the
    values below give a visibly correct bounce as the first-pass default. All knobs live here so a
    calibration pass only edits this one place.
    """

    # Ball (dynamic). restitution == C_v target (vertical/normal restitution).
    ball_restitution: float = 0.906
    ball_static_friction: float = 1.0
    ball_dynamic_friction: float = 1.0
    # Table top.
    table_restitution: float = 1.0
    table_static_friction: float = 0.162
    table_dynamic_friction: float = 0.162
    # Floor.
    floor_restitution: float = 0.4
    floor_static_friction: float = 0.8
    floor_dynamic_friction: float = 0.8
    # Net (low restitution so the ball dies on contact).
    net_restitution: float = 0.1
    net_static_friction: float = 0.5
    net_dynamic_friction: float = 0.5


@dataclass
class OutOfBoundsBox:
    """Axis-aligned region (HOPE frame) outside which the ball is considered dead / out of play.

    Generous margins around the table so the ball can fly well past either player before the episode
    is reset. Used by the termination / serve-reset logic.
    """

    x: tuple[float, float] = (-2.0, 4.74)
    y: tuple[float, float] = (-3.0, 1.5)
    # A ball resting on the floor has center z = FLOOR_Z + BALL_RADIUS = -0.74; trigger just above that
    # (-0.73) so a grounded/missed ball ends the episode promptly instead of idling until timeout.
    z: tuple[float, float] = (FLOOR_Z + BALL_RADIUS + 0.01, 3.0)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {"x": self.x, "y": self.y, "z": self.z}
