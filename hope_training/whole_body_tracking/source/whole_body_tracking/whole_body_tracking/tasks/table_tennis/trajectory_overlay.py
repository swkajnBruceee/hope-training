"""Isaac viewport overlay for live table-tennis ball trajectory prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np


def _ensure_hope_planner_on_path() -> None:
    """Make the ROS planner package importable from the Isaac training shell."""
    for parent in Path(__file__).resolve().parents:
        src = parent / "hope_ws" / "src"
        candidate = src / "hope_planner"
        if (candidate / "hope_planner").exists():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return


_ensure_hope_planner_on_path()

from hope_planner.ball_state_estimator import BallStateEstimator  # noqa: E402
from hope_planner.constants import BallPhysics, PlannerConfig, TableParams  # noqa: E402

# Heavy planner path kept for future WBC/decision logic only.
# Do not enable these imports inside the live debug overlay unless the call is
# moved out of the per-segment/per-frame hot loop.
# from hope_planner.racket_target_planner import RacketTargetPlanner  # noqa: E402
# from hope_planner.ball_trajectory_predictor import StrikeTarget  # noqa: E402


# NOTE: hit_plane is currently kept only for backwards-compatible CLI/config.
# The simplified overlay no longer uses a fixed hit-plane or manually-defined
# hit zone. Each predicted trajectory segment is evaluated as a candidate strike.
@dataclass
class RacketHitPlane:
    """Vertical racket marker plane, parallel to the net.

    Kept for backwards-compatible CLI/config only. The simplified overlay no
    longer uses a fixed hit-plane or manually-defined hit zone — every
    predicted trajectory segment is evaluated as a candidate strike.
    """

    x: float = 0.0
    tolerance: float = 0.12
    marker_plane_gap: float = 0.0365
    y_min: float = -1.525
    y_max: float = 0.0
    z_min: float = 0.06
    z_max: float = 0.80

    @property
    def ball_center_x(self) -> float:
        return self.x + self.marker_plane_gap

    def contains_yz(self, p: np.ndarray) -> bool:
        return (
            self.y_min <= p[1] <= self.y_max
            and self.z_min <= p[2] <= self.z_max
        )


@dataclass
class TrajectoryOverlayConfig:
    enabled: bool = True
    env_index: int = 0
    draw_period_s: float = 0.03
    horizon_s: float = 1.2
    sample_stride: int = 8
    reset_jump_m: float = 0.5
    line_width: float = 5.0
    red_line_width: float = 14.0
    red_point_size: float = 22.0
    hit_plane: RacketHitPlane = field(default_factory=RacketHitPlane)


class IsaacTrajectoryOverlay:
    """Draw one live future trajectory in the Isaac viewport.

    Before the first P1-side table bounce, the predicted future trajectory is
    drawn black. After that bounce, the same prediction is drawn green. Red
    segments are candidate strike locations that clear the net and first land
    on the opponent half after a stationary vertical racket reflection.
    """

    def __init__(self, cfg: TrajectoryOverlayConfig | None = None):
        self.cfg = cfg or TrajectoryOverlayConfig()
        self.physics = BallPhysics()
        self.config = PlannerConfig(max_predict_time=self.cfg.horizon_s)
        self.table = TableParams()
        self.estimator = BallStateEstimator(self.config)
        # NOTE: RacketTargetPlanner is no longer constructed here — the heavy
        # inverse planner (24× fixed-point + 5-candidate dt enumeration) is
        # preserved for the real WBC decision pipeline but must not run inside
        # the per-frame debug-draw loop. To restore the original logic, uncomment
        # the import above and re-add:
        #   self.target_planner = RacketTargetPlanner(self.physics, self.config, self.table)
        self._draw = self._acquire_debug_draw()
        self._active_after_p1_bounce = False
        self._last_p: np.ndarray | None = None
        self._z_hist: list[float | None] = [None, None, None]
        self._p_hist: list[np.ndarray | None] = [None, None, None]
        self._last_draw_t = -float("inf")
        self._reported_red_segments = False
    @property
    def available(self) -> bool:
        return self._draw is not None

    def clear(self) -> None:
        if self._draw is not None:
            self._draw.clear_lines()
            self._draw.clear_points()

    def close(self) -> None:
        self.clear()

    def push(self, t: float, p: np.ndarray) -> None:
        if not self.cfg.enabled or self._draw is None:
            return

        p = np.asarray(p, dtype=float)
        if self._last_p is not None and np.linalg.norm(p - self._last_p) > self.cfg.reset_jump_m:
            self._reset_tracking(clear=True)

        self._last_p = p.copy()
        self._push_bounce_history(p)
        if self._detect_p1_bounce():
            self._active_after_p1_bounce = True
            self.estimator.reset()

        self.estimator.push(t, p)

        if not self.estimator.ready:
            self.clear()
            return
        if t - self._last_draw_t < self.cfg.draw_period_s:
            return

        p_est, v_est, t_est = self.estimator.estimate()

        points, velocities, times = self._simulate_future(p_est, v_est, t_est)
        base_color = (0.0, 1.0, 0.15, 1.0) if self._active_after_p1_bounce else (0.0, 0.0, 0.0, 1.0)
        self._draw_trajectory(points, velocities, times, t, base_color)
        self._last_draw_t = t

    def _reset_tracking(self, clear: bool) -> None:
        self.estimator.reset()
        self._active_after_p1_bounce = False
        self._z_hist = [None, None, None]
        self._p_hist = [None, None, None]
        self._last_draw_t = -float("inf")
        if clear:
            self.clear()

    def _push_bounce_history(self, p: np.ndarray) -> None:
        self._z_hist[0] = self._z_hist[1]
        self._z_hist[1] = self._z_hist[2]
        self._z_hist[2] = float(p[2])
        self._p_hist[0] = self._p_hist[1]
        self._p_hist[1] = self._p_hist[2]
        self._p_hist[2] = p.copy()

    def _detect_p1_bounce(self) -> bool:
        z_pp, z_p, z_c = self._z_hist
        p_prev = self._p_hist[1]
        if z_pp is None or z_p is None or z_c is None or p_prev is None:
            return False

        contact_z = self.physics.radius + 0.02
        descending_then_rising = z_pp > z_p and z_c > z_p
        near_table = z_p <= contact_z
        on_p1_table = (
            -self.physics.radius <= p_prev[0] <= self.table.net_x
            and -self.table.width - self.physics.radius <= p_prev[1] <= self.physics.radius
        )
        return descending_then_rising and near_table and on_p1_table

    def _simulate_future(
        self, p0: np.ndarray, v0: np.ndarray, t0: float
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        dt = self.config.dt_integrate
        max_steps = int(self.cfg.horizon_s / dt)
        p = p0.copy()
        v = v0.copy()
        t = t0
        points = [p.copy()]
        velocities = [v.copy()]
        times = [t]

        for step in range(max_steps):
            a = self._flight_acceleration(v)
            p_new = p + v * dt + 0.5 * a * dt**2
            v_new = v + a * dt
            t += dt

            if p_new[2] < self.physics.radius and v_new[2] < 0.0:
                if self._is_on_table(p_new):
                    dz = p[2] - p_new[2]
                    frac = np.clip(
                        (p[2] - self.physics.radius) / dz if dz > 1e-9 else 0.5,
                        0.0,
                        1.0,
                    )
                    p_bounce = p + frac * (p_new - p)
                    p_bounce[2] = self.physics.radius
                    v_at_bounce = v + a * (frac * dt)
                    v_post = self._apply_bounce(v_at_bounce)
                    remaining_dt = (1.0 - frac) * dt
                    a_post = self._flight_acceleration(v_post)
                    p_new = p_bounce + v_post * remaining_dt + 0.5 * a_post * remaining_dt**2
                    v_new = v_post + a_post * remaining_dt
                else:
                    break

            p = p_new
            v = v_new

            if (step + 1) % self.cfg.sample_stride == 0:
                points.append(p.copy())
                velocities.append(v.copy())
                times.append(t)

            if p[2] < -0.1 or p[0] < -0.6 or p[0] > self.table.length + 0.6:
                break

        if not np.allclose(points[-1], p):
            points.append(p.copy())
            velocities.append(v.copy())
            times.append(t)
        return points, velocities, times

    def _draw_trajectory(
        self,
        points: list[np.ndarray],
        velocities: list[np.ndarray],
        times: list[float],
        t: float = 0.0,
        base_color: tuple[float, float, float, float] = (0.0, 1.0, 0.15, 1.0),
    ) -> None:
        self.clear()

        if len(points) < 2:
            return

        starts = [tuple(p.tolist()) for p in points[:-1]]
        ends = [tuple(p.tolist()) for p in points[1:]]
        self._draw.draw_lines(starts, ends, [base_color] * len(starts), [self.cfg.line_width] * len(starts))

        red_starts: list[tuple[float, float, float]] = []
        red_ends: list[tuple[float, float, float]] = []
        red_points: list[tuple[float, float, float]] = []
        for i in range(len(points) - 1):
            if self._is_hittable_segment(
                points[i],
                points[i + 1],
                velocities[i],
                velocities[i + 1],
                times[i],
                times[i + 1],
            ):
                red_starts.append(tuple(points[i].tolist()))
                red_ends.append(tuple(points[i + 1].tolist()))
                midpoint = 0.5 * (points[i] + points[i + 1])
                red_points.extend(
                    (
                        tuple(points[i].tolist()),
                        tuple(midpoint.tolist()),
                        tuple(points[i + 1].tolist()),
                    )
                )

        if red_starts:
            red = [(1.0, 0.05, 0.0, 1.0)] * len(red_starts)
            self._draw.draw_lines(red_starts, red_ends, red, [self.cfg.red_line_width] * len(red_starts))
            self._draw.draw_points(
                red_points,
                [(1.0, 0.0, 0.0, 1.0)] * len(red_points),
                [self.cfg.red_point_size] * len(red_points),
            )
            if not self._reported_red_segments:
                print(
                    f"[play_table_tennis] trajectory overlay red segments active: {len(red_starts)} segment(s)"
                )
                self._reported_red_segments = True

    def _is_hittable_segment(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        t0: float,
        t1: float,
    ) -> bool:
        """Return True if hitting the ball at this predicted segment can return it.

        Each green trajectory segment is treated as one candidate strike
        opportunity. The candidate strike point is the segment start point
        ``p0``, and the incoming velocity is ``v0``. A stationary vertical
        racket is placed there, parallel to the net, with face normal in +X.
        The reflected outgoing trajectory is then integrated forward. The
        segment is marked red only if the reflected ball clears the net and
        first lands on the opponent's (P2) half of the table.
        """
        # ------------------------------------------------------------------
        # Preserved heavy inverse-planning path (disabled for live overlay).
        #
        # This was the old/complex hittability check:
        #
        #   from hope_planner.ball_trajectory_predictor import StrikeTarget
        #   from hope_planner.racket_target_planner import RacketTargetPlanner
        #
        #   strike = StrikeTarget(
        #       p_ball=p.copy(),
        #       v_ball=v.copy(),
        #       t_strike=t0,
        #       num_bounces=1,
        #       valid=True,
        #   )
        #   command = self.target_planner.plan(strike)
        #   return bool(
        #       command.valid
        #       and command.clears_net
        #       and not command.bypasses_net_posts
        #   )
        #
        # It is intentionally NOT used in the debug-draw loop because
        # RacketTargetPlanner.plan() is expensive: it performs inverse planning,
        # fixed-point iterations, net clearance checks, and candidate flight-time
        # retries. Keep this path for future WBC/decision-level logic only.
        # ------------------------------------------------------------------

        p = p0.copy()
        v = v0.copy()

        n = np.array([1.0, 0.0, 0.0], dtype=float)
        C_r = self.config.C_r

        # Physical sanity check only: the ball must be moving toward the paddle
        # face. This is not an artificial hit-zone filter; it only prevents
        # reflecting a ball that is already moving away from the paddle in +X.
        v_n_scalar = float(np.dot(v, n))
        if v_n_scalar >= -1e-6:
            return False

        # Stationary vertical racket reflection:
        # normal component reverses and is scaled by restitution;
        # tangential components are preserved.
        v_out = v - (1.0 + C_r) * v_n_scalar * n

        return self._can_clear_and_land(p, v_out)

    def _can_clear_and_land(self, p_strike: np.ndarray, v_outgoing: np.ndarray) -> bool:
        """Return True iff outgoing ball clears the net and first lands in P2 court.

        This is the strict physical check for the simplified overlay:

          1. The reflected ball must cross ``x = net_x`` while moving in +X and
             at that instant satisfy the net-height and net-Y extents.
          2. The reflected ball must first land on the table surface inside the
             opponent's (P2) half of the table.

        No ground bounce. No "bounce forever" continuation. If the ball lands
        outside the table, falls below the play floor, exits the play volume,
        or simply runs out of horizon before the conditions are both met, the
        segment is not hittable.
        """
        net_x = self.table.net_x
        z_net = self.table.net_height
        margin = 0.03  # m, same safety margin used in RacketTargetPlanner
        y_net_min = -self.table.width - self.table.net_overhang
        y_net_max = self.table.net_overhang

        r = self.physics.radius

        # P2/opponent half extents in the HOPE table coordinate system:
        # x in [net_x, length], y in [-width, 0]. We pad by ball radius on the
        # outer edges so a ball whose center lands exactly on the edge counts
        # as "in bounds on the opponent's half".
        x_p2_min = net_x - r
        x_p2_max = self.table.length + r
        y_p2_min = -self.table.width - r
        y_p2_max = r

        dt = self.config.dt_integrate
        max_t = self.config.max_predict_time

        p = p_strike.copy()
        v = v_outgoing.copy()

        elapsed = 0.0
        cleared_net = False

        while elapsed < max_t - 1e-12:
            a = self._flight_acceleration(v)
            p_next = p + v * dt + 0.5 * a * dt ** 2
            v_next = v + a * dt

            # 1) Net crossing capture. Do NOT return True here — we still need
            # to confirm that the ball first lands on the opponent's half.
            if not cleared_net:
                crosses_net = (p[0] - net_x) * (p_next[0] - net_x) <= 0.0
                moving_toward_p2 = p_next[0] > p[0]
                if crosses_net and moving_toward_p2:
                    dx = p_next[0] - p[0]
                    frac = ((net_x - p[0]) / dx) if abs(dx) > 1e-9 else 0.0
                    frac = float(np.clip(frac, 0.0, 1.0))
                    p_net = p + frac * (p_next - p)

                    if p_net[2] <= z_net + margin:
                        return False
                    if not (y_net_min <= p_net[1] <= y_net_max):
                        return False

                    cleared_net = True

            # 2) First table contact / landing check. Once the reflected ball
            # first reaches table height while descending, decide whether it
            # landed in P2. Do not keep bouncing forever.
            if p_next[2] <= r and v_next[2] < 0.0:
                dz = p[2] - p_next[2]
                frac = ((p[2] - r) / dz) if abs(dz) > 1e-9 else 0.5
                frac = float(np.clip(frac, 0.0, 1.0))
                p_land = p + frac * (p_next - p)
                p_land[2] = r

                if not cleared_net:
                    return False

                landed_p2 = (
                    x_p2_min <= p_land[0] <= x_p2_max
                    and y_p2_min <= p_land[1] <= y_p2_max
                )
                if landed_p2:
                    return True
                return False

            # 3) Out-of-play / impossible states. No ground bounce: a ball that
            # falls below the floor or exits the play volume cannot satisfy the
            # "first lands in P2" condition, so we treat it as not hittable.
            if p_next[2] < -0.1:
                return False
            if p_next[0] < -0.6 or p_next[0] > self.table.length + 0.6:
                return False

            p = p_next
            v = v_next
            elapsed += dt

        return False

    def _flight_acceleration(self, v: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(v)
        return -self.physics.k * speed * v + self.physics.g

    def _apply_bounce(self, v: np.ndarray) -> np.ndarray:
        return np.diag([self.physics.C_h, self.physics.C_h, -self.physics.C_v]) @ v

    def _is_on_table(self, p: np.ndarray) -> bool:
        r = self.physics.radius
        return -r <= p[0] <= self.table.length + r and -self.table.width - r <= p[1] <= r

    def _acquire_debug_draw(self):
        try:
            from isaacsim.util.debug_draw import _debug_draw

            return _debug_draw.acquire_debug_draw_interface()
        except Exception:
            pass

        try:
            import omni.kit.app

            manager = omni.kit.app.get_app().get_extension_manager()
            manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
            from isaacsim.util.debug_draw import _debug_draw

            return _debug_draw.acquire_debug_draw_interface()
        except Exception:
            pass

        try:
            from omni.isaac.debug_draw import _debug_draw

            return _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:
            print(f"[play_table_tennis] trajectory overlay disabled: {exc!r}")
            return None
