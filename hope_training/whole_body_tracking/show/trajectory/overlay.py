"""Isaac viewport display for externally-computed table-tennis trajectories.

Trajectory estimation and prediction belong to the C++ ROS workspace
(`hope_ws/src/trajectory`). This module only receives sampled trajectory points
over UDP and draws them in the Isaac viewport.

Only one draw backend is currently active:

* ``debug_draw`` (preferred): uses Isaac's ``isaacsim.util.debug_draw`` (or
  legacy ``omni.isaac.debug_draw`` / ``omni.debugdraw``) interface, which
  renders through the debug overlay and survives
  ``--rendering_mode performance``.

Both the legacy ``usd_basis_curves`` backend (BasisCurves with a ``widths``
primvar -- Hydra rejects this with "Unrecognized primvar 'widths'") and the
``usd_mesh_segments`` USD Cube fallback have been disabled while we stabilize
``debug_draw``. If ``debug_draw`` cannot be acquired the overlay is simply
turned off -- no silent fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import struct
import time

from shared_debug_draw import (  # type: ignore[import-not-found]
    drop_debug_draw_owner,
    register_debug_draw,
    update_debug_draw_owner,
)


@dataclass
class TrajectoryOverlayConfig:
    enabled: bool = True
    env_index: int = 0
    draw_period_s: float = 0.03
    horizon_s: float = 1.2
    line_width: float = 6.0
    udp_host: str = "127.0.0.1"
    udp_port: int = 19532
    hit_udp_host: str = "127.0.0.1"
    hit_udp_port: int = 19533
    return_line_width: float = 4.5
    return_horizon_s: float = 1.2
    return_dt_s: float = 0.001
    return_drag_coefficient: float = 0.09375
    return_gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    return_ball_radius: float = 0.02
    return_table_normal_restitution: float = 0.906
    return_table_tangential_retention: float = 0.649
    return_table_length: float = 2.74
    return_table_width: float = 1.525

    # --- new tuning knobs ---
    use_debug_draw: bool = True            # try omni.isaac.debug_draw first
    usd_curve_width_m: float = 0.03        # USD BasisCurves width (meters)
    stale_keep_s: float = 0.3              # keep last valid frame this long
    log_period_s: float = 1.0              # throttled diagnostic log cadence


class _TrajectoryUdpReceiver:
    _HEADER = struct.Struct("<4sIII")
    _POINT = struct.Struct("<3d")
    _MAGIC = b"HTRJ"

    def __init__(self, host: str, port: int):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.bind((host, int(port)))
        self._last_sequence: int | None = None

    def close(self) -> None:
        try:
            self._socket.close()
        except Exception:
            pass

    def poll(self):
        """Drain all currently-available UDP packets, return the latest valid one.

        Returns a tuple ``(points, color_state, sequence, total_drained)`` or
        ``None`` if no valid packet was received.
        """
        latest = None
        total = 0
        while True:
            try:
                payload, _ = self._socket.recvfrom(65535)
            except BlockingIOError:
                break

            total += 1
            packet = self._parse(payload)
            if packet is not None:
                latest = packet
        if latest is None:
            return None
        points, color_state, sequence = latest
        return points, color_state, sequence, total

    def _parse(self, payload: bytes):
        if len(payload) < self._HEADER.size:
            return None

        magic, sequence, color_state, count = self._HEADER.unpack_from(payload, 0)
        if magic != self._MAGIC:
            return None
        if self._last_sequence is not None and sequence == self._last_sequence:
            return None
        expected_size = self._HEADER.size + int(count) * self._POINT.size
        if len(payload) < expected_size:
            return None

        offset = self._HEADER.size
        points = []
        for _ in range(int(count)):
            points.append(self._POINT.unpack_from(payload, offset))
            offset += self._POINT.size

        self._last_sequence = int(sequence)
        return points, bool(color_state), int(sequence)


class _HitPlanUdpReceiver:
    _HEADER = struct.Struct("<4sIII")
    _VEC3 = struct.Struct("<3d")
    _MAGIC = b"HITS"

    def __init__(self, host: str, port: int):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.bind((host, int(port)))
        self._last_sequence: int | None = None

    def close(self) -> None:
        try:
            self._socket.close()
        except Exception:
            pass

    def poll(self):
        latest = None
        while True:
            try:
                payload, _ = self._socket.recvfrom(65535)
            except BlockingIOError:
                break
            parsed = self._parse(payload)
            if parsed is not None:
                latest = parsed
        return latest

    def _parse(self, payload: bytes):
        if len(payload) < self._HEADER.size:
            return None
        magic, sequence, valid, has_plan = self._HEADER.unpack_from(payload, 0)
        if magic != self._MAGIC:
            return None
        if self._last_sequence is not None and int(sequence) == self._last_sequence:
            return None
        self._last_sequence = int(sequence)
        if not int(valid) or not int(has_plan):
            return None
        expected_size = self._HEADER.size + 6 * self._VEC3.size
        if len(payload) < expected_size:
            return None
        offset = self._HEADER.size
        vecs = []
        for _ in range(6):
            vecs.append(self._VEC3.unpack_from(payload, offset))
            offset += self._VEC3.size
        return {
            "hit_position": vecs[0],
            "target_land": vecs[1],
            "ball_v_in": vecs[2],
            "ball_v_out": vecs[3],
            "racket_vel": vecs[4],
            "racket_normal": vecs[5],
            "sequence": int(sequence),
        }


def _sample_return_trajectory(plan: dict, cfg: TrajectoryOverlayConfig):
    p = [float(plan["hit_position"][0]), float(plan["hit_position"][1]), float(plan["hit_position"][2])]
    v = [float(plan["ball_v_out"][0]), float(plan["ball_v_out"][1]), float(plan["ball_v_out"][2])]
    g = cfg.return_gravity
    dt = max(1e-4, float(cfg.return_dt_s))
    horizon = max(0.0, float(cfg.return_horizon_s))
    max_steps = int(horizon / dt)
    radius = float(cfg.return_ball_radius)
    c_h = float(cfg.return_table_tangential_retention)
    c_v = float(cfg.return_table_normal_restitution)
    table_length = float(cfg.return_table_length)
    table_width = float(cfg.return_table_width)
    drag_k = float(cfg.return_drag_coefficient)

    points = [tuple(p)]
    for step in range(max_steps):
        speed = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        ax = -drag_k * speed * v[0] + float(g[0])
        ay = -drag_k * speed * v[1] + float(g[1])
        az = -drag_k * speed * v[2] + float(g[2])
        p_new = [
            p[0] + v[0] * dt + 0.5 * ax * dt * dt,
            p[1] + v[1] * dt + 0.5 * ay * dt * dt,
            p[2] + v[2] * dt + 0.5 * az * dt * dt,
        ]
        v_new = [
            v[0] + ax * dt,
            v[1] + ay * dt,
            v[2] + az * dt,
        ]

        if p_new[2] <= radius and v_new[2] < 0.0:
            on_table = (
                0.0 <= p_new[0] <= table_length and
                -table_width <= p_new[1] <= 0.0
            )
            if on_table:
                dz = p[2] - p_new[2]
                frac = ((p[2] - radius) / dz) if abs(dz) > 1e-9 else 0.5
                frac = max(0.0, min(1.0, frac))
                p_bounce = [
                    p[0] + frac * (p_new[0] - p[0]),
                    p[1] + frac * (p_new[1] - p[1]),
                    radius,
                ]
                v_bounce = [
                    v[0] + ax * (frac * dt),
                    v[1] + ay * (frac * dt),
                    v[2] + az * (frac * dt),
                ]
                v_post = [c_h * v_bounce[0], c_h * v_bounce[1], -c_v * v_bounce[2]]
                remaining_dt = (1.0 - frac) * dt
                speed_post = math.sqrt(v_post[0] * v_post[0] + v_post[1] * v_post[1] + v_post[2] * v_post[2])
                ax_post = -drag_k * speed_post * v_post[0] + float(g[0])
                ay_post = -drag_k * speed_post * v_post[1] + float(g[1])
                az_post = -drag_k * speed_post * v_post[2] + float(g[2])
                p_new = [
                    p_bounce[0] + v_post[0] * remaining_dt + 0.5 * ax_post * remaining_dt * remaining_dt,
                    p_bounce[1] + v_post[1] * remaining_dt + 0.5 * ay_post * remaining_dt * remaining_dt,
                    p_bounce[2] + v_post[2] * remaining_dt + 0.5 * az_post * remaining_dt * remaining_dt,
                ]
                v_new = [
                    v_post[0] + ax_post * remaining_dt,
                    v_post[1] + ay_post * remaining_dt,
                    v_post[2] + az_post * remaining_dt,
                ]
                points.append(tuple(p_bounce))
            else:
                break

        p = p_new
        v = v_new
        points.append(tuple(p))
        if p[2] < -0.1 or p[0] < -0.7 or p[0] > 3.5 or p[1] < -1.7 or p[1] > 0.2:
            break
        if step > 2 and math.dist(points[-1], points[-2]) < 1e-6:
            break
    return points


class _UsdTrajectoryDrawAdapter:
    """DEPRECATED BasisCurves backend -- kept only as a stub to fail loudly.

    The current renderer (Hydra under ``--rendering_mode performance``) rejects
    the ``widths`` primvar emitted by BasisCurves with::

        Unrecognized primvar 'widths' detected in geometry
            '/World/HOPE_TrajectoryOverlay/trajectory_curve'

    so this backend must NOT be used. The active backend is either
    ``debug_draw`` (preferred) or ``_UsdMeshSegmentAdapter`` (USD fallback).
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "_UsdTrajectoryDrawAdapter (USD BasisCurves) is deprecated because "
            "Hydra rejects its 'widths' primvar. Use debug_draw or "
            "_UsdMeshSegmentAdapter instead."
        )


# ---------------------------------------------------------------------------
# Debug-draw backend (preferred for performance renderer)
# ---------------------------------------------------------------------------


def _acquire_debug_draw_iface(log_tag: str = "[trajectory_overlay_isaac]"):
    """Try every known debug_draw entry point in order. Print each attempt.

    Returns the acquired interface or ``None`` if every path failed.
    """
    attempts = [
        ("isaacsim.util.debug_draw", "Isaac Sim 4.5+"),
        ("omni.isaac.debug_draw",   "Isaac Sim 2023.1 -- 4.x"),
        ("omni.debugdraw",          "legacy"),
    ]
    last_err = None
    for mod_name, hint in attempts:
        try:
            mod = __import__(mod_name, fromlist=["_debug_draw"])
            iface_mod = getattr(mod, "_debug_draw")
        except Exception as exc:
            print(f"{log_tag} try debug_draw backend: {mod_name} ({hint}) "
                  f"failed: {exc!r}")
            last_err = exc
            continue
        try:
            iface = iface_mod.acquire_debug_draw_interface()
        except Exception as exc:
            print(f"{log_tag} try debug_draw backend: {mod_name} "
                  f"acquire_debug_draw_interface failed: {exc!r}")
            last_err = exc
            continue
        print(f"{log_tag} try debug_draw backend: {mod_name} ({hint}) OK")
        return iface, mod_name
    print(f"{log_tag} all debug_draw backends failed; last_err={last_err!r}")
    return None, None


class _DebugDrawTrajectoryAdapter:
    """Renders the trajectory via Isaac's debug_draw overlay.

    This backend does NOT touch USD prims at all, so it survives
    ``--rendering_mode performance`` and never triggers Hydra primvar warnings.

    Stale-curve handling lives here (instead of in the top-level overlay) so
    each backend can choose its own stale policy.
    """

    def __init__(self, iface, iface_module: str, line_width_px: float = 6.0,
                 stale_keep_s: float = 0.3):
        self._iface = iface
        self._iface_module = iface_module
        self._line_width_px = float(line_width_px)
        self._stale_keep_s = float(stale_keep_s)
        self._last_valid_draw_t = 0.0
        self._draw_calls = 0
        self._draw_fail_count = 0   # circuit breaker for draw_lines failures
        self.prim_path = "(debug_draw)"
        self._owner_id = "trajectory_overlay"
        register_debug_draw(self._iface, self._iface_module)
        # Probe the API once so we know whether draw_lines accepts the
        # (starts, ends, colors, widths) signature we want to call.
        self._probe_api()

    def _probe_api(self) -> None:
        try:
            sig_info = (
                f"type={type(self._iface).__name__} "
                f"has_draw_lines={hasattr(self._iface, 'draw_lines')} "
                f"has_clear_lines={hasattr(self._iface, 'clear_lines')} "
                f"has_clear={hasattr(self._iface, 'clear')}"
            )
        except Exception as exc:
            sig_info = f"probe_failed: {exc!r}"
        print(f"[trajectory_overlay_isaac] debug_draw iface: {sig_info} "
              f"module={self._iface_module}")

    def clear(self) -> None:
        drop_debug_draw_owner(self._owner_id)

    def draw_polyline(self, points, color, width_px: float = None, owner_id: str | None = None) -> bool:
        """Draw a polyline.

        Returns True iff the line geometry was actually pushed to the
        viewport. Implements its own stale-keep logic so the top-level overlay
        doesn't have to.

        Isaac Sim 4.5 debug_draw.draw_lines signature:
            draw_lines(
                List[carb.Float3],    # starts
                List[carb.Float3],    # ends
                List[carb.ColorRgba], # colors (RGBA, not RGB)
                List[float],          # widths (one per segment)
            )
        """
        import time as _time
        now = _time.monotonic()

        # Circuit breaker: after a few failures stop hammering draw_lines so
        # the log doesn't blow up.
        if self._draw_fail_count > 3:
            return False

        if len(points) < 2:
            # Hold the previous frame for stale_keep_s, then clear.
            if (self._last_valid_draw_t > 0.0
                    and now - self._last_valid_draw_t < self._stale_keep_s):
                return False
            drop_debug_draw_owner(owner_id or self._owner_id)
            return False

        # ---- Basic sanity filter: drop NaN / Inf / out-of-scene points. ----
        # The C++ trajectory_overlay_udp_node already filters these server
        # side, but we double-check here so the viewer never sees a NaN or a
        # point teleported across the scene.
        def _is_valid(p) -> bool:
            try:
                x, y, z = float(p[0]), float(p[1]), float(p[2])
            except Exception:
                return False
            if math.isnan(x) or math.isnan(y) or math.isnan(z): return False
            if math.isinf(x) or math.isinf(y) or math.isinf(z): return False
            if z < -0.05 or z > 1.5: return False
            if x < -0.7 or x > 3.5: return False
            if y < -1.7 or y > 0.2: return False
            return True

        filtered = [p for p in points if _is_valid(p)]
        # Drop whole packet if too few points survive.
        if len(filtered) < 2:
            if (self._last_valid_draw_t > 0.0
                    and now - self._last_valid_draw_t < self._stale_keep_s):
                return False
            self.clear()
            return False

        starts = filtered[:-1]
        ends = filtered[1:]
        width = float(width_px) if width_px is not None else self._line_width_px

        r = float(color[0])
        g = float(color[1])
        b = float(color[2])
        a = float(color[3]) if len(color) >= 4 else 1.0

        segments = list(zip(starts, ends))
        colors = [(r, g, b, a)] * len(segments)
        widths = [float(width)] * len(segments)
        update_debug_draw_owner(
            owner_id or self._owner_id,
            segments,
            colors,
            widths,
            stale_keep_s=self._stale_keep_s,
        )
        self._last_valid_draw_t = now
        self._draw_calls += 1
        return True

    @property
    def draw_calls(self) -> int:
        return self._draw_calls


# ---------------------------------------------------------------------------
# USD mesh-segment fallback (no BasisCurves, no 'widths' primvar)
# ---------------------------------------------------------------------------


class _UsdMeshSegmentAdapter:
    """DISABLED USD mesh-segment backend.

    Kept as a fail-loud stub so any stale call site fails fast instead of
    silently using a fallback we're not ready to support. Re-enable here
    once debug_draw is stable.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "_UsdMeshSegmentAdapter is intentionally disabled while we "
            "stabilize debug_draw. Do not call this backend."
        )


# ---------------------------------------------------------------------------
# Top-level overlay
# ---------------------------------------------------------------------------


def _fmt_point(p) -> str:
    try:
        return f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
    except Exception:
        return "(?,?,?)"


def _fmt_bounds(points) -> str:
    if not points:
        return "x[] y[] z[]"
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (
        f"x[{min(xs):.3f},{max(xs):.3f}] "
        f"y[{min(ys):.3f},{max(ys):.3f}] "
        f"z[{min(zs):.3f},{max(zs):.3f}]"
    )


class IsaacTrajectoryOverlay:
    """Draw one externally-computed future trajectory in the Isaac viewport."""

    def __init__(self, cfg: TrajectoryOverlayConfig | None = None):
        self.cfg = cfg or TrajectoryOverlayConfig()
        self._draw = None
        self._draw_backend = None
        self._receiver = None
        self._hit_receiver = None
        self._last_packet_seq: int | None = None
        self._total_recv_packets = 0
        self._total_drawn_frames = 0
        self._last_log_t = 0.0
        self._last_no_recv_log_t = 0.0
        self._last_color = (0.2, 1.0, 0.3, 1.0)
        self._last_return_points = None
        self._last_return_t = 0.0

        if not self.cfg.enabled:
            return

        try:
            self._receiver = _TrajectoryUdpReceiver(self.cfg.udp_host, self.cfg.udp_port)
        except Exception as exc:
            print(f"[trajectory_overlay_isaac] UDP receiver init failed: {exc!r}")
            return
        try:
            self._hit_receiver = _HitPlanUdpReceiver(self.cfg.hit_udp_host, self.cfg.hit_udp_port)
        except Exception as exc:
            print(f"[trajectory_overlay_isaac] hit UDP receiver init failed: {exc!r}")
            self._hit_receiver = None

        # Backend selection: ONLY debug_draw.
        # BasisCurves was removed (Hydra rejects the widths primvar), and the
        # USD mesh-segment fallback is temporarily disabled while we stabilize
        # debug_draw. If debug_draw cannot be acquired, the overlay is simply
        # disabled -- no fallbacks, no surprises.
        if self.cfg.use_debug_draw:
            iface, iface_module = _acquire_debug_draw_iface()
            if iface is not None:
                try:
                    self._draw = _DebugDrawTrajectoryAdapter(
                        iface, iface_module,
                        line_width_px=6.0,
                        stale_keep_s=float(self.cfg.stale_keep_s),
                    )
                    self._draw_backend = "debug_draw"
                except Exception as exc:
                    print(f"[trajectory_overlay_isaac] debug_draw adapter "
                          f"init failed: {exc!r}")
                    self._draw = None

        if self._draw is not None:
            print(
                f"[trajectory_overlay_isaac] backend={self._draw_backend} "
                f"prim={self._draw.prim_path} udp={self.cfg.udp_host}:{self.cfg.udp_port}"
            )
        else:
            print(
                f"[trajectory_overlay_isaac] debug_draw unavailable; "
                f"trajectory overlay disabled "
                f"(udp={self.cfg.udp_host}:{self.cfg.udp_port})"
            )

    @property
    def available(self) -> bool:
        return self._draw is not None and self._receiver is not None

    def clear(self) -> None:
        if self._draw is not None:
            self._draw.clear()

    def close(self) -> None:
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None
        if self._hit_receiver is not None:
            self._hit_receiver.close()
            self._hit_receiver = None
        self._last_return_points = None
        self._last_return_t = 0.0
        if self._draw is not None:
            self._draw.clear()
            self._draw = None

    def push(self, t: float, p) -> None:
        del t, p
        if not self.available:
            return

        now = time.monotonic()
        polled = self._receiver.poll()
        if polled is None:
            # No UDP packet this frame.  Emit a throttled heartbeat (every 3s)
            # so the operator can see that the receiver is alive but starving.
            if now - self._last_no_recv_log_t >= 3.0:
                self._last_no_recv_log_t = now
                print(
                    f"[trajectory_overlay_isaac] no UDP packet received for "
                    f">=3s on {self.cfg.udp_host}:{self.cfg.udp_port}"
                )
            return

        points, after_p1_bounce, sequence, drained = polled
        hit_plan = self._hit_receiver.poll() if self._hit_receiver is not None else None
        self._total_recv_packets += 1
        self._last_packet_seq = sequence

        # Bright green for after-p1-bounce; bright red for pre-bounce.
        # Avoid black -- it gets culled in performance mode.
        color = (0.2, 1.0, 0.3, 1.0) if after_p1_bounce else (1.0, 0.2, 0.1, 1.0)
        self._last_color = color

        drew = False
        # Stale-curve handling is owned by the debug_draw adapter; we just
        # pass the points through. The adapter internally either refreshes
        # or holds for stale_keep_s based on points length.
        if len(points) >= 2:
            drew = self._draw.draw_polyline(
                points,
                color,
                width_px=self.cfg.line_width,
                owner_id="trajectory_overlay_main",
            )
            if drew:
                self._total_drawn_frames += 1
        else:
            self._draw.draw_polyline(
                points,
                color,
                width_px=self.cfg.line_width,
                owner_id="trajectory_overlay_main",
            )

        # Only refresh the black return trajectory once the main trajectory has
        # switched to the post-bounce green phase. This makes the return line
        # follow the same post-bounce "real/adjusted" solve instead of the
        # earlier pre-aim solve.
        if after_p1_bounce and hit_plan is not None:
            self._last_return_points = _sample_return_trajectory(hit_plan, self.cfg)
            self._last_return_t = now
        elif not after_p1_bounce:
            self._last_return_points = None
            self._last_return_t = 0.0

        if self._last_return_points is not None and (
            now - self._last_return_t <= float(self.cfg.stale_keep_s)
        ):
            self._draw.draw_polyline(
                self._last_return_points,
                (0.05, 0.05, 0.05, 1.0),
                width_px=self.cfg.return_line_width,
                owner_id="trajectory_overlay_return",
            )
        else:
            self._last_return_points = None
            drop_debug_draw_owner("trajectory_overlay_return")

        # Throttled diagnostics (default 1 Hz).  Per-iteration we only update
        # counters; printing happens once per log_period_s.
        if now - self._last_log_t >= float(self.cfg.log_period_s):
            self._last_log_t = now
            first = _fmt_point(points[0]) if points else "(none)"
            last = _fmt_point(points[-1]) if points else "(none)"
            bounds = _fmt_bounds(points)
            print(
                f"[trajectory_overlay_isaac] recv packets={self._total_recv_packets} "
                f"seq={sequence} n={len(points)} first={first} last={last} "
                f"bounds={bounds} backend={self._draw_backend}"
            )
            print(
                f"[trajectory_overlay_draw] draw calls={self._total_drawn_frames} "
                f"n={len(points)} prim={self._draw.prim_path}"
            )
