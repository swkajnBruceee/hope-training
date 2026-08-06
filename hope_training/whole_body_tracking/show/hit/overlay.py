"""Isaac viewport display for the HOPE hit state.

The C++ ROS node ``hit_state_udp_bridge`` (``hope_ws/src/bringup``) subscribes
to ``/hit/state`` (``msgs/msg/HitState``) and forwards the plan over UDP. This
module receives the packets and renders the planned hit point, target landing,
racket velocity / normal, and the in/out ball velocity arrows directly in the
Isaac Sim viewport using ``debug_draw``.

Mirrors :class:`show.trajectory.IsaacTrajectoryOverlay` 1:1 in style and
backend choices (both live under ``show/``; trajectory is ``show/trajectory``,
hit is ``show/hit``):

* Only the ``debug_draw`` backend is supported. It survives
  ``--rendering_mode performance`` and never creates USD prims (so no Hydra
  primvar warnings).
* If ``debug_draw`` cannot be acquired the overlay is silently turned off --
  the trajectory overlay follows the same policy.
* Stale-frame handling lives in the backend adapter; the top-level overlay
  only forwards UDP packets.

Wire format
-----------
::

    magic:    4s  = b"HITS"
    sequence: u32 (LE)
    valid:    u32 (LE)   (1 if hit_state.valid)
    has_plan: u32 (LE)   (1 if the planned hit point is in-scene and finite)
    if has_plan == 1:
      hit_position    (3 doubles, LE)
      target_land     (3 doubles, LE)
      ball_v_in       (3 doubles, LE)
      ball_v_out      (3 doubles, LE)
      racket_vel      (3 doubles, LE)
      racket_normal   (3 doubles, LE)

Coordinates are in ``hit_state.header.frame_id`` (HOPE ``world``).
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


# ---------------------------------------------------------------------------
# Config + UDP receiver
# ---------------------------------------------------------------------------


@dataclass
class HitOverlayConfig:
    enabled: bool = True
    udp_host: str = "127.0.0.1"
    udp_port: int = 19533

    # Drawing tuning knobs.
    stale_keep_s: float = 0.3
    log_period_s: float = 1.0

    # Arrow scaling (meters per m/s) and linewidth.
    velocity_arrow_scale: float = 0.08
    racket_velocity_arrow_scale: float = 0.06
    line_width_px: float = 6.0

    # On-screen marker size for hit / target points (meters).
    hit_marker_size_m: float = 0.05
    target_marker_size_m: float = 0.05

    # Use debug_draw; no USD fallback (matches show/trajectory).
    use_debug_draw: bool = True


class _HitUdpReceiver:
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
        """Drain pending UDP packets and return the latest valid one.

        Returns ``(payload, sequence, total_drained)`` or ``None`` if no
        valid packet was received in this drain cycle.
        """
        latest = None
        total = 0
        while True:
            try:
                payload, _ = self._socket.recvfrom(65535)
            except BlockingIOError:
                break

            total += 1
            parsed = self._parse(payload)
            if parsed is not None:
                latest = parsed
        if latest is None:
            return None
        body, sequence = latest
        return body, sequence, total

    def _parse(self, payload: bytes):
        if len(payload) < self._HEADER.size:
            return None
        magic, sequence, valid, has_plan = self._HEADER.unpack_from(payload, 0)
        if magic != self._MAGIC:
            return None
        if self._last_sequence is not None and int(sequence) == self._last_sequence:
            return None
        self._last_sequence = int(sequence)
        if not int(has_plan):
            return {"valid": bool(valid), "has_plan": False}, int(sequence)
        expected_size = self._HEADER.size + 6 * self._VEC3.size
        if len(payload) < expected_size:
            return None
        offset = self._HEADER.size
        vecs = []
        for _ in range(6):
            vecs.append(self._VEC3.unpack_from(payload, offset))
            offset += self._VEC3.size
        body = {
            "valid": bool(valid),
            "has_plan": True,
            "hit_position": vecs[0],
            "target_land": vecs[1],
            "ball_v_in": vecs[2],
            "ball_v_out": vecs[3],
            "racket_vel": vecs[4],
            "racket_normal": vecs[5],
        }
        return body, int(sequence)


# ---------------------------------------------------------------------------
# debug_draw backend (mirrors _DebugDrawTrajectoryAdapter in show/trajectory)
# ---------------------------------------------------------------------------


def _acquire_debug_draw_iface(log_tag: str = "[hit_overlay_isaac]"):
    """Try every known debug_draw entry point in order. Print each attempt."""
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


class _DebugDrawHitAdapter:
    """Renders the hit state via Isaac's debug_draw overlay."""

    def __init__(self, iface, iface_module: str,
                 line_width_px: float = 6.0, stale_keep_s: float = 0.3):
        self._iface = iface
        self._iface_module = iface_module
        self._line_width_px = float(line_width_px)
        self._stale_keep_s = float(stale_keep_s)
        self._last_valid_draw_t = 0.0
        self._draw_calls = 0
        self._draw_fail_count = 0
        self.prim_path = "(debug_draw)"
        self._owner_id = "hit_overlay"
        register_debug_draw(self._iface, self._iface_module)

    def clear(self) -> None:
        drop_debug_draw_owner(self._owner_id)

    def _draw_polyline_segments(self, segments, colors, widths):
        """Draw a list of (start, end) segments in one batch.

        ``segments`` is a list of ((x, y, z), (x, y, z)) tuples.
        ``colors``  is a list of (r, g, b, a) per segment.
        ``widths``  is a list of floats per segment.
        """
        if not segments:
            return True
        update_debug_draw_owner(
            self._owner_id,
            segments,
            colors,
            widths,
            stale_keep_s=self._stale_keep_s,
        )
        self._draw_calls += 1
        return True

    def _is_finite_point(self, p) -> bool:
        try:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
        except Exception:
            return False
        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            return False
        if math.isinf(x) or math.isinf(y) or math.isinf(z):
            return False
        if z < -0.05 or z > 1.5:
            return False
        if x < -0.7 or x > 3.5:
            return False
        if y < -1.7 or y > 0.2:
            return False
        return True

    @staticmethod
    def _add(start, vec, t):
        return (start[0] + t * vec[0], start[1] + t * vec[1], start[2] + t * vec[2])

    def draw(self, plan: dict, cfg: HitOverlayConfig) -> bool:
        """Render one plan. Returns True iff anything was actually drawn."""
        import time as _time
        now = _time.monotonic()

        if not plan.get("has_plan", False):
            if (self._last_valid_draw_t > 0.0
                    and now - self._last_valid_draw_t < self._stale_keep_s):
                return False
            self.clear()
            return False

        hit_pos = plan["hit_position"]
        target = plan["target_land"]
        ball_v_in = plan["ball_v_in"]
        ball_v_out = plan["ball_v_out"]
        racket_vel = plan["racket_vel"]
        racket_normal = plan["racket_normal"]

        if not (self._is_finite_point(hit_pos) and self._is_finite_point(target)):
            if (self._last_valid_draw_t > 0.0
                    and now - self._last_valid_draw_t < self._stale_keep_s):
                return False
            self.clear()
            return False

        segments = []
        colors = []
        widths = []

        line_width = float(cfg.line_width_px)

        # ---- Arrows emanating from the hit point. ----
        # ball velocity (incoming): drawn as a thin gray segment toward hit_pos.
        segments.append((tuple(hit_pos), self._add(hit_pos, ball_v_in,
                                                   cfg.velocity_arrow_scale)))
        colors.append((0.55, 0.58, 0.62, 1.0))   # neutral gray
        widths.append(max(1.0, line_width * 0.6))

        # racket normal: short, dark -- indicates racket facing.
        segments.append((tuple(hit_pos), self._add(hit_pos, racket_normal,
                                                   cfg.racket_velocity_arrow_scale * 1.5)))
        colors.append((0.15, 0.27, 0.33, 1.0))   # dark slate
        widths.append(max(1.0, line_width * 0.8))

        # racket velocity: bright teal.
        segments.append((tuple(hit_pos), self._add(hit_pos, racket_vel,
                                                   cfg.racket_velocity_arrow_scale)))
        colors.append((0.16, 0.61, 0.55, 1.0))   # teal
        widths.append(line_width)

        # ball velocity (outgoing): orange-red, projected from hit to target.
        if self._is_finite_point(ball_v_out):
            segments.append((tuple(hit_pos), self._add(hit_pos, ball_v_out,
                                                       cfg.velocity_arrow_scale)))
            colors.append((0.89, 0.34, 0.18, 1.0))  # orange-red
            widths.append(line_width)

        # Hit-to-target line (dashed feel via two segments for cheapness).
        segments.append((tuple(hit_pos), tuple(target)))
        colors.append((0.18, 0.52, 0.67, 1.0))    # blue
        widths.append(line_width * 0.8)

        # Hit cross (3 short segments along +/- x/y/z).
        h = float(cfg.hit_marker_size_m)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            segments.append(
                (self._add(hit_pos, axis, -h), self._add(hit_pos, axis, h))
            )
            colors.append((0.95, 0.32, 0.18, 1.0))  # bright red-orange
            widths.append(line_width)

        # Target cross.
        t = float(cfg.target_marker_size_m)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            segments.append(
                (self._add(target, axis, -t), self._add(target, axis, t))
            )
            colors.append((0.18, 0.52, 0.67, 1.0))  # blue
            widths.append(max(1.0, line_width * 0.7))

        ok = self._draw_polyline_segments(segments, colors, widths)
        if ok:
            self._last_valid_draw_t = now
        return ok

    @property
    def draw_calls(self) -> int:
        return self._draw_calls


# ---------------------------------------------------------------------------
# Top-level overlay
# ---------------------------------------------------------------------------


def _fmt_point(p) -> str:
    try:
        return f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
    except Exception:
        return "(?,?,?)"


class IsaacHitOverlay:
    """Draw one externally-received hit plan in the Isaac viewport."""

    def __init__(self, cfg: HitOverlayConfig | None = None):
        self.cfg = cfg or HitOverlayConfig()
        self._draw = None
        self._draw_backend = None
        self._receiver = None
        self._last_packet_seq: int | None = None
        self._total_recv_packets = 0
        self._total_drawn_frames = 0
        self._last_log_t = 0.0
        self._last_no_recv_log_t = 0.0

        if not self.cfg.enabled:
            return

        try:
            self._receiver = _HitUdpReceiver(self.cfg.udp_host, self.cfg.udp_port)
        except Exception as exc:
            print(f"[hit_overlay_isaac] UDP receiver init failed: {exc!r}")
            return

        if self.cfg.use_debug_draw:
            iface, iface_module = _acquire_debug_draw_iface("[hit_overlay_isaac]")
            if iface is not None:
                try:
                    self._draw = _DebugDrawHitAdapter(
                        iface, iface_module,
                        line_width_px=float(self.cfg.line_width_px),
                        stale_keep_s=float(self.cfg.stale_keep_s),
                    )
                    self._draw_backend = "debug_draw"
                except Exception as exc:
                    print(f"[hit_overlay_isaac] debug_draw adapter "
                          f"init failed: {exc!r}")
                    self._draw = None

        if self._draw is not None:
            print(
                f"[hit_overlay_isaac] backend={self._draw_backend} "
                f"prim={self._draw.prim_path} udp={self.cfg.udp_host}:{self.cfg.udp_port}"
            )
        else:
            print(
                f"[hit_overlay_isaac] debug_draw unavailable; "
                f"hit overlay disabled "
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
        if self._draw is not None:
            self._draw.clear()
            self._draw = None

    def push(self, _t: float | None = None) -> None:
        if not self.available:
            return

        now = time.monotonic()
        polled = self._receiver.poll()
        if polled is None:
            if now - self._last_no_recv_log_t >= 3.0:
                self._last_no_recv_log_t = now
                print(
                    f"[hit_overlay_isaac] no UDP packet received for >=3s on "
                    f"{self.cfg.udp_host}:{self.cfg.udp_port}"
                )
            return

        plan, sequence, _drained = polled
        self._total_recv_packets += 1
        self._last_packet_seq = sequence

        drew = self._draw.draw(plan, self.cfg)
        if drew:
            self._total_drawn_frames += 1

        if now - self._last_log_t >= float(self.cfg.log_period_s):
            self._last_log_t = now
            if plan.get("has_plan", False):
                hp = plan["hit_position"]
                tp = plan["target_land"]
                print(
                    f"[hit_overlay_isaac] recv={self._total_recv_packets} "
                    f"seq={sequence} valid={plan.get('valid')} "
                    f"hit={_fmt_point(hp)} target={_fmt_point(tp)} "
                    f"backend={self._draw_backend}"
                )
            else:
                print(
                    f"[hit_overlay_isaac] recv={self._total_recv_packets} "
                    f"seq={sequence} valid={plan.get('valid')} "
                    f"has_plan=False (waiting for /hit/state) "
                    f"backend={self._draw_backend}"
                )
