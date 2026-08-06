"""Draw landing decision candidates in the Isaac viewport."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import time

from shared_debug_draw import (  # type: ignore[import-not-found]
    drop_debug_draw_owner,
    register_debug_draw,
    update_debug_draw_owner,
)


@dataclass
class LandingCandidateOverlayConfig:
    udp_host: str = "127.0.0.1"
    udp_port: int = 19534
    marker_size_m: float = 0.035
    selected_marker_size_m: float = 0.070
    line_width_px: float = 5.0
    stale_keep_s: float = 2.0


class _LandingCandidateReceiver:
    _HEADER = struct.Struct("<4sII")
    _ITEM = struct.Struct("<dddddI")
    _MAGIC = b"LCAN"

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, int(port)))
        self._sock.setblocking(False)
        self._last_sequence = None

    def close(self) -> None:
        self._sock.close()

    def poll(self):
        latest = None
        while True:
            try:
                payload, _ = self._sock.recvfrom(65535)
            except BlockingIOError:
                break
            parsed = self._parse(payload)
            if parsed is not None:
                latest = parsed
        return latest

    def _parse(self, payload: bytes):
        if len(payload) < self._HEADER.size:
            return None
        magic, sequence, count = self._HEADER.unpack_from(payload, 0)
        if magic != self._MAGIC:
            return None
        if self._last_sequence is not None and int(sequence) == self._last_sequence:
            return None
        self._last_sequence = int(sequence)
        offset = self._HEADER.size
        candidates = []
        for _ in range(int(count)):
            if offset + self._ITEM.size > len(payload):
                break
            x, y, z, dt, score, state = self._ITEM.unpack_from(payload, offset)
            offset += self._ITEM.size
            candidates.append({
                "point": (float(x), float(y), float(z)),
                "dt": float(dt),
                "score": float(score),
                "state": int(state),
            })
        return {"sequence": int(sequence), "candidates": candidates}


def _acquire_debug_draw_iface(log_tag: str = "[landing_candidates_overlay]"):
    last_err = None
    for mod_name in ("isaacsim.util.debug_draw", "omni.isaac.debug_draw", "omni.debugdraw"):
        try:
            mod = __import__(mod_name, fromlist=["_debug_draw"])
            iface_mod = getattr(mod, "_debug_draw")
            iface = iface_mod.acquire_debug_draw_interface()
            print(f"{log_tag} debug_draw backend: {mod_name} OK")
            return iface, mod_name
        except Exception as exc:  # pragma: no cover - Isaac runtime dependent
            last_err = exc
    print(f"{log_tag} debug_draw unavailable; last_err={last_err!r}")
    return None, None


class _DebugDrawLandingCandidateAdapter:
    def __init__(self, iface, iface_module: str, stale_keep_s: float):
        self._iface = iface
        self._iface_module = iface_module
        self._owner_id = "landing_candidates_overlay"
        self._last_valid_draw_t = 0.0
        register_debug_draw(self._iface, self._iface_module)

    def clear(self) -> None:
        drop_debug_draw_owner(self._owner_id)

    @staticmethod
    def _add(p, d, scale):
        return (p[0] + d[0] * scale, p[1] + d[1] * scale, p[2] + d[2] * scale)

    def draw(self, candidates, cfg: LandingCandidateOverlayConfig) -> bool:
        now = time.monotonic()
        if not candidates:
            if self._last_valid_draw_t > 0.0 and now - self._last_valid_draw_t < cfg.stale_keep_s:
                return False
            self.clear()
            return False

        segments = []
        colors = []
        widths = []
        for c in candidates:
            if int(c["state"]) <= 0:
                continue
            p0 = c["point"]
            p = (p0[0], p0[1], p0[2] + 0.025)
            state = int(c["state"])
            size = cfg.selected_marker_size_m if state == 2 else cfg.marker_size_m
            if state == 2:
                color = (0.10, 0.45, 1.0, 1.0)
                width = cfg.line_width_px * 1.6
            else:
                color = (1.0, 0.76, 0.15, 1.0)
                width = cfg.line_width_px

            axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
            for axis in axes:
                segments.append((self._add(p, axis, -size), self._add(p, axis, size)))
                colors.append(color)
                widths.append(width)
            if state == 2:
                segments.append((self._add(p, (0.0, 0.0, 1.0), -size), self._add(p, (0.0, 0.0, 1.0), size)))
                colors.append(color)
                widths.append(width)

        if not segments:
            if self._last_valid_draw_t > 0.0 and now - self._last_valid_draw_t < cfg.stale_keep_s:
                return False
            self.clear()
            return False

        update_debug_draw_owner(self._owner_id, segments, colors, widths, cfg.stale_keep_s)
        self._last_valid_draw_t = now
        return True


class IsaacLandingCandidateOverlay:
    def __init__(self, cfg: LandingCandidateOverlayConfig | None = None):
        self.cfg = cfg or LandingCandidateOverlayConfig()
        self._receiver = None
        self._draw = None
        self._total_recv_packets = 0
        self._total_drawn_frames = 0
        self._last_log_t = 0.0
        try:
            self._receiver = _LandingCandidateReceiver(self.cfg.udp_host, self.cfg.udp_port)
        except OSError as exc:
            print(f"[landing_candidates_overlay] UDP receiver init failed: {exc!r}")
            return
        iface, iface_module = _acquire_debug_draw_iface()
        if iface is not None:
            self._draw = _DebugDrawLandingCandidateAdapter(iface, iface_module, self.cfg.stale_keep_s)
        print(f"[landing_candidates_overlay] active={self.available} udp={self.cfg.udp_host}:{self.cfg.udp_port}")

    @property
    def available(self) -> bool:
        return self._receiver is not None and self._draw is not None

    def close(self) -> None:
        if self._draw is not None:
            self._draw.clear()
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None

    def push(self, t: float) -> None:
        del t
        if not self.available:
            return
        packet = self._receiver.poll()
        if packet is None:
            return
        self._total_recv_packets += 1
        candidates = packet["candidates"]
        if self._draw.draw(candidates, self.cfg):
            self._total_drawn_frames += 1
        now = time.monotonic()
        if now - self._last_log_t > 2.0:
            self._last_log_t = now
            selected = sum(1 for c in candidates if int(c["state"]) == 2)
            hard_valid = sum(1 for c in candidates if int(c["state"]) >= 1)
            print(
                f"[landing_candidates_overlay] recv={self._total_recv_packets} "
                f"draw={self._total_drawn_frames} candidates={len(candidates)} "
                f"hard_valid={hard_valid} selected={selected}"
            )
