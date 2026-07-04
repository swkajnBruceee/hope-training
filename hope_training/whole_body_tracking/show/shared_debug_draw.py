"""Shared debug_draw compositor for multiple Isaac overlays.

Isaac's debug_draw interface exposes a single global line buffer per process.
If multiple overlays each call ``clear()`` + ``draw_lines()`` independently,
they erase one another and appear to flicker. This module lets overlays submit
their current frame geometry under a stable owner id, then flushes the merged
line set once per simulation step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass
class _OwnerFrame:
    segments: list
    colors: list
    widths: list
    stale_keep_s: float
    last_valid_t: float


class _SharedDebugDrawState:
    def __init__(self):
        self._iface = None
        self._iface_module = None
        self._owners: dict[str, _OwnerFrame] = {}
        self._draw_fail_count = 0

    def bind(self, iface, iface_module: str | None = None) -> None:
        if self._iface is None:
            self._iface = iface
            self._iface_module = iface_module
            return
        if iface is not self._iface:
            raise RuntimeError("shared debug_draw received multiple interface instances")

    def update(self, owner_id: str, segments, colors, widths, stale_keep_s: float) -> None:
        now = time.monotonic()
        if segments:
            self._owners[owner_id] = _OwnerFrame(
                segments=list(segments),
                colors=list(colors),
                widths=list(widths),
                stale_keep_s=float(stale_keep_s),
                last_valid_t=now,
            )
            return

        existing = self._owners.get(owner_id)
        if existing is None:
            return
        if now - existing.last_valid_t >= existing.stale_keep_s:
            self._owners.pop(owner_id, None)

    def drop(self, owner_id: str) -> None:
        self._owners.pop(owner_id, None)

    def clear_all(self) -> None:
        self._owners.clear()
        self._clear_iface()

    def flush(self) -> bool:
        if self._iface is None:
            return False

        now = time.monotonic()
        active = []
        stale_owners = []
        for owner_id, frame in self._owners.items():
            if now - frame.last_valid_t < frame.stale_keep_s:
                active.append(frame)
            else:
                stale_owners.append(owner_id)
        for owner_id in stale_owners:
            self._owners.pop(owner_id, None)

        self._clear_iface()
        if not active:
            return False

        segments = []
        colors = []
        widths = []
        for frame in active:
            segments.extend(frame.segments)
            colors.extend(frame.colors)
            widths.extend(frame.widths)

        try:
            import carb
            carb_starts = [
                carb.Float3(float(s[0][0]), float(s[0][1]), float(s[0][2]))
                for s in segments
            ]
            carb_ends = [
                carb.Float3(float(s[1][0]), float(s[1][1]), float(s[1][2]))
                for s in segments
            ]
            carb_colors = [
                carb.ColorRgba(float(c[0]), float(c[1]), float(c[2]), float(c[3]))
                for c in colors
            ]
            carb_widths = [float(w) for w in widths]
        except Exception as exc:
            self._draw_fail_count += 1
            if self._draw_fail_count <= 3:
                print(f"[shared_debug_draw] carb build failed: {type(exc).__name__}: {str(exc)[:300]}")
            return False

        try:
            self._iface.draw_lines(carb_starts, carb_ends, carb_colors, carb_widths)
        except Exception as exc:
            self._draw_fail_count += 1
            if self._draw_fail_count <= 3:
                print(f"[shared_debug_draw] draw_lines failed: {type(exc).__name__}: {str(exc)[:300]}")
            return False

        self._draw_fail_count = 0
        return True

    def _clear_iface(self) -> None:
        if self._iface is None:
            return
        for name in ("clear_lines", "clear_points", "clear"):
            try:
                getattr(self._iface, name)()
                return
            except Exception:
                continue


_STATE = _SharedDebugDrawState()


def register_debug_draw(iface, iface_module: str | None = None) -> None:
    _STATE.bind(iface, iface_module)


def update_debug_draw_owner(owner_id: str, segments, colors, widths, stale_keep_s: float) -> None:
    _STATE.update(owner_id, segments, colors, widths, stale_keep_s)


def drop_debug_draw_owner(owner_id: str) -> None:
    _STATE.drop(owner_id)


def clear_shared_debug_draw() -> None:
    _STATE.clear_all()


def flush_shared_debug_draw() -> bool:
    return _STATE.flush()
