"""Unit conversion helpers for mocap data."""

from __future__ import annotations


def position_scale_to_meters(unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in ("millimeter", "millimeters", "mm"):
        return 0.001
    if normalized in ("centimeter", "centimeters", "cm"):
        return 0.01
    if normalized in ("meter", "meters", "m"):
        return 1.0
    raise ValueError(f"unsupported position unit: {unit}")

