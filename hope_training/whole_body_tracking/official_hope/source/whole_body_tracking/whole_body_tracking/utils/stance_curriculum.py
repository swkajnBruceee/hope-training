"""Pure stance-curriculum math shared by training code and consistency tests."""

from __future__ import annotations


def smoothstep_stance_alpha(
    iteration: int,
    *,
    ramp_start_iteration: int = 300,
    ramp_end_iteration: int = 2100,
) -> float:
    """Return the fixed 3000-iteration stance schedule's alpha."""
    start = int(ramp_start_iteration)
    end = int(ramp_end_iteration)
    if start < 0 or end <= start:
        raise ValueError(
            "stance curriculum requires 0 <= ramp_start_iteration < "
            "ramp_end_iteration"
        )
    step = int(iteration)
    if step <= start:
        return 0.0
    if step >= end:
        return 1.0
    x = (step - start) / float(end - start)
    return 3.0 * x * x - 2.0 * x * x * x


def lerp(old: float, new: float, alpha: float) -> float:
    """Interpolate one stance target using the shared alpha."""
    a = min(1.0, max(0.0, float(alpha)))
    return (1.0 - a) * float(old) + a * float(new)
