"""Pure-Python V1.3B schedule and contract helpers."""

from __future__ import annotations

import math


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def teacher_alpha(progress: float) -> float:
    """Training-only teacher schedule; final 35% is exactly zero."""
    p = max(0.0, min(1.0, float(progress)))
    if p <= 0.05:
        return 0.95
    if p >= 0.65:
        return 0.0
    u = (p - 0.05) / 0.60
    return 0.95 * 0.5 * (1.0 + math.cos(math.pi * u))


def _piecewise_smooth(progress: float, knots: tuple[tuple[float, float], ...]) -> float:
    """Smoothly interpolate a monotone V1.3B prior schedule."""
    p = max(0.0, min(1.0, float(progress)))
    if not knots:
        return 0.0
    if p <= knots[0][0]:
        return float(knots[0][1])
    for (left_p, left_value), (right_p, right_value) in zip(knots[:-1], knots[1:]):
        if p <= right_p:
            u = smoothstep((p - left_p) / max(right_p - left_p, 1.0e-8))
            return float(left_value + u * (right_value - left_value))
    return float(knots[-1][1])


def lower_prior_alpha(progress: float) -> float:
    """3396 additive-prior schedule; exactly zero for the final 30%."""
    return _piecewise_smooth(
        progress,
        ((0.00, 1.00), (0.10, 1.00), (0.25, 0.85), (0.45, 0.55), (0.60, 0.30), (0.70, 0.00), (1.00, 0.00)),
    )


def upper_prior_alpha(progress: float) -> float:
    """Complete model_900 strike-prior schedule; zero from 60% onward."""
    return _piecewise_smooth(
        progress,
        ((0.00, 0.90), (0.10, 0.90), (0.25, 0.65), (0.45, 0.30), (0.60, 0.00), (1.00, 0.00)),
    )


def beta_global(progress: float) -> float:
    """Reference-centered -> global target schedule; complete by 70%."""
    p = max(0.0, min(1.0, float(progress)))
    if p <= 0.10:
        return 0.0
    if p >= 0.70:
        return 1.0
    return smoothstep((p - 0.10) / 0.60)


def reference_reward_multiplier(progress: float) -> float:
    """Reference reward is coupled to teacher authority and cannot persist."""
    return teacher_alpha(progress) ** 2


def blend_joint_targets(q_teacher, q_student, progress: float):
    """Tensor-compatible teacher/student blend used by optional training hooks."""
    alpha = teacher_alpha(progress)
    return alpha * q_teacher + (1.0 - alpha) * q_student
