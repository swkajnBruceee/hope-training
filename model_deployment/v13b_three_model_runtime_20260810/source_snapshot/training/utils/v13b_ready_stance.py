"""Single source of truth for the V1.3B right-front staggered READY pose.

The deployable direct policy, training-only priors, and PhysX reset must use
the same lower-body nominal.  Keeping the inverse-kinematic construction in
one module prevents a reset/action-target seam.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class V13BReadyStance:
    """Explicit, auditable lower-body READY state in robot joint coordinates."""

    front_foot: str
    staggered_stance_half_span_m: float
    lateral_widen_per_foot_m: float
    knee_flexion_rad: float
    root_height_delta_m: float
    joint_positions: dict[str, float]


def build_v13b_ready_stance(
    *,
    front_foot: str = "right",
    staggered_stance_half_span_m: float = 0.06,
    lateral_widen_per_foot_m: float = 0.05,
    knee_flexion_rad: float = 0.48,
) -> V13BReadyStance:
    """Return the IK-consistent wide/deep staggered READY contract.

    ``front_foot`` follows the existing training convention.  The returned
    root-height delta preserves sole contact after the sagittal and lateral
    transforms; it must be applied to the scene reset root position.
    """

    front = str(front_foot).strip().lower()
    if front not in {"left", "right"}:
        raise ValueError("front_foot must be 'left' or 'right'")
    half_span = float(staggered_stance_half_span_m)
    lateral = float(lateral_widen_per_foot_m)
    knee = float(knee_flexion_rad)
    if not 0.0 <= half_span <= 0.10:
        raise ValueError("staggered_stance_half_span_m must be in [0, 0.10]")
    if not 0.0 <= lateral <= 0.08:
        raise ValueError("lateral_widen_per_foot_m must be in [0, 0.08]")
    if not 0.32 <= knee <= 0.60:
        raise ValueError("knee_flexion_rad must be in [0.32, 0.60]")

    thigh_length = 0.370
    shank_length = 0.415
    base_hip_pitch = -0.160
    base_knee = 0.320
    base_ankle_pitch = -0.155
    base_foot_pitch = base_hip_pitch + base_knee + base_ankle_pitch
    nominal_hip_pitch = -0.5 * knee

    a = thigh_length + shank_length * math.cos(knee)
    b = shank_length * math.sin(knee)
    radius = math.hypot(a, b)
    phase = math.atan2(b, a)

    def foot_x(hip_pitch: float) -> float:
        return (
            -thigh_length * math.sin(hip_pitch)
            - shank_length * math.sin(hip_pitch + knee)
        )

    def foot_z(hip_pitch: float) -> float:
        return (
            -thigh_length * math.cos(hip_pitch)
            - shank_length * math.cos(hip_pitch + knee)
        )

    def hip_for_x(target_x: float) -> float:
        ratio = max(-1.0, min(1.0, -target_x / radius))
        primary = math.asin(ratio) - phase
        alternate = math.pi - math.asin(ratio) - phase
        candidates = (
            primary,
            alternate,
            primary + 2.0 * math.pi,
            primary - 2.0 * math.pi,
            alternate + 2.0 * math.pi,
            alternate - 2.0 * math.pi,
        )
        return min(candidates, key=lambda value: abs(value - nominal_hip_pitch))

    front_sign = 1.0 if front == "left" else -1.0
    nominal_x = foot_x(nominal_hip_pitch)
    left_hip_pitch = hip_for_x(nominal_x + front_sign * half_span)
    right_hip_pitch = hip_for_x(nominal_x - front_sign * half_span)
    left_ankle_pitch = base_foot_pitch - left_hip_pitch - knee
    right_ankle_pitch = base_foot_pitch - right_hip_pitch - knee

    base_hip_roll_abs = 0.080
    base_left_ankle_roll = -0.0078
    base_right_ankle_roll = 0.0078
    mean_staggered_z = 0.5 * (foot_z(left_hip_pitch) + foot_z(right_hip_pitch))
    leg_height = max(0.40, -mean_staggered_z)
    widened_sin = math.sin(base_hip_roll_abs) + lateral / leg_height
    if widened_sin >= 0.95:
        raise ValueError("requested lateral READY stance is unreachable")
    hip_roll_abs = math.asin(widened_sin)
    hip_roll_delta = hip_roll_abs - base_hip_roll_abs
    left_ankle_roll = base_left_ankle_roll - hip_roll_delta
    right_ankle_roll = base_right_ankle_roll + hip_roll_delta

    baseline_z = (
        -thigh_length * math.cos(base_hip_pitch)
        - shank_length * math.cos(base_hip_pitch + base_knee)
    )
    baseline_vertical = baseline_z * math.cos(base_hip_roll_abs)
    staggered_vertical = mean_staggered_z * math.cos(hip_roll_abs)
    root_height_delta = baseline_vertical - staggered_vertical

    return V13BReadyStance(
        front_foot=front,
        staggered_stance_half_span_m=half_span,
        lateral_widen_per_foot_m=lateral,
        knee_flexion_rad=knee,
        root_height_delta_m=root_height_delta,
        joint_positions={
            "left_hip_pitch_joint": left_hip_pitch,
            "right_hip_pitch_joint": right_hip_pitch,
            "left_knee_joint": knee,
            "right_knee_joint": knee,
            "left_ankle_pitch_joint": left_ankle_pitch,
            "right_ankle_pitch_joint": right_ankle_pitch,
            "left_hip_roll_joint": hip_roll_abs,
            "right_hip_roll_joint": -hip_roll_abs,
            "left_ankle_roll_joint": left_ankle_roll,
            "right_ankle_roll_joint": right_ankle_roll,
        },
    )


V13B_RIGHT_FRONT_READY = build_v13b_ready_stance()

