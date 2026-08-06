"""Read-only P2 shadow pipeline for a Planner strike goal.

This module resolves time, frame and contact-point semantics and emits audit
records.  It intentionally has no policy/Isaac dependency and exposes no
method that can mutate an observation, command, action, reward or termination.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from .strike_goal import (
    AxialRacketContactCalibration,
    LatchedStrikeGoal,
    RacketContactTarget,
    StrikeGoal10D,
    StrikeGoalFrameTransform,
    StrikeGoalNormalizer,
    StrikeGoalValidationError,
    StrikeGoalValidator,
)


def _vector3(value: Iterable[float], name: str) -> tuple[float, float, float]:
    result = tuple(float(component) for component in value)
    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        raise StrikeGoalValidationError(f"{name} must contain exactly three finite values")
    return result  # type: ignore[return-value]


def _norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(float(component) ** 2 for component in vector))


def _subtract(lhs: Iterable[float], rhs: Iterable[float]) -> tuple[float, float, float]:
    left = tuple(lhs)
    right = tuple(rhs)
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def _cross(lhs: Iterable[float], rhs: Iterable[float]) -> tuple[float, float, float]:
    a = tuple(lhs)
    b = tuple(rhs)
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@dataclass(frozen=True)
class RacketFaceState:
    """Actual effective racket-face state in one explicit coordinate frame."""

    link_origin_position: tuple[float, float, float]
    face_contact_position: tuple[float, float, float]
    face_normal: tuple[float, float, float]
    face_linear_velocity: tuple[float, float, float]
    link_angular_velocity: tuple[float, float, float]
    frame_id: str

    def __post_init__(self) -> None:
        for name in (
            "link_origin_position",
            "face_contact_position",
            "face_normal",
            "face_linear_velocity",
            "link_angular_velocity",
        ):
            object.__setattr__(self, name, _vector3(getattr(self, name), name))
        if abs(_norm(self.face_normal) - 1.0) > 1.0e-3:
            raise StrikeGoalValidationError("actual face_normal must be unit length")
        if not self.frame_id:
            raise StrikeGoalValidationError("actual racket frame_id must be non-empty")

    @classmethod
    def from_link_state(
        cls,
        *,
        link_origin_position: Iterable[float],
        link_origin_linear_velocity: Iterable[float],
        link_angular_velocity: Iterable[float],
        face_normal: Iterable[float],
        frame_id: str,
        calibration: AxialRacketContactCalibration,
    ) -> "RacketFaceState":
        link_position = _vector3(link_origin_position, "link_origin_position")
        link_velocity = _vector3(
            link_origin_linear_velocity, "link_origin_linear_velocity"
        )
        angular_velocity = _vector3(link_angular_velocity, "link_angular_velocity")
        normal = _vector3(face_normal, "face_normal")
        if abs(_norm(normal) - 1.0) > 1.0e-3:
            raise StrikeGoalValidationError("face_normal must be unit length")
        offset = tuple(
            calibration.link_origin_to_effective_face_along_normal_m * component
            for component in normal
        )
        angular_component = _cross(angular_velocity, offset)
        return cls(
            link_origin_position=link_position,
            face_contact_position=tuple(
                link_position[index] + offset[index] for index in range(3)
            ),
            face_normal=normal,
            face_linear_velocity=tuple(
                link_velocity[index] + angular_component[index] for index in range(3)
            ),
            link_angular_velocity=angular_velocity,
            frame_id=frame_id,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "link_origin_position": list(self.link_origin_position),
            "face_contact_position": list(self.face_contact_position),
            "face_normal": list(self.face_normal),
            "face_linear_velocity": list(self.face_linear_velocity),
            "link_angular_velocity": list(self.link_angular_velocity),
            "frame_id": self.frame_id,
        }


@dataclass(frozen=True)
class StrikeGoalShadowSample:
    control_step: int
    control_time_s: float
    source_goal: StrikeGoal10D
    policy_goal: StrikeGoal10D
    target: RacketContactTarget
    actual: RacketFaceState | None
    normalized_policy_goal: tuple[float, ...] | None
    errors: dict[str, float] | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "control_step": self.control_step,
            "control_time_s": self.control_time_s,
            "source_goal": self.source_goal.to_mapping(),
            "policy_goal": self.policy_goal.to_mapping(),
            "target": self.target.to_mapping(),
            "actual": None if self.actual is None else self.actual.to_mapping(),
            "normalized_policy_goal": (
                None
                if self.normalized_policy_goal is None
                else list(self.normalized_policy_goal)
            ),
            "errors": self.errors,
            "action_effect": False,
        }


class StrikeGoalShadowPipeline:
    """Monotonic read-only trace generator for one latched strike command."""

    def __init__(
        self,
        *,
        latched_goal: LatchedStrikeGoal,
        source_to_policy_transform: StrikeGoalFrameTransform,
        contact_calibration: AxialRacketContactCalibration,
        normalizer: StrikeGoalNormalizer | None = None,
    ) -> None:
        if source_to_policy_transform.source_frame != latched_goal.goal_at_receipt.frame_id:
            raise StrikeGoalValidationError(
                "shadow transform source does not match latched goal frame"
            )
        self.latched_goal = latched_goal
        self.source_to_policy_transform = source_to_policy_transform
        self.contact_calibration = contact_calibration
        self.normalizer = normalizer
        self.samples: list[StrikeGoalShadowSample] = []
        self._last_control_step = -1
        self._last_control_time_s = float("-inf")

    def capture(
        self,
        *,
        control_step: int,
        current_control_time_s: float,
        actual: RacketFaceState | None = None,
    ) -> StrikeGoalShadowSample:
        if not isinstance(control_step, int) or control_step < 0:
            raise StrikeGoalValidationError("control_step must be a non-negative integer")
        control_time = float(current_control_time_s)
        if not math.isfinite(control_time):
            raise StrikeGoalValidationError("current_control_time_s must be finite")
        if control_step <= self._last_control_step:
            raise StrikeGoalValidationError("shadow control_step must increase monotonically")
        if control_time < self._last_control_time_s:
            raise StrikeGoalValidationError("shadow control time moved backwards")

        source_goal = self.latched_goal.goal_at(control_time)
        policy_goal = self.source_to_policy_transform.apply(source_goal)
        StrikeGoalValidator().validate(policy_goal)
        target = self.contact_calibration.resolve(policy_goal)
        normalized = (
            None if self.normalizer is None else self.normalizer.normalize(policy_goal)
        )
        errors = None
        if actual is not None:
            if actual.frame_id != policy_goal.frame_id:
                raise StrikeGoalValidationError(
                    "actual racket state and shadow target frame do not match"
                )
            dot = max(
                -1.0,
                min(
                    1.0,
                    sum(
                        actual.face_normal[index] * target.face_normal[index]
                        for index in range(3)
                    ),
                ),
            )
            errors = {
                "face_position_error_m": _norm(
                    _subtract(actual.face_contact_position, target.face_contact_position)
                ),
                "link_origin_position_error_m": _norm(
                    _subtract(actual.link_origin_position, target.link_origin_position)
                ),
                "face_velocity_error_mps": _norm(
                    _subtract(actual.face_linear_velocity, target.face_linear_velocity)
                ),
                "normal_angle_error_deg": math.degrees(math.acos(dot)),
            }

        sample = StrikeGoalShadowSample(
            control_step=control_step,
            control_time_s=control_time,
            source_goal=source_goal,
            policy_goal=policy_goal,
            target=target,
            actual=actual,
            normalized_policy_goal=normalized,
            errors=errors,
        )
        self.samples.append(sample)
        self._last_control_step = control_step
        self._last_control_time_s = control_time
        return sample

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "pipeline": "strike_goal_p2_shadow",
            "control_clock_domain": self.latched_goal.control_clock_domain,
            "source_frame": self.source_to_policy_transform.source_frame,
            "policy_frame": self.source_to_policy_transform.target_frame,
            "source_to_policy_transform": {
                "rotation": [list(row) for row in self.source_to_policy_transform.rotation],
                "translation": list(self.source_to_policy_transform.translation),
            },
            "calibration_version": self.contact_calibration.calibration_version,
            "qualified_domain": self.contact_calibration.qualified_domain,
            "action_effect": False,
            "sample_count": len(self.samples),
            "samples": [sample.to_mapping() for sample in self.samples],
        }
