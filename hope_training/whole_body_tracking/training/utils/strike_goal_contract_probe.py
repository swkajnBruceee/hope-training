"""Record and audit the still-unresolved TCP and time parts of P1.

This module intentionally does not depend on ROS or Isaac Lab.  A ROS bridge
or an Isaac evaluation hook supplies values in its own native types, then
serializes the dictionaries produced by the sample classes.  The analyser is
strict about clock domains: it never subtracts timestamps from different
clocks unless an explicit source-to-control offset has been supplied.

The TCP sample compares the *raw Planner command position* to the policy TCP;
it does not assume that the Planner field already names a physical racket
point.  That distinction is material in this repository because the current
solver assigns the predicted ball centre to ``RacketCommand.position``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import fmean
from typing import Any, Mapping, Sequence

from .strike_goal import StrikeGoalValidationError


def _vec3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise StrikeGoalValidationError(f"{name} must have exactly three values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise StrikeGoalValidationError(f"{name} must be finite")
    return result  # type: ignore[return-value]


def _matrix3(value: Sequence[Sequence[float]], name: str) -> tuple[tuple[float, float, float], ...]:
    if len(value) != 3:
        raise StrikeGoalValidationError(f"{name} must have shape [3, 3]")
    return tuple(_vec3(row, name) for row in value)


def _sub(lhs: Sequence[float], rhs: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(lhs[index]) - float(rhs[index]) for index in range(3))  # type: ignore[return-value]


def _transpose_mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][column] * vector[row] for row in range(3)) for column in range(3))  # type: ignore[return-value]


def _dot(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    return sum(float(lhs[index]) * float(rhs[index]) for index in range(3))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _mean_vector(values: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    return tuple(fmean(value[index] for value in values) for index in range(3))  # type: ignore[return-value]


def _std_vector(values: Sequence[Sequence[float]], mean: Sequence[float]) -> tuple[float, float, float]:
    return tuple(
        math.sqrt(fmean((value[index] - mean[index]) ** 2 for value in values)) for index in range(3)
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class TcpProbeSample:
    """One synchronized Planner/FK observation at a deliberately chosen pose.

    ``world_from_racket_rotation`` is a 3x3 rotation matrix whose columns are
    the racket-frame axes in world coordinates.  Recording this explicit
    matrix avoids hidden ROS/Isaac quaternion-order assumptions.
    """

    sample_id: str
    timestamp_s: float
    pose_label: str
    planner_command_position_world: tuple[float, float, float]
    planner_normal_world: tuple[float, float, float]
    policy_tcp_position_world: tuple[float, float, float]
    policy_tcp_normal_world: tuple[float, float, float]
    racket_link_origin_world: tuple[float, float, float]
    world_from_racket_rotation: tuple[tuple[float, float, float], ...]
    policy_tcp_name: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.pose_label or not self.policy_tcp_name:
            raise StrikeGoalValidationError("TCP probe identifiers must be non-empty")
        if not math.isfinite(float(self.timestamp_s)):
            raise StrikeGoalValidationError("TCP probe timestamp_s must be finite")
        for field_name in (
            "planner_command_position_world",
            "planner_normal_world",
            "policy_tcp_position_world",
            "policy_tcp_normal_world",
            "racket_link_origin_world",
        ):
            object.__setattr__(self, field_name, _vec3(getattr(self, field_name), field_name))
        object.__setattr__(self, "world_from_racket_rotation", _matrix3(self.world_from_racket_rotation, "world_from_racket_rotation"))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TcpProbeSample":
        return cls(**dict(payload))

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def analyze_tcp_samples(samples: Sequence[TcpProbeSample]) -> dict[str, Any]:
    """Summarize TCP agreement and whether a rigid local offset is plausible.

    This is evidence, not an automatic pass.  The caller chooses acceptance
    tolerance from task/contact requirements after reviewing the report.
    """

    if not samples:
        raise StrikeGoalValidationError("at least one TCP probe sample is required")
    delta_world = [_sub(item.planner_command_position_world, item.policy_tcp_position_world) for item in samples]
    delta_local = [
        _transpose_mat_vec(item.world_from_racket_rotation, difference)
        for item, difference in zip(samples, delta_world, strict=True)
    ]
    normal_angle_deg: list[float] = []
    for item in samples:
        planner_length = _norm(item.planner_normal_world)
        policy_length = _norm(item.policy_tcp_normal_world)
        if planner_length <= 1.0e-12 or policy_length <= 1.0e-12:
            raise StrikeGoalValidationError("TCP probe normals must be non-zero")
        cosine = max(-1.0, min(1.0, _dot(item.planner_normal_world, item.policy_tcp_normal_world) / (planner_length * policy_length)))
        normal_angle_deg.append(math.degrees(math.acos(cosine)))

    world_mean = _mean_vector(delta_world)
    local_mean = _mean_vector(delta_local)
    world_std = _std_vector(delta_world, world_mean)
    local_std = _std_vector(delta_local, local_mean)
    world_norms = [_norm(value) for value in delta_world]
    local_residual_norms = [_norm(_sub(value, local_mean)) for value in delta_local]
    return {
        "sample_count": len(samples),
        "pose_labels": [item.pose_label for item in samples],
        "policy_tcp_names": sorted({item.policy_tcp_name for item in samples}),
        "planner_minus_policy_tcp_world_m": {
            "mean": list(world_mean), "std": list(world_std), "max_norm": max(world_norms),
        },
        "planner_minus_policy_tcp_racket_m": {
            "mean": list(local_mean), "std": list(local_std), "max_residual_norm": max(local_residual_norms),
        },
        "normal_angle_deg": {
            "mean": fmean(normal_angle_deg), "max": max(normal_angle_deg), "min": min(normal_angle_deg),
        },
        "interpretation": (
            "A small local-frame offset spread supports a fixed TCP transform; "
            "it does not prove that the raw Planner position is the desired physical TCP."
        ),
    }


@dataclass(frozen=True)
class TimeProbeSample:
    """One receive/control-time observation for a Planner command.

    ``source_clock_domain`` applies to header/strike timestamps.  The policy
    fields use ``control_clock_domain``.  Analyser output remains explicitly
    cross-domain unless a mapping is provided at analysis time.
    """

    command_id: str
    source_clock_domain: str
    control_clock_domain: str
    header_stamp_s: float
    strike_time_s: float
    message_time_to_strike_s: float
    received_control_time_s: float
    current_control_time_s: float
    policy_time_to_strike_s: float | None = None
    simulation_time_s: float | None = None
    control_step: int | None = None

    def __post_init__(self) -> None:
        if not self.command_id or not self.source_clock_domain or not self.control_clock_domain:
            raise StrikeGoalValidationError("time probe IDs and clock domains must be non-empty")
        for field_name in (
            "header_stamp_s", "strike_time_s", "message_time_to_strike_s",
            "received_control_time_s", "current_control_time_s",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise StrikeGoalValidationError(f"{field_name} must be finite")
        for field_name in ("policy_time_to_strike_s", "simulation_time_s"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(float(value)):
                raise StrikeGoalValidationError(f"{field_name} must be finite when supplied")
        if self.control_step is not None and self.control_step < 0:
            raise StrikeGoalValidationError("control_step must be non-negative when supplied")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TimeProbeSample":
        return cls(**dict(payload))

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def analyze_time_samples(
    samples: Sequence[TimeProbeSample], *, source_to_control_offset_s: float | None = None
) -> dict[str, Any]:
    """Audit intra-source consistency and policy-side countdown semantics.

    ``source_to_control_offset_s`` means ``control_time = source_time +
    offset`` and is only accepted when confirmed by an external clock-sync
    measurement.  It is never guessed from message data.
    """

    if not samples:
        raise StrikeGoalValidationError("at least one time probe sample is required")
    if source_to_control_offset_s is not None and not math.isfinite(float(source_to_control_offset_s)):
        raise StrikeGoalValidationError("source_to_control_offset_s must be finite when supplied")

    source_residuals = [
        (item.strike_time_s - item.header_stamp_s) - item.message_time_to_strike_s for item in samples
    ]
    policy_residuals: list[float] = []
    for item in samples:
        if item.policy_time_to_strike_s is not None:
            expected = max(item.message_time_to_strike_s - (item.current_control_time_s - item.received_control_time_s), 0.0)
            policy_residuals.append(item.policy_time_to_strike_s - expected)

    report: dict[str, Any] = {
        "sample_count": len(samples),
        "source_clock_domains": sorted({item.source_clock_domain for item in samples}),
        "control_clock_domains": sorted({item.control_clock_domain for item in samples}),
        "strike_minus_header_minus_message_tts_s": {
            "mean": fmean(source_residuals),
            "max_abs": max(abs(value) for value in source_residuals),
        },
        "policy_countdown": {
            "samples_with_policy_value": len(policy_residuals),
            "residual_mean_s": fmean(policy_residuals) if policy_residuals else None,
            "residual_max_abs_s": max((abs(value) for value in policy_residuals), default=None),
            "definition": "message_tts - (current_control_time - received_control_time), clamped at zero",
        },
        "interpretation": (
            "The source timestamp relation is safe to check within the Planner clock. "
            "No source/control latency is inferred without an explicit clock mapping."
        ),
    }
    if source_to_control_offset_s is None:
        report["mapped_remaining_time_at_current_control_s"] = None
        report["clock_mapping_status"] = "not_provided"
    else:
        mapped = [
            item.strike_time_s + source_to_control_offset_s - item.current_control_time_s for item in samples
        ]
        report["mapped_remaining_time_at_current_control_s"] = {
            "mean": fmean(mapped), "min": min(mapped), "max": max(mapped),
        }
        report["clock_mapping_status"] = "provided_by_caller"
    return report
