"""Versioned 10-D strike-goal contract shared by training and deployment.

This module deliberately has no Isaac Lab dependency.  It is the single
serialization, validation, frame-transform and normalization boundary for a
Planner goal and for a synthetic training goal.  It does *not* alter the
legacy P9 position-only executor; integration is opt-in in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Literal, Mapping, Sequence


STRIKE_GOAL_DIMENSION = 10
# The version deliberately encodes the non-obvious mixed Planner semantics.
# ``position`` is the predicted ball centre at interception, while ``normal``
# and ``linear_velocity`` describe the idealised racket impact state computed
# by Stage 3.  They are not three fields of one already-calibrated robot TCP.
STRIKE_GOAL_CONTRACT_VERSION = "strike_goal_10d/ball_center_impact_v1"
STRIKE_GOAL_POSITION_SEMANTICS = "predicted_ball_center_at_strike"
STRIKE_GOAL_NORMAL_SEMANTICS = "desired_ideal_racket_face_normal"
STRIKE_GOAL_LINEAR_VELOCITY_SEMANTICS = "desired_ideal_racket_impact_velocity"
ISAAC_PROXY_CONTACT_CALIBRATION_VERSION = "isaac_diagnostic_proxy_contact/v1"
POLICY_RACKET_LINK_POINT_V1 = "pingpang_red_Link_origin/v1"
# ``world`` is the literal header.frame_id currently emitted by
# hope_ws/src/solver/src/solver_node.cpp.  Its geometric convention is the
# HOPE canonical frame (near-side-left table-surface origin, X opponentward,
# Y left from P1, Z up), not an arbitrary simulator world.
HOPE_WORLD_FRAME = "world"
BASE_HEADING_RECEIPT_FRAME_V1 = "base_heading_receipt/v1"
StrikeGoalSource = Literal["planner", "synthetic", "replay"]


class StrikeGoalValidationError(ValueError):
    """Raised when a goal cannot safely enter a policy or transform."""


def _vector3(value: Iterable[float], name: str) -> tuple[float, float, float]:
    values = tuple(float(component) for component in value)
    if len(values) != 3 or not all(math.isfinite(component) for component in values):
        raise StrikeGoalValidationError(f"{name} must contain exactly three finite values")
    return values  # type: ignore[return-value]


def _matrix3(value: Sequence[Sequence[float]], name: str) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(_vector3(row, name) for row in value)
    if len(rows) != 3:
        raise StrikeGoalValidationError(f"{name} must have shape [3, 3]")
    return rows


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return tuple(sum(row[index] * vector[index] for index in range(3)) for row in matrix)  # type: ignore[return-value]


def _add(lhs: Sequence[float], rhs: Sequence[float]) -> tuple[float, float, float]:
    return tuple(lhs[index] + rhs[index] for index in range(3))  # type: ignore[return-value]


def _mat_mul(
    lhs: Sequence[Sequence[float]], rhs: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            sum(lhs[row][inner] * rhs[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


@dataclass(frozen=True)
class StrikeGoal10D:
    """The versioned 10-D command emitted by the current Planner adapter.

    ``frame_id`` names one coordinate convention for all 9 vector values.
    The module does not silently reinterpret frozen receipt-frame values as
    current-base values: callers must provide an explicit transform.  For the
    current Planner adapter, ``position`` is the predicted ball centre at
    strike.  ``normal`` and ``linear_velocity`` are the ideal racket impact
    state from the ball-impact solver; the velocity is not yet qualified as
    the linear velocity of a named robot TCP.  Converting this command to a
    robot reference point therefore requires a separately calibrated contact
    transform (and, for exact point velocities, an angular-velocity contract).
    """

    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    linear_velocity: tuple[float, float, float]
    time_to_hit_s: float
    frame_id: str
    source: StrikeGoalSource
    receipt_time_s: float | None = None
    contract_version: str = STRIKE_GOAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector3(self.position, "position"))
        object.__setattr__(self, "normal", _vector3(self.normal, "normal"))
        object.__setattr__(self, "linear_velocity", _vector3(self.linear_velocity, "linear_velocity"))
        if not math.isfinite(float(self.time_to_hit_s)) or float(self.time_to_hit_s) < 0.0:
            raise StrikeGoalValidationError("time_to_hit_s must be finite and non-negative")
        object.__setattr__(self, "time_to_hit_s", float(self.time_to_hit_s))
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise StrikeGoalValidationError("frame_id must be a non-empty string")
        if self.source not in ("planner", "synthetic", "replay"):
            raise StrikeGoalValidationError("source must be planner, synthetic, or replay")
        if self.receipt_time_s is not None and not math.isfinite(float(self.receipt_time_s)):
            raise StrikeGoalValidationError("receipt_time_s must be finite when supplied")
        if self.contract_version != STRIKE_GOAL_CONTRACT_VERSION:
            raise StrikeGoalValidationError(
                f"unsupported strike-goal contract version: {self.contract_version!r}"
            )

    @classmethod
    def from_vector(
        cls,
        values: Sequence[float],
        *,
        frame_id: str,
        source: StrikeGoalSource,
        receipt_time_s: float | None = None,
    ) -> "StrikeGoal10D":
        vector = tuple(float(value) for value in values)
        if len(vector) != STRIKE_GOAL_DIMENSION or not all(math.isfinite(value) for value in vector):
            raise StrikeGoalValidationError("strike goal vector must contain exactly 10 finite values")
        return cls(vector[:3], vector[3:6], vector[6:9], vector[9], frame_id, source, receipt_time_s)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StrikeGoal10D":
        expected = {
            "contract_version",
            "frame_id",
            "source",
            "position",
            "normal",
            "linear_velocity",
            "time_to_hit_s",
            "receipt_time_s",
        }
        unknown = set(payload) - expected
        if unknown:
            raise StrikeGoalValidationError(f"unsupported strike-goal fields: {sorted(unknown)}")
        required = expected - {"receipt_time_s"}
        missing = required - set(payload)
        if missing:
            raise StrikeGoalValidationError(f"missing strike-goal fields: {sorted(missing)}")
        return cls(
            position=payload["position"],
            normal=payload["normal"],
            linear_velocity=payload["linear_velocity"],
            time_to_hit_s=payload["time_to_hit_s"],
            frame_id=payload["frame_id"],
            source=payload["source"],
            receipt_time_s=payload.get("receipt_time_s"),
            contract_version=payload["contract_version"],
        )

    def to_vector(self) -> tuple[float, ...]:
        return (*self.position, *self.normal, *self.linear_velocity, self.time_to_hit_s)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "frame_id": self.frame_id,
            "source": self.source,
            "position": list(self.position),
            "normal": list(self.normal),
            "linear_velocity": list(self.linear_velocity),
            "time_to_hit_s": self.time_to_hit_s,
            "receipt_time_s": self.receipt_time_s,
        }

    def advance(self, elapsed_s: float) -> "StrikeGoal10D":
        """Return the same physical goal with a safely decremented time-to-hit."""

        if not math.isfinite(float(elapsed_s)) or float(elapsed_s) < 0.0:
            raise StrikeGoalValidationError("elapsed_s must be finite and non-negative")
        return replace(self, time_to_hit_s=max(self.time_to_hit_s - float(elapsed_s), 0.0))


@dataclass(frozen=True)
class RacketContactTarget:
    """Derived strike state with ball, effective face and link points separated.

    The target face velocity remains the Planner impact-model velocity.  No
    target link-origin velocity is manufactured because an exact rigid-body
    point conversion would require the desired angular velocity, which is not
    present in the current 10-D contract.
    """

    ball_center_position: tuple[float, float, float]
    face_contact_position: tuple[float, float, float]
    link_origin_position: tuple[float, float, float]
    face_normal: tuple[float, float, float]
    face_linear_velocity: tuple[float, float, float]
    time_to_hit_s: float
    frame_id: str
    policy_link_point_id: str
    calibration_version: str
    qualified_domain: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "ball_center_position": list(self.ball_center_position),
            "face_contact_position": list(self.face_contact_position),
            "link_origin_position": list(self.link_origin_position),
            "face_normal": list(self.face_normal),
            "face_linear_velocity": list(self.face_linear_velocity),
            "time_to_hit_s": self.time_to_hit_s,
            "frame_id": self.frame_id,
            "policy_link_point_id": self.policy_link_point_id,
            "calibration_version": self.calibration_version,
            "qualified_domain": self.qualified_domain,
        }


@dataclass(frozen=True)
class AxialRacketContactCalibration:
    """Domain-qualified ball-centre/contact/link mapping along face normal.

    ``link_origin_to_effective_face_along_normal_m`` is positive when the
    effective incoming collision face lies in the positive-normal direction
    from the policy link origin.  This axial representation is sufficient for
    the isolated Isaac proxy; a hardware calibration may replace it with a
    full rigid transform if tangential offsets are measurable.
    """

    ball_radius_m: float
    link_origin_to_effective_face_along_normal_m: float
    calibration_version: str
    qualified_domain: str
    policy_link_point_id: str = POLICY_RACKET_LINK_POINT_V1

    def __post_init__(self) -> None:
        radius = float(self.ball_radius_m)
        face_offset = float(self.link_origin_to_effective_face_along_normal_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise StrikeGoalValidationError("ball_radius_m must be finite and positive")
        if not math.isfinite(face_offset):
            raise StrikeGoalValidationError(
                "link_origin_to_effective_face_along_normal_m must be finite"
            )
        if not self.calibration_version or not self.qualified_domain:
            raise StrikeGoalValidationError(
                "contact calibration version and qualified domain must be non-empty"
            )
        if not self.policy_link_point_id:
            raise StrikeGoalValidationError("policy_link_point_id must be non-empty")
        object.__setattr__(self, "ball_radius_m", radius)
        object.__setattr__(
            self, "link_origin_to_effective_face_along_normal_m", face_offset
        )

    @property
    def ball_center_to_link_origin_along_normal_m(self) -> float:
        return self.ball_radius_m - self.link_origin_to_effective_face_along_normal_m

    def resolve(self, goal: StrikeGoal10D) -> RacketContactTarget:
        StrikeGoalValidator().validate(goal)
        face_position = tuple(
            goal.position[index] + self.ball_radius_m * goal.normal[index]
            for index in range(3)
        )
        link_position = tuple(
            face_position[index]
            - self.link_origin_to_effective_face_along_normal_m * goal.normal[index]
            for index in range(3)
        )
        return RacketContactTarget(
            ball_center_position=goal.position,
            face_contact_position=face_position,  # type: ignore[arg-type]
            link_origin_position=link_position,  # type: ignore[arg-type]
            face_normal=goal.normal,
            face_linear_velocity=goal.linear_velocity,
            time_to_hit_s=goal.time_to_hit_s,
            frame_id=goal.frame_id,
            policy_link_point_id=self.policy_link_point_id,
            calibration_version=self.calibration_version,
            qualified_domain=self.qualified_domain,
        )


def isaac_diagnostic_proxy_contact_calibration() -> AxialRacketContactCalibration:
    """Return the only contact mapping currently qualified by a collision run.

    The 20 mm ball radius and 3 mm effective face offset reproduce the tested
    ``p_link = p_ball + 17 mm * normal`` mapping.  The function name and domain
    prevent this value from being silently reused as a URDF or hardware TCP.
    """

    return AxialRacketContactCalibration(
        ball_radius_m=0.020,
        link_origin_to_effective_face_along_normal_m=0.003,
        calibration_version=ISAAC_PROXY_CONTACT_CALIBRATION_VERSION,
        qualified_domain="isaac_diagnostic_proxy_only",
    )


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise StrikeGoalValidationError(f"planner RacketCommand is missing {name!r}")
        return value[name]
    if not hasattr(value, name):
        raise StrikeGoalValidationError(f"planner RacketCommand is missing {name!r}")
    return getattr(value, name)


def _ros_vector3(value: Any, name: str) -> tuple[float, float, float]:
    return _vector3((_field(value, "x"), _field(value, "y"), _field(value, "z")), name)


def _optional_field(value: Any, name: str) -> Any | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _ros_stamp_seconds(header: Any) -> float | None:
    """Preserve source time without silently treating it as receipt/control time."""

    stamp = _optional_field(header, "stamp")
    if stamp is None:
        return None
    sec = _field(stamp, "sec")
    nanosec = _field(stamp, "nanosec")
    if isinstance(sec, bool) or isinstance(nanosec, bool):
        raise StrikeGoalValidationError("planner header stamp components must be numeric")
    seconds = float(sec) + float(nanosec) * 1.0e-9
    if not math.isfinite(seconds):
        raise StrikeGoalValidationError("planner header stamp must be finite")
    return seconds


@dataclass(frozen=True)
class PlannerRacketCommand:
    """Lossless policy-relevant view of the existing ROS ``RacketCommand``.

    ``goal`` is the candidate 10-D policy input.  Its field semantics are
    fixed by ``STRIKE_GOAL_CONTRACT_VERSION``; it must not enter an actor until
    the contact-point and clock contracts pass P1.  The remaining values are Planner
    metadata and safety facts, deliberately kept outside that vector rather
    than silently discarded or accidentally learned as task coordinates.
    """

    goal: StrikeGoal10D
    strike_time_s: float
    ball_velocity_incoming: tuple[float, float, float]
    ball_velocity_outgoing: tuple[float, float, float]
    valid: bool
    clears_net: bool
    bypasses_net_posts: bool
    predicted_bounces: int
    header_stamp_s: float | None = None

    @classmethod
    def from_ros_message(cls, message: Any) -> "PlannerRacketCommand":
        """Adapt a ROS message object or equivalent nested mapping.

        Expected fields exactly follow ``hope_ws/src/msgs/msg/RacketCommand.msg``.
        The message's ``header.frame_id`` is authoritative; a missing or
        non-canonical frame is rejected instead of being relabelled.
        """

        header = _field(message, "header")
        frame_id = str(_field(header, "frame_id"))
        header_stamp_s = _ros_stamp_seconds(header)
        if frame_id != HOPE_WORLD_FRAME:
            raise StrikeGoalValidationError(
                f"planner RacketCommand frame must be {HOPE_WORLD_FRAME!r}, got {frame_id!r}"
            )
        valid = _field(message, "valid")
        if not isinstance(valid, bool) or not valid:
            raise StrikeGoalValidationError("planner RacketCommand is invalid")
        strike_time_s = float(_field(message, "strike_time"))
        if not math.isfinite(strike_time_s):
            raise StrikeGoalValidationError("planner RacketCommand strike_time must be finite")
        predicted_bounces = _field(message, "predicted_bounces")
        if not isinstance(predicted_bounces, int) or predicted_bounces < 0:
            raise StrikeGoalValidationError("planner RacketCommand predicted_bounces must be non-negative integer")
        goal = StrikeGoal10D(
            position=_ros_vector3(_field(message, "position"), "planner position"),
            normal=_ros_vector3(_field(message, "normal"), "planner normal"),
            linear_velocity=_ros_vector3(_field(message, "velocity"), "planner velocity"),
            time_to_hit_s=float(_field(message, "time_to_strike")),
            frame_id=frame_id,
            source="planner",
            receipt_time_s=None,
        )
        return cls(
            goal=StrikeGoalValidator(accepted_frames=(HOPE_WORLD_FRAME,)).validate(
                goal, require_future_hit=True
            ),
            strike_time_s=strike_time_s,
            ball_velocity_incoming=_ros_vector3(
                _field(message, "ball_velocity_incoming"), "planner ball_velocity_incoming"
            ),
            ball_velocity_outgoing=_ros_vector3(
                _field(message, "ball_velocity_outgoing"), "planner ball_velocity_outgoing"
            ),
            valid=True,
            clears_net=bool(_field(message, "clears_net")),
            bypasses_net_posts=bool(_field(message, "bypasses_net_posts")),
            predicted_bounces=predicted_bounces,
            header_stamp_s=header_stamp_s,
        )


@dataclass(frozen=True)
class LatchedStrikeGoal:
    """A Planner goal tied to one explicit monotonic control clock.

    The Planner message's ``time_to_strike`` is a source-time value.  The
    current solver forwards it unchanged, so communication and pipeline delay
    must be removed before latching when that delay has actually been
    measured.  After receipt, countdown uses only the selected control clock.

    ``verified_pre_receipt_delay_s`` is never inferred here.  Set it from a
    measured same-clock/clock-synchronization service, or leave it at zero and
    treat the remaining pre-receipt latency as an unresolved P1 item.
    """

    goal_at_receipt: StrikeGoal10D
    received_control_time_s: float
    control_clock_domain: str
    verified_pre_receipt_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.received_control_time_s)):
            raise StrikeGoalValidationError("received_control_time_s must be finite")
        if not isinstance(self.control_clock_domain, str) or not self.control_clock_domain:
            raise StrikeGoalValidationError("control_clock_domain must be a non-empty string")
        delay = float(self.verified_pre_receipt_delay_s)
        if not math.isfinite(delay) or delay < 0.0:
            raise StrikeGoalValidationError(
                "verified_pre_receipt_delay_s must be finite and non-negative"
            )
        object.__setattr__(self, "received_control_time_s", float(self.received_control_time_s))
        object.__setattr__(self, "verified_pre_receipt_delay_s", delay)
        effective_time = max(self.goal_at_receipt.time_to_hit_s - delay, 0.0)
        object.__setattr__(
            self,
            "goal_at_receipt",
            replace(
                self.goal_at_receipt,
                time_to_hit_s=effective_time,
                receipt_time_s=float(self.received_control_time_s),
            ),
        )

    @classmethod
    def from_planner_command(
        cls,
        command: PlannerRacketCommand,
        *,
        received_control_time_s: float,
        control_clock_domain: str,
        verified_pre_receipt_delay_s: float = 0.0,
    ) -> "LatchedStrikeGoal":
        return cls(
            command.goal,
            received_control_time_s,
            control_clock_domain,
            verified_pre_receipt_delay_s,
        )

    def goal_at(self, current_control_time_s: float) -> StrikeGoal10D:
        current_time = float(current_control_time_s)
        if not math.isfinite(current_time):
            raise StrikeGoalValidationError("current_control_time_s must be finite")
        elapsed = current_time - self.received_control_time_s
        if elapsed < -1.0e-12:
            raise StrikeGoalValidationError(
                "control clock moved backwards relative to goal receipt"
            )
        return self.goal_at_receipt.advance(max(elapsed, 0.0))

    def is_expired(self, current_control_time_s: float) -> bool:
        return self.goal_at(current_control_time_s).time_to_hit_s <= 0.0


@dataclass(frozen=True)
class StrikeGoalValidator:
    """Contract-level validation; workspace feasibility remains a later service."""

    normal_tolerance: float = 1.0e-3
    min_time_to_hit_s: float = 0.0
    accepted_frames: tuple[str, ...] = ()

    def validate(self, goal: StrikeGoal10D, *, require_future_hit: bool = False) -> StrikeGoal10D:
        if self.normal_tolerance < 0.0 or not math.isfinite(self.normal_tolerance):
            raise StrikeGoalValidationError("normal_tolerance must be finite and non-negative")
        normal_length = _norm(goal.normal)
        if abs(normal_length - 1.0) > self.normal_tolerance:
            raise StrikeGoalValidationError(
                f"normal must have unit length within {self.normal_tolerance:g}; got {normal_length:.9g}"
            )
        if goal.time_to_hit_s < self.min_time_to_hit_s:
            raise StrikeGoalValidationError(
                f"time_to_hit_s={goal.time_to_hit_s:g} is below minimum {self.min_time_to_hit_s:g}"
            )
        if require_future_hit and goal.time_to_hit_s <= 0.0:
            raise StrikeGoalValidationError("time_to_hit_s is expired")
        if self.accepted_frames and goal.frame_id not in self.accepted_frames:
            raise StrikeGoalValidationError(f"unaccepted goal frame: {goal.frame_id!r}")
        return goal


@dataclass(frozen=True)
class StrikeGoalFrameTransform:
    """Rigid transform from ``source_frame`` to ``target_frame``.

    ``rotation`` maps source-frame vectors to target-frame vectors.  The
    translation is the target-frame position of the source origin; it applies
    to position only, never normal or velocity.
    """

    source_frame: str
    target_frame: str
    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", _matrix3(self.rotation, "rotation"))
        object.__setattr__(self, "translation", _vector3(self.translation, "translation"))
        if not self.source_frame or not self.target_frame:
            raise StrikeGoalValidationError("source_frame and target_frame must be non-empty")

    def apply(self, goal: StrikeGoal10D) -> StrikeGoal10D:
        if goal.frame_id != self.source_frame:
            raise StrikeGoalValidationError(
                f"frame mismatch: transform expects {self.source_frame!r}, got {goal.frame_id!r}"
            )
        return replace(
            goal,
            position=_add(_mat_vec(self.rotation, goal.position), self.translation),
            normal=_mat_vec(self.rotation, goal.normal),
            linear_velocity=_mat_vec(self.rotation, goal.linear_velocity),
            frame_id=self.target_frame,
        )

    def followed_by(self, next_transform: "StrikeGoalFrameTransform") -> "StrikeGoalFrameTransform":
        """Compose this source->middle transform with a middle->target one."""

        if self.target_frame != next_transform.source_frame:
            raise StrikeGoalValidationError(
                "cannot compose strike-goal transforms with mismatched middle frames"
            )
        rotation = _mat_mul(next_transform.rotation, self.rotation)
        translation = _add(
            _mat_vec(next_transform.rotation, self.translation),
            next_transform.translation,
        )
        return StrikeGoalFrameTransform(
            source_frame=self.source_frame,
            target_frame=next_transform.target_frame,
            rotation=rotation,
            translation=translation,
        )


@dataclass(frozen=True)
class StrikeGoalNormalizer:
    """Shared, explicit scales for policy observation preprocessing."""

    position_scale_m: tuple[float, float, float]
    velocity_scale_mps: tuple[float, float, float]
    time_scale_s: float
    contract_version: str = STRIKE_GOAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_scale_m", _vector3(self.position_scale_m, "position_scale_m"))
        object.__setattr__(self, "velocity_scale_mps", _vector3(self.velocity_scale_mps, "velocity_scale_mps"))
        if any(scale <= 0.0 for scale in self.position_scale_m + self.velocity_scale_mps):
            raise StrikeGoalValidationError("position and velocity scales must be positive")
        if not math.isfinite(float(self.time_scale_s)) or self.time_scale_s <= 0.0:
            raise StrikeGoalValidationError("time_scale_s must be finite and positive")
        if self.contract_version != STRIKE_GOAL_CONTRACT_VERSION:
            raise StrikeGoalValidationError("normalizer contract version mismatch")

    def normalize(self, goal: StrikeGoal10D) -> tuple[float, ...]:
        return (
            *(goal.position[index] / self.position_scale_m[index] for index in range(3)),
            *goal.normal,
            *(goal.linear_velocity[index] / self.velocity_scale_mps[index] for index in range(3)),
            goal.time_to_hit_s / self.time_scale_s,
        )

    def denormalize(
        self,
        values: Sequence[float],
        *,
        frame_id: str,
        source: StrikeGoalSource,
        receipt_time_s: float | None = None,
    ) -> StrikeGoal10D:
        vector = tuple(float(value) for value in values)
        if len(vector) != STRIKE_GOAL_DIMENSION or not all(math.isfinite(value) for value in vector):
            raise StrikeGoalValidationError("normalized strike goal must contain exactly 10 finite values")
        return StrikeGoal10D(
            position=tuple(vector[index] * self.position_scale_m[index] for index in range(3)),
            normal=vector[3:6],
            linear_velocity=tuple(vector[index + 6] * self.velocity_scale_mps[index] for index in range(3)),
            time_to_hit_s=vector[9] * self.time_scale_s,
            frame_id=frame_id,
            source=source,
            receipt_time_s=receipt_time_s,
        )


@dataclass(frozen=True)
class StrikeGoalTrace:
    """Immutable audit record for one policy observation boundary.

    This is intentionally independent of Isaac tensors so the same raw and
    normalized values can be written by a synthetic generator, Planner bridge,
    simulator, or deployment wrapper.
    """

    policy_step: int
    goal: StrikeGoal10D
    normalized_goal: tuple[float, ...] | None

    @classmethod
    def capture(
        cls,
        policy_step: int,
        goal: StrikeGoal10D,
        normalizer: StrikeGoalNormalizer | None = None,
    ) -> "StrikeGoalTrace":
        if not isinstance(policy_step, int) or policy_step < 0:
            raise StrikeGoalValidationError("policy_step must be a non-negative integer")
        normalized = None if normalizer is None else normalizer.normalize(goal)
        return cls(policy_step=policy_step, goal=goal, normalized_goal=normalized)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "policy_step": self.policy_step,
            "goal": self.goal.to_mapping(),
            "normalized_goal": None if self.normalized_goal is None else list(self.normalized_goal),
        }
