"""Pure-Python HOPE Schema-2 and deterministic lifecycle prototype."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable
import numpy as np

from .hope_open_source_contract import ContractMetadata, select_nearest_station_side


class TimingError(ValueError): pass
class SchemaError(ValueError): pass
class IdentityError(ValueError): pass


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector3(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise SchemaError(f"{name} must be three finite values")
    return result


def age_command_timing(header_stamp_s, source_now_s, command_tts_s) -> float:
    header = _finite(header_stamp_s, "header_stamp_s")
    now = _finite(source_now_s, "source_now_s")
    tts = _finite(command_tts_s, "command_tts_s")
    control_tts = tts - max(0.0, now - header)
    if control_tts <= 0.0:
        raise TimingError("command is expired")
    return control_tts


def reanchor_to_wall(control_tts_s, producer_wall_s) -> tuple[float, int, int]:
    tts = _finite(control_tts_s, "control_tts_s")
    wall = _finite(producer_wall_s, "producer_wall_s")
    if tts <= 0.0 or wall <= 0.0:
        raise TimingError("control_tts_s and producer_wall_s must be positive")
    sec = math.floor(wall)
    nsec = round((wall - sec) * 1_000_000_000.0)
    if nsec == 1_000_000_000:
        sec, nsec = sec + 1, 0
    return wall + tts, sec, nsec


@dataclass(frozen=True)
class FlightRevision:
    command_seq: int
    flight_id: int
    revision_id: int
    strike_time_s: float


class FlightRevisionManager:
    """Deterministic POSSIBLE_WITH_RULE lifecycle; not claimed physically SAFE."""
    def __init__(self, shot_reuse_tolerance_s: float = 0.050) -> None:
        tolerance = _finite(shot_reuse_tolerance_s, "shot_reuse_tolerance_s")
        if tolerance < 0.0:
            raise ValueError("shot_reuse_tolerance_s must be non-negative")
        self.shot_reuse_tolerance_s = tolerance
        self.next_flight_id = 1
        self.active_flight_id = None
        self.revision_id = 0
        self.active_strike_time = None
        self.consumed = False
        self.expired = False
        self.command_seq = 0
        self._last_seen_command_seq = 0
        self._last_seen_revision_by_flight = {}

    def observe_valid(self, strike_time_s: float) -> FlightRevision:
        strike = _finite(strike_time_s, "strike_time_s")
        if strike <= 0.0:
            raise IdentityError("strike_time_s must be positive")
        new_flight = (self.active_flight_id is None or self.consumed or self.expired
                      or self.active_strike_time is None
                      or abs(strike - self.active_strike_time) > self.shot_reuse_tolerance_s)
        if new_flight:
            self.active_flight_id = self.next_flight_id
            self.next_flight_id += 1
            self.revision_id = 1
            self.consumed = self.expired = False
        else:
            self.revision_id += 1
        self.active_strike_time = strike
        self.command_seq += 1
        return FlightRevision(self.command_seq, self.active_flight_id, self.revision_id, strike)

    def mark_consumed(self):
        if self.active_flight_id is None: raise IdentityError("no active flight")
        self.consumed = True

    def mark_expired(self):
        if self.active_flight_id is None: raise IdentityError("no active flight")
        self.expired = True

    def validate_received_identity(self, command_seq, flight_id, revision_id):
        values = (int(command_seq), int(flight_id), int(revision_id))
        if any(value <= 0 for value in values):
            raise IdentityError("identity values must be positive")
        if values[0] <= self._last_seen_command_seq:
            raise IdentityError("duplicate or reordered command_seq")
        prior = self._last_seen_revision_by_flight.get(values[1], 0)
        if values[2] <= prior:
            raise IdentityError("duplicate or reordered same-flight revision_id")
        self._last_seen_command_seq = values[0]
        self._last_seen_revision_by_flight[values[1]] = values[2]


class AdapterStationMirror:
    """Adapter-owned station mirror; requires native parity testing before deployment."""
    def __init__(self, metadata: ContractMetadata) -> None:
        self.metadata = metadata
        self.held_station_xy = None
        self.active_flight_id = None
        self.locked_swing_sign = None

    def candidate_for(self, flight_id: int, target_xy, current_base_xy):
        if int(flight_id) <= 0:
            raise IdentityError("flight_id must be positive")
        target = np.asarray(tuple(target_xy), dtype=np.float64)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise SchemaError("target_xy must be two finite values")
        if self.active_flight_id != int(flight_id):
            anchor = self.held_station_xy if self.held_station_xy is not None else current_base_xy
            sign, station = select_nearest_station_side(target, anchor, self.metadata)
            self.active_flight_id = int(flight_id)
            self.locked_swing_sign = sign
        else:
            clip = 0 if self.locked_swing_sign == 1 else 1
            station = target - np.asarray(self.metadata.reach_offsets[clip])
        return int(self.locked_swing_sign), np.asarray(station, dtype=np.float64)

    def accept_candidate(self, flight_id: int, swing_sign: int, station_xy) -> None:
        if int(flight_id) != self.active_flight_id or swing_sign != self.locked_swing_sign:
            raise IdentityError("candidate does not match active flight/locked side")
        station = np.asarray(tuple(station_xy), dtype=np.float64)
        if station.shape != (2,) or not np.all(np.isfinite(station)):
            raise SchemaError("station_xy must be two finite values")
        self.held_station_xy = station.copy()


def build_schema2_packet(*, valid: bool, swing_sign: int, position, velocity,
                         control_tts_s, producer_wall_s, command_seq, flight_id,
                         revision_id, estimator_sample_count=0, estimator_span_s=0.0):
    if not isinstance(valid, (bool, np.bool_)): raise SchemaError("valid must be boolean")
    if swing_sign not in (-1, 1): raise SchemaError("swing_sign must be exactly +1 or -1")
    pos, vel = _vector3(position, "position"), _vector3(velocity, "velocity")
    tts, span = _finite(control_tts_s, "control_tts_s"), _finite(estimator_span_s, "estimator_span_s")
    if tts <= 0.0 or span < 0.0: raise SchemaError("invalid timing/span")
    ints = (command_seq, flight_id, revision_id, estimator_sample_count)
    if any(isinstance(v, bool) or int(v) != v for v in ints): raise SchemaError("integer field required")
    command_seq, flight_id, revision_id, estimator_sample_count = map(int, ints)
    if min(command_seq, flight_id, revision_id) <= 0 or not 0 <= estimator_sample_count <= 10000:
        raise SchemaError("identity/sample range invalid")
    strike_wall, sec, nsec = reanchor_to_wall(tts, producer_wall_s)
    packet = np.array([2.0, float(valid), float(swing_sign), *pos, *vel, tts, strike_wall,
                       0.0, float(sec), float(nsec), float(command_seq), float(flight_id),
                       float(revision_id), float(estimator_sample_count), span], dtype=np.float64)
    if packet.shape != (19,) or not np.all(np.isfinite(packet)):
        raise SchemaError("packet must be finite float64 shape (19,)")
    if abs(packet[10] - (packet[12] + packet[13] * 1e-9 + packet[9])) > 1e-9:
        raise SchemaError("wall re-anchor invariant failed")
    return packet
