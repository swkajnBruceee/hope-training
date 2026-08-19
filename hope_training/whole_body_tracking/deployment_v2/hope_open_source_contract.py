"""Canonical model_21800 action, frame, position, and side contracts.

Values are identity-checked against metadata embedded in the canonical ONNX.
This reproduces HOPE open-source software, not hardware TCP/mocap calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np

POSITION_CONTRACT = "HOPE_OPEN_SOURCE_SOLVER_POSITION"
FRAME_CONTRACT = "HOPE_OPEN_SOURCE_WORLD_TABLE_FRAME_CODE_0"
MODEL21800_SHA256 = "6bf1a2418f8538e23577a0153f2fe6a1e78dee91f41650a232259432a84a4dc8"

_EXPECTED_METADATA = {
    "hitter_pure_pos_range_per_clip": (
        (0.58, 0.58, -0.48, -0.40, 0.85, 1.30),
        (0.58, 0.58, -0.13, -0.05, 0.85, 1.30),
    ),
    "hitter_pure_vel_core_range_per_clip": (
        (1.24, 2.24, -0.31, 0.69, 0.66, 1.66),
        (1.60, 2.60, -0.66, 0.34, 0.00, 0.54),
    ),
    "hitter_pure_vel_planner_range_per_clip": (
        (1.57, 2.55, 0.10, 0.52, 0.41, 1.35),
        (1.55, 2.52, -0.18, 0.29, 0.40, 1.32),
    ),
}


class ContractError(ValueError):
    """Canonical metadata or contract input is invalid."""


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise ContractError("malformed protobuf varint")


def _protobuf_fields(data: bytes):
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field, wire = tag >> 3, tag & 7
        if field == 0:
            raise ContractError("invalid protobuf field zero")
        if wire == 0:
            value, offset = _read_varint(data, offset)
        elif wire == 1:
            value, offset = data[offset : offset + 8], offset + 8
        elif wire == 2:
            size, offset = _read_varint(data, offset)
            value, offset = data[offset : offset + size], offset + size
            if len(value) != size:
                raise ContractError("truncated protobuf field")
        elif wire == 5:
            value, offset = data[offset : offset + 4], offset + 4
        else:
            raise ContractError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def read_onnx_metadata(path: str | Path) -> dict[str, str]:
    """Read ModelProto metadata_props (field 14), without the optional onnx package."""
    result: dict[str, str] = {}
    for field, wire, entry in _protobuf_fields(Path(path).read_bytes()):
        if field != 14 or wire != 2:
            continue
        key = value = None
        for inner_field, inner_wire, raw in _protobuf_fields(entry):
            if inner_wire == 2 and inner_field == 1:
                key = raw.decode("utf-8")
            elif inner_wire == 2 and inner_field == 2:
                value = raw.decode("utf-8")
        if key is not None and value is not None:
            result[key] = value
    return result


def _parse_boxes(value: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    clips = tuple(tuple(float(x) for x in part.split(",")) for part in value.split(";"))
    if len(clips) != 2 or any(len(box) != 6 for box in clips):
        raise ContractError("expected two six-value per-clip boxes")
    if not all(math.isfinite(x) for box in clips for x in box):
        raise ContractError("metadata box contains non-finite value")
    return clips  # type: ignore[return-value]


@dataclass(frozen=True)
class ContractMetadata:
    model_sha256: str
    position_boxes: tuple[tuple[float, ...], tuple[float, ...]]
    core_velocity_boxes: tuple[tuple[float, ...], tuple[float, ...]]
    planner_velocity_boxes: tuple[tuple[float, ...], tuple[float, ...]]
    reach_offsets: tuple[tuple[float, float], tuple[float, float]]


def load_canonical_metadata(path: str | Path) -> ContractMetadata:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL21800_SHA256:
        raise ContractError(f"MODEL_METADATA_MISMATCH: sha256={digest}")
    raw = read_onnx_metadata(path)
    parsed = {}
    for key, expected in _EXPECTED_METADATA.items():
        if key not in raw:
            raise ContractError(f"MODEL_METADATA_MISMATCH: missing {key}")
        actual = _parse_boxes(raw[key])
        if actual != expected:
            raise ContractError(f"MODEL_METADATA_MISMATCH: {key}={actual!r}")
        parsed[key] = actual
    positions = parsed["hitter_pure_pos_range_per_clip"]
    reach = tuple((0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3])) for b in positions)
    return ContractMetadata(
        digest, positions,
        parsed["hitter_pure_vel_core_range_per_clip"],
        parsed["hitter_pure_vel_planner_range_per_clip"],
        reach,  # type: ignore[arg-type]
    )


def _side_index(swing_sign: int | float) -> int:
    if float(swing_sign) == 1.0:
        return 0
    if float(swing_sign) == -1.0:
        return 1
    raise ContractError("swing_sign must be exactly +1 or -1")


def _finite_vector(values: Iterable[float], name: str, size: int) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ContractError(f"{name} must have shape ({size},) and be finite")
    return vector


def _bounds(box: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    return np.array(box[0::2], dtype=np.float64), np.array(box[1::2], dtype=np.float64)


def map_normalized_velocity(normalized_action, swing_sign, metadata: ContractMetadata) -> np.ndarray:
    action = _finite_vector(normalized_action, "normalized_action", 3)
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ContractError("normalized_action must be inside [-1,1]^3")
    low, high = _bounds(metadata.planner_velocity_boxes[_side_index(swing_sign)])
    return low + 0.5 * (action + 1.0) * (high - low)


def velocity_inside_planner_box(velocity, swing_sign, metadata: ContractMetadata) -> bool:
    vector = _finite_vector(velocity, "velocity", 3)
    low, high = _bounds(metadata.planner_velocity_boxes[_side_index(swing_sign)])
    return bool(np.all(vector >= low) and np.all(vector <= high))


def velocity_inside_native_component_support(
    velocity, swing_sign, metadata: ContractMetadata, gate_margin: float = 0.30
) -> bool:
    vector = _finite_vector(velocity, "velocity", 3)
    margin = float(gate_margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ContractError("gate_margin must be finite and non-negative")
    clip = _side_index(swing_sign)
    inside = []
    for box in (metadata.core_velocity_boxes[clip], metadata.planner_velocity_boxes[clip]):
        low, high = _bounds(box)
        inside.append(np.all(vector >= low - margin) and np.all(vector <= high + margin))
    return bool(inside[0] or inside[1])


def select_nearest_station_side(target_xy, anchor_xy, metadata: ContractMetadata):
    """Return native side and candidate station; exact tie selects forehand."""
    target = _finite_vector(target_xy, "target_xy", 2)
    anchor = _finite_vector(anchor_xy, "anchor_xy", 2)
    stations = tuple(target - np.asarray(offset) for offset in metadata.reach_offsets)
    clip = 0 if np.linalg.norm(stations[0] - anchor) <= np.linalg.norm(stations[1] - anchor) else 1
    return (1 if clip == 0 else -1), stations[clip].copy()
