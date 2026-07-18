"""Validation for stance-aware strike manifest metadata.

This module intentionally has no Isaac imports.  It validates the data contract
before a manifest is allowed to become a training input.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_TOLERANCE_M = 2.0e-5


def _vector(value: Any, name: str, size: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must be a {size}-vector, got {value!r}")
    result = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} contains non-finite values: {value!r}")
    return result


def _field(mapping: Mapping[str, Any], name: str, parent: str) -> Any:
    if name not in mapping:
        raise ValueError(f"{parent} is missing required field '{name}'")
    return mapping[name]


def _pose(mapping: Mapping[str, Any], name: str) -> tuple[tuple[float, ...], float]:
    position = _vector(_field(mapping, "position_m", name), f"{name}.position_m", 3)
    yaw = float(_field(mapping, "yaw_rad", name))
    if not math.isfinite(yaw):
        raise ValueError(f"{name}.yaw_rad is non-finite")
    return position, yaw


def _rotate_z(vector: tuple[float, ...], yaw: float) -> tuple[float, ...]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * vector[0] - s * vector[1], s * vector[0] + c * vector[1], vector[2])


def _max_abs_delta(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def validate_stance_entry(
    entry: Mapping[str, Any],
    *,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    expected_mode: str | None = "prepositioned",
) -> dict[str, Any]:
    """Validate one entry and return measured consistency errors.

    ``original_hit_position_w_m`` is the immutable source/planning point.  A
    native-calibrated ``strike_target`` may intentionally differ, so it is only
    checked for finite values here and is not substituted into this geometry
    identity.
    """

    metadata = entry.get("stance_metadata")
    if not isinstance(metadata, Mapping):
        # Legacy fixed-base manifests predate the stance-aware contract and
        # intentionally omit metadata.  An explicit ``fixed`` validation is
        # still useful for those manifests, but must not require fields that
        # only exist for prepositioned/mixed entries.
        if expected_mode == "fixed":
            _vector(
                entry.get("strike_target", {}).get("racket_position_m", [0.0] * 3),
                "strike_target.racket_position_m",
                3,
            )
            return {
                "episode_id": str(entry.get("episode_id", "")),
                "stance_mode": "fixed",
                "metadata_present": False,
                "offset_error_m": 0.0,
                "world_hit_reconstruction_error_m": 0.0,
                "base_yaw_delta_rad": 0.0,
            }
        raise ValueError(f"{entry.get('episode_id', '<unknown>')}: missing stance_metadata")

    mode = str(_field(metadata, "stance_mode", "stance_metadata"))
    if expected_mode is not None and expected_mode != "mixed" and mode != expected_mode:
        raise ValueError(
            f"{entry.get('episode_id', '<unknown>')}: stance_mode={mode!r}, "
            f"expected {expected_mode!r}"
        )

    original_hit = _vector(
        _field(metadata, "original_hit_position_w_m", "stance_metadata"),
        "stance_metadata.original_hit_position_w_m",
        3,
    )
    before, before_yaw = _pose(
        _field(metadata, "base_pose_before_w", "stance_metadata"),
        "stance_metadata.base_pose_before_w",
    )
    target, target_yaw = _pose(
        _field(metadata, "base_pose_target_w", "stance_metadata"),
        "stance_metadata.base_pose_target_w",
    )
    offset = _vector(
        _field(metadata, "stance_offset_xy_w_m", "stance_metadata"),
        "stance_metadata.stance_offset_xy_w_m",
        2,
    )
    target_base = _vector(
        _field(metadata, "strike_target_base_m", "stance_metadata"),
        "stance_metadata.strike_target_base_m",
        3,
    )

    expected_offset = (target[0] - before[0], target[1] - before[1])
    offset_error = _max_abs_delta(offset, expected_offset)
    if offset_error > tolerance_m:
        raise ValueError(
            f"{entry.get('episode_id', '<unknown>')}: stance offset mismatch "
            f"error={offset_error:.6g} m"
        )

    reconstructed_hit = tuple(
        target[i] + _rotate_z(target_base, target_yaw)[i] for i in range(3)
    )
    hit_error = _max_abs_delta(original_hit, reconstructed_hit)
    if hit_error > tolerance_m:
        raise ValueError(
            f"{entry.get('episode_id', '<unknown>')}: base-relative strike target "
            f"does not reconstruct original world hit, error={hit_error:.6g} m"
        )

    # The base pose before/after must be finite and have a defined yaw.  A yaw
    # change is permitted by the contract; the target pose is what defines the
    # base-frame reconstruction above.
    _ = before_yaw
    _ = _vector(entry.get("strike_target", {}).get("racket_position_m", [0.0] * 3),
                "strike_target.racket_position_m", 3)

    return {
        "episode_id": str(entry.get("episode_id", "")),
        "stance_mode": mode,
        "metadata_present": True,
        "offset_error_m": offset_error,
        "world_hit_reconstruction_error_m": hit_error,
        "base_yaw_delta_rad": target_yaw - before_yaw,
    }


def validate_stance_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_file: str | None = None,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    expected_mode: str | None = "prepositioned",
    check_motion_paths: bool = False,
) -> dict[str, Any]:
    """Validate all stance metadata in a manifest.

    This is opt-in because ordinary fixed-base manifests intentionally have no
    stance metadata.  ``check_motion_paths`` is useful for a pre-training gate;
    the normal loader still performs its own motion-path resolution.
    """

    entries = manifest.get("motions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("stance manifest must contain a non-empty 'motions' list")

    top_contract = manifest.get("stance_contract")
    if isinstance(top_contract, Mapping):
        top_mode = top_contract.get("mode")
        if (
            top_mode is not None
            and expected_mode is not None
            and expected_mode != "mixed"
            and str(top_mode) != expected_mode
        ):
            raise ValueError(f"stance_contract.mode={top_mode!r}, expected {expected_mode!r}")

    results = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"manifest motion entry is not an object: {entry!r}")
        result = validate_stance_entry(entry, tolerance_m=tolerance_m, expected_mode=expected_mode)
        if check_motion_paths and manifest_file:
            candidates = [entry.get("motion_npz"), entry.get("library_motion_npz"), entry.get("motion_path")]
            path = next((p for p in candidates if p), None)
            if path is None:
                raise ValueError(f"{result['episode_id']}: no motion path in stance manifest")
            resolved = os.path.expanduser(str(path))
            if not os.path.isabs(resolved):
                resolved = os.path.join(os.path.dirname(os.path.abspath(manifest_file)), resolved)
            if not os.path.isfile(resolved):
                raise FileNotFoundError(f"{result['episode_id']}: motion path does not exist: {resolved}")
            # Legacy fixed-base entries can be path-checked without the
            # prepositioned root-pose identity below; they have no stance
            # metadata by design.
            if not isinstance(entry.get("stance_metadata"), Mapping):
                results.append(result)
                continue
            data = np.load(resolved)
            body_pos = np.asarray(data.get("body_pos_w"))
            if body_pos.ndim != 3 or body_pos.shape[1:] != (32, 3):
                raise ValueError(
                    f"{result['episode_id']}: body_pos_w shape {body_pos.shape}, expected [T,32,3]"
                )
            hit_frame = int(entry.get("hit_event", {}).get("motion_hit_frame", -1))
            if not 0 <= hit_frame < body_pos.shape[0]:
                raise ValueError(f"{result['episode_id']}: invalid motion_hit_frame={hit_frame}")
            target_position = _pose(
                entry["stance_metadata"]["base_pose_target_w"],
                "stance_metadata.base_pose_target_w",
            )[0]
            root_at_hit = tuple(float(x) for x in body_pos[hit_frame, 0])
            root_error = _max_abs_delta(root_at_hit, target_position)
            if root_error > tolerance_m:
                raise ValueError(
                    f"{result['episode_id']}: NPZ root at hit does not match target base pose, "
                    f"error={root_error:.6g} m"
                )
            root_xy_span = np.ptp(body_pos[:, 0, :2], axis=0)
            if float(np.max(root_xy_span)) > tolerance_m:
                raise ValueError(
                    f"{result['episode_id']}: prepositioned NPZ root moves during clip, "
                    f"xy_span={root_xy_span.tolist()}"
                )
            result["npz_root_at_hit_error_m"] = root_error
            result["npz_root_xy_span_m"] = [float(x) for x in root_xy_span]
        results.append(result)

    return {
        "motion_count": len(results),
        "expected_mode": expected_mode or "any",
        "max_offset_error_m": max(r["offset_error_m"] for r in results),
        "max_world_hit_reconstruction_error_m": max(
            r["world_hit_reconstruction_error_m"] for r in results
        ),
        "max_npz_root_at_hit_error_m": max(
            r.get("npz_root_at_hit_error_m", 0.0) for r in results
        ),
        "motions": results,
    }
