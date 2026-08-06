"""Canonicalise authored world-frame motions into a base-heading local prior.

The output removes only the source scene's rigid X/Y/yaw placement.  Joint
trajectories, vertical/root motion, body-relative motion and velocities are
preserved.  It is a prior-data conversion, not a Planner frame transform and
not a policy retargeter.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MOTION_PRIOR_CONTRACT_VERSION = "motion_prior_base_heading_frame0/v1"


class MotionPriorCanonicalError(ValueError):
    """Raised when source motion data cannot be safely canonicalised."""


def _finite_array(value: Any, name: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < len(shape_tail) or tuple(array.shape[-len(shape_tail) :]) != shape_tail:
        raise MotionPriorCanonicalError(
            f"{name} must end in shape {shape_tail}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise MotionPriorCanonicalError(f"{name} contains non-finite values")
    return array


def _normalize_quaternions_wxyz(quaternions: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-9):
        raise MotionPriorCanonicalError(f"{name} contains a zero quaternion")
    return quaternions / norms


def _yaw_from_quaternion_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _heading_quaternion_wxyz(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def _quaternion_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def _quaternion_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _world_to_heading_rotation(yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    # Transpose of R_world_from_heading.
    return np.asarray(
        ((cosine, sine, 0.0), (-sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _rotate(rotation: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", rotation, vectors)


def canonicalize_motion_arrays(
    arrays: Mapping[str, Any], *, root_body_index: int = 0
) -> dict[str, np.ndarray]:
    """Return one motion expressed in its frame-0 root yaw-heading frame."""

    body_pos_w = _finite_array(arrays["body_pos_w"], "body_pos_w", (3,))
    body_quat_w = _normalize_quaternions_wxyz(
        _finite_array(arrays["body_quat_w"], "body_quat_w", (4,)),
        "body_quat_w",
    )
    body_lin_vel_w = _finite_array(
        arrays["body_lin_vel_w"], "body_lin_vel_w", (3,)
    )
    body_ang_vel_w = _finite_array(
        arrays["body_ang_vel_w"], "body_ang_vel_w", (3,)
    )
    if body_pos_w.ndim != 3 or body_quat_w.shape[:2] != body_pos_w.shape[:2]:
        raise MotionPriorCanonicalError("body position/quaternion shapes do not agree")
    if body_lin_vel_w.shape != body_pos_w.shape or body_ang_vel_w.shape != body_pos_w.shape:
        raise MotionPriorCanonicalError("body velocity shapes do not agree with body positions")
    if not 0 <= root_body_index < body_pos_w.shape[1]:
        raise MotionPriorCanonicalError("root_body_index is outside the body array")
    # Joint arrays are [time, joint], so the time dimension is checked
    # explicitly rather than with _finite_array's trailing-shape helper.
    joint_pos = np.asarray(arrays["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(arrays["joint_vel"], dtype=np.float64)
    if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
        raise MotionPriorCanonicalError("joint position/velocity shapes do not agree")
    if joint_pos.shape[0] != body_pos_w.shape[0]:
        raise MotionPriorCanonicalError("joint and body trajectories have different frame counts")
    if not np.isfinite(joint_pos).all() or not np.isfinite(joint_vel).all():
        raise MotionPriorCanonicalError("joint trajectory contains non-finite values")
    fps = np.asarray(arrays["fps"])
    if fps.size != 1 or not np.isfinite(fps).all() or float(fps.reshape(-1)[0]) <= 0.0:
        raise MotionPriorCanonicalError("fps must be one positive finite scalar")

    root_anchor_position_w = body_pos_w[0, root_body_index].copy()
    root_anchor_quaternion_w = body_quat_w[0, root_body_index].copy()
    anchor_yaw = _yaw_from_quaternion_wxyz(root_anchor_quaternion_w)
    world_to_local = _world_to_heading_rotation(anchor_yaw)
    heading_quaternion = _heading_quaternion_wxyz(anchor_yaw)
    heading_inverse = _quaternion_conjugate_wxyz(heading_quaternion)

    body_pos_b0 = _rotate(world_to_local, body_pos_w - root_anchor_position_w)
    body_lin_vel_b0 = _rotate(world_to_local, body_lin_vel_w)
    body_ang_vel_b0 = _rotate(world_to_local, body_ang_vel_w)
    body_quat_b0 = _quaternion_multiply_wxyz(heading_inverse, body_quat_w)
    body_quat_b0 = _normalize_quaternions_wxyz(body_quat_b0, "body_quat_b0")

    result: dict[str, np.ndarray] = {
        "contract_version_utf8": np.frombuffer(
            MOTION_PRIOR_CONTRACT_VERSION.encode("utf-8"), dtype=np.uint8
        ),
        "fps": fps,
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_b0": body_pos_b0.astype(np.float32),
        "body_quat_b0_wxyz": body_quat_b0.astype(np.float32),
        "body_lin_vel_b0": body_lin_vel_b0.astype(np.float32),
        "body_ang_vel_b0": body_ang_vel_b0.astype(np.float32),
        "source_root_anchor_position_w": root_anchor_position_w.astype(np.float64),
        "source_root_anchor_quat_wxyz": root_anchor_quaternion_w.astype(np.float64),
        "source_root_anchor_yaw_rad": np.asarray([anchor_yaw], dtype=np.float64),
    }
    for optional in ("upper_momentum_pelvis", "upper_mass_kg", "upper_length_scale_m"):
        if optional in arrays:
            value = np.asarray(arrays[optional])
            if not np.isfinite(value).all():
                raise MotionPriorCanonicalError(f"{optional} contains non-finite values")
            result[optional] = value
    return result


def canonicalize_strike_target(
    strike_target: Mapping[str, Any],
    *,
    root_anchor_position_w: np.ndarray,
    root_anchor_yaw_rad: float,
) -> dict[str, Any]:
    """Transform manifest strike/ball metadata into the same local frame."""

    rotation = _world_to_heading_rotation(root_anchor_yaw_rad)

    def position(name: str) -> list[float] | None:
        if name not in strike_target:
            return None
        vector = _finite_array(strike_target[name], name, (3,))
        return _rotate(rotation, vector - root_anchor_position_w).tolist()

    def vector(name: str, *, normalize: bool = False) -> list[float] | None:
        if name not in strike_target:
            return None
        value = _finite_array(strike_target[name], name, (3,))
        transformed = _rotate(rotation, value)
        if normalize:
            length = float(np.linalg.norm(transformed))
            if length < 1.0e-9:
                raise MotionPriorCanonicalError(f"{name} cannot be normalized")
            transformed = transformed / length
        return transformed.tolist()

    output = {
        "racket_position_b0_m": position("racket_position_m"),
        "racket_velocity_b0_mps": vector("racket_velocity_mps"),
        "racket_normal_b0": vector("racket_normal_w", normalize=True),
        "racket_tangent_b0": vector("racket_tangent_w", normalize=True),
        "ball_position_b0_m": position("ball_position_m"),
        "ball_in_velocity_b0_mps": vector("ball_in_velocity_mps"),
        "ball_out_velocity_b0_mps": vector("ball_out_velocity_mps"),
    }
    return {key: value for key, value in output.items() if value is not None}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_motion_package(manifest_path: Path, output_dir: Path) -> Path:
    """Create a new canonical package without modifying any source artifact."""

    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("motions", [])
    if not entries:
        raise MotionPriorCanonicalError("source manifest contains no motions")
    output_motion_dir = output_dir / "motion_npz"
    output_motion_dir.mkdir(parents=True, exist_ok=True)
    output_entries = []
    for motion_id, entry in enumerate(entries):
        source_path = Path(entry["motion_npz"]).expanduser()
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve()
        with np.load(source_path, allow_pickle=False) as source:
            canonical = canonicalize_motion_arrays(source)
        filename = f"motion_{motion_id:02d}_{source_path.stem}.npz"
        output_path = output_motion_dir / filename
        np.savez_compressed(output_path, **canonical)
        root_position = canonical["source_root_anchor_position_w"]
        root_yaw = float(canonical["source_root_anchor_yaw_rad"][0])
        canonical_target = canonicalize_strike_target(
            entry.get("strike_target", {}),
            root_anchor_position_w=root_position,
            root_anchor_yaw_rad=root_yaw,
        )
        output_entries.append(
            {
                "motion_id": motion_id,
                "episode_id": entry.get("episode_id", str(motion_id)),
                "stroke_type": entry.get("stroke_type", "unknown"),
                "source_motion_npz": str(source_path),
                "source_motion_sha256": sha256_file(source_path),
                "canonical_motion_npz": str(output_path.relative_to(output_dir)),
                "canonical_motion_sha256": sha256_file(output_path),
                "fps": int(np.asarray(canonical["fps"]).reshape(-1)[0]),
                "frames": int(canonical["joint_pos"].shape[0]),
                "hit_frame": int(entry.get("hit_event", {}).get("motion_hit_frame", -1)),
                "source_root_anchor_position_w": root_position.tolist(),
                "source_root_anchor_quat_wxyz": canonical[
                    "source_root_anchor_quat_wxyz"
                ].tolist(),
                "source_root_anchor_yaw_rad": root_yaw,
                "strike_target_b0": canonical_target,
            }
        )

    output_manifest = {
        "contract_version": MOTION_PRIOR_CONTRACT_VERSION,
        "qualification": "coordinate_canonicalization_only",
        "training_approved": False,
        "qualification_note": (
            "Canonicalization proves rigid-frame equivalence only; formal-scene joint margins, "
            "collisions, dynamic tracking, balance and contact remain separate gates."
        ),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_manifest_status": manifest.get("status"),
        "source_manifest_training_approved": manifest.get("v27_bent_ready_contract", {}).get(
            "training_approved"
        ),
        "root_body_index": 0,
        "frame_definition": (
            "translation origin and yaw are frozen from root body at motion frame 0; "
            "Z remains metric relative to that root origin"
        ),
        "source_artifacts_modified": False,
        "motion_count": len(output_entries),
        "motions": output_entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_dir / "manifest.json"
    temporary = output_manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(output_manifest_path)
    return output_manifest_path
