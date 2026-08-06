"""Place a canonical motion prior into one explicitly versioned scene frame.

This module performs one rigid transform only.  It does not alter joint
positions, joint velocities, timing, or the motion's internal geometry.  The
result uses the legacy ``body_*_w`` NPZ field names so an existing policy can
be evaluated without adding a new runtime transform to its command path.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from training.utils.motion_prior_canonical import (
    MOTION_PRIOR_CONTRACT_VERSION,
    MotionPriorCanonicalError,
    sha256_file,
)


SCENE_PLACEMENT_CONTRACT_VERSION = "canonical_prior_scene_placement/v1"


def _finite_vector(value: Any, name: str, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise MotionPriorCanonicalError(
            f"{name} must be one finite vector with shape {(size,)}, got {vector.shape}"
        )
    return vector


def _heading_rotation(yaw_rad: float) -> np.ndarray:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _heading_quaternion_wxyz(yaw_rad: float) -> np.ndarray:
    return np.asarray(
        (math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)),
        dtype=np.float64,
    )


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


def _rotate(rotation: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", rotation, vectors)


def place_canonical_motion_arrays(
    canonical: Mapping[str, Any],
    *,
    root_anchor_w_m: Any,
    root_heading_w_rad: float,
) -> dict[str, np.ndarray]:
    """Materialize canonical arrays as legacy world-frame motion arrays."""

    anchor = _finite_vector(root_anchor_w_m, "root_anchor_w_m", 3)
    yaw = float(root_heading_w_rad)
    if not math.isfinite(yaw):
        raise MotionPriorCanonicalError("root_heading_w_rad must be finite")
    try:
        version = bytes(np.asarray(canonical["contract_version_utf8"], dtype=np.uint8)).decode(
            "utf-8"
        )
    except (KeyError, UnicodeDecodeError, ValueError) as error:
        raise MotionPriorCanonicalError("canonical NPZ has no valid contract version") from error
    if version != MOTION_PRIOR_CONTRACT_VERSION:
        raise MotionPriorCanonicalError(
            f"canonical NPZ contract is {version!r}, expected {MOTION_PRIOR_CONTRACT_VERSION!r}"
        )

    rotation = _heading_rotation(yaw)
    heading_quaternion = _heading_quaternion_wxyz(yaw)
    body_pos_b0 = np.asarray(canonical["body_pos_b0"], dtype=np.float64)
    body_quat_b0 = np.asarray(canonical["body_quat_b0_wxyz"], dtype=np.float64)
    body_lin_vel_b0 = np.asarray(canonical["body_lin_vel_b0"], dtype=np.float64)
    body_ang_vel_b0 = np.asarray(canonical["body_ang_vel_b0"], dtype=np.float64)
    if body_pos_b0.ndim != 3 or body_pos_b0.shape[-1] != 3:
        raise MotionPriorCanonicalError("body_pos_b0 must have shape [time, body, 3]")
    if body_quat_b0.shape != (*body_pos_b0.shape[:-1], 4):
        raise MotionPriorCanonicalError("body quaternion shape does not match body positions")
    if body_lin_vel_b0.shape != body_pos_b0.shape or body_ang_vel_b0.shape != body_pos_b0.shape:
        raise MotionPriorCanonicalError("body velocity shape does not match body positions")
    for name, value in (
        ("body_pos_b0", body_pos_b0),
        ("body_quat_b0_wxyz", body_quat_b0),
        ("body_lin_vel_b0", body_lin_vel_b0),
        ("body_ang_vel_b0", body_ang_vel_b0),
    ):
        if not np.isfinite(value).all():
            raise MotionPriorCanonicalError(f"{name} contains non-finite values")

    body_quat_w = _quaternion_multiply_wxyz(heading_quaternion, body_quat_b0)
    quaternion_norm = np.linalg.norm(body_quat_w, axis=-1, keepdims=True)
    if np.any(quaternion_norm < 1.0e-9):
        raise MotionPriorCanonicalError("placed body quaternion contains a zero quaternion")
    body_quat_w /= quaternion_norm

    result = {
        "fps": np.asarray(canonical["fps"]).copy(),
        "joint_pos": np.asarray(canonical["joint_pos"], dtype=np.float32).copy(),
        "joint_vel": np.asarray(canonical["joint_vel"], dtype=np.float32).copy(),
        "body_pos_w": (_rotate(rotation, body_pos_b0) + anchor).astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_lin_vel_w": _rotate(rotation, body_lin_vel_b0).astype(np.float32),
        "body_ang_vel_w": _rotate(rotation, body_ang_vel_b0).astype(np.float32),
        "scene_placement_contract_utf8": np.frombuffer(
            SCENE_PLACEMENT_CONTRACT_VERSION.encode("utf-8"), dtype=np.uint8
        ),
        "scene_root_anchor_w_m": anchor.astype(np.float64),
        "scene_root_heading_w_rad": np.asarray((yaw,), dtype=np.float64),
    }
    if result["joint_pos"].ndim != 2 or result["joint_vel"].shape != result["joint_pos"].shape:
        raise MotionPriorCanonicalError("joint position/velocity arrays are incompatible")
    if result["joint_pos"].shape[0] != body_pos_b0.shape[0]:
        raise MotionPriorCanonicalError("joint and body trajectories have different lengths")
    if not np.isfinite(result["joint_pos"]).all() or not np.isfinite(result["joint_vel"]).all():
        raise MotionPriorCanonicalError("joint trajectory contains non-finite values")
    for optional in ("upper_momentum_pelvis", "upper_mass_kg", "upper_length_scale_m"):
        if optional in canonical:
            result[optional] = np.asarray(canonical[optional]).copy()
    return result


_POSITION_FIELDS = {
    "racket_position_b0_m": "racket_position_m",
    "ball_position_b0_m": "ball_position_m",
}
_VECTOR_FIELDS = {
    "racket_velocity_b0_mps": "racket_velocity_mps",
    "racket_normal_b0": "racket_normal_w",
    "racket_tangent_b0": "racket_tangent_w",
    "ball_in_velocity_b0_mps": "ball_in_velocity_mps",
    "ball_out_velocity_b0_mps": "ball_out_velocity_mps",
}


def place_canonical_strike_target(
    strike_target_b0: Mapping[str, Any],
    *,
    root_anchor_w_m: Any,
    root_heading_w_rad: float,
    source_strike_target: Mapping[str, Any] | None = None,
    source_root_heading_w_rad: float | None = None,
) -> dict[str, Any]:
    """Place strike metadata with the exact same rigid transform as the motion."""

    anchor = _finite_vector(root_anchor_w_m, "root_anchor_w_m", 3)
    yaw = float(root_heading_w_rad)
    if not math.isfinite(yaw):
        raise MotionPriorCanonicalError("root_heading_w_rad must be finite")
    rotation = _heading_rotation(yaw)
    output: dict[str, Any] = {}
    for source_name, output_name in _POSITION_FIELDS.items():
        if source_name in strike_target_b0:
            local = _finite_vector(strike_target_b0[source_name], source_name, 3)
            output[output_name] = (_rotate(rotation, local) + anchor).tolist()
    for source_name, output_name in _VECTOR_FIELDS.items():
        if source_name in strike_target_b0:
            local = _finite_vector(strike_target_b0[source_name], source_name, 3)
            placed = _rotate(rotation, local)
            if source_name in ("racket_normal_b0", "racket_tangent_b0"):
                norm = float(np.linalg.norm(placed))
                if norm < 1.0e-9:
                    raise MotionPriorCanonicalError(f"{source_name} cannot be normalized")
                placed /= norm
            output[output_name] = placed.tolist()

    source = dict(source_strike_target or {})
    for name, value in source.items():
        if name not in output and name not in {
            "racket_quat_xyzw",
            "racket_velocity_direction_w",
        }:
            output[name] = copy.deepcopy(value)
    if "racket_velocity_mps" in output:
        velocity = np.asarray(output["racket_velocity_mps"], dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        if speed > 1.0e-9:
            output["racket_velocity_direction_w"] = (velocity / speed).tolist()

    if "racket_quat_xyzw" in source:
        if source_root_heading_w_rad is None or not math.isfinite(
            float(source_root_heading_w_rad)
        ):
            raise MotionPriorCanonicalError(
                "source_root_heading_w_rad is required to place racket_quat_xyzw"
            )
        source_xyzw = _finite_vector(source["racket_quat_xyzw"], "racket_quat_xyzw", 4)
        source_wxyz = source_xyzw[[3, 0, 1, 2]]
        delta_yaw = yaw - float(source_root_heading_w_rad)
        placed_wxyz = _quaternion_multiply_wxyz(
            _heading_quaternion_wxyz(delta_yaw), source_wxyz
        )
        placed_wxyz /= np.linalg.norm(placed_wxyz)
        output["racket_quat_xyzw"] = placed_wxyz[[1, 2, 3, 0]].tolist()
    return output


def write_scene_placed_motion_package(
    canonical_manifest_path: Path,
    output_dir: Path,
    *,
    root_anchor_w_m: Any,
    root_heading_w_rad: float,
    scene_frame_version: str,
) -> Path:
    """Write a read-only evaluation package consumable by the legacy loader."""

    canonical_manifest_path = canonical_manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    if canonical_manifest.get("contract_version") != MOTION_PRIOR_CONTRACT_VERSION:
        raise MotionPriorCanonicalError("input manifest is not the canonical prior contract")
    canonical_entries = list(canonical_manifest.get("motions", []))
    if not canonical_entries:
        raise MotionPriorCanonicalError("canonical manifest contains no motions")
    source_manifest_path = Path(canonical_manifest["source_manifest"]).expanduser().resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_by_episode = {
        str(entry.get("episode_id")): entry for entry in source_manifest.get("motions", [])
    }
    anchor = _finite_vector(root_anchor_w_m, "root_anchor_w_m", 3)
    scene_frame_version = str(scene_frame_version).strip()
    if not scene_frame_version:
        raise MotionPriorCanonicalError("scene_frame_version cannot be empty")

    motion_dir = output_dir / "motion_npz"
    motion_dir.mkdir(parents=True, exist_ok=True)
    placed_entries: list[dict[str, Any]] = []
    for canonical_entry in canonical_entries:
        episode_id = str(canonical_entry.get("episode_id"))
        if episode_id not in source_by_episode:
            raise MotionPriorCanonicalError(f"source manifest is missing episode {episode_id}")
        source_entry = source_by_episode[episode_id]
        canonical_npz = (canonical_manifest_path.parent / canonical_entry["canonical_motion_npz"]).resolve()
        with np.load(canonical_npz, allow_pickle=False) as data:
            placed = place_canonical_motion_arrays(
                data,
                root_anchor_w_m=anchor,
                root_heading_w_rad=root_heading_w_rad,
            )
        output_npz = motion_dir / f"motion_{int(canonical_entry['motion_id']):02d}_{episode_id}.npz"
        np.savez_compressed(output_npz, **placed)

        target = place_canonical_strike_target(
            canonical_entry.get("strike_target_b0", {}),
            root_anchor_w_m=anchor,
            root_heading_w_rad=root_heading_w_rad,
            source_strike_target=source_entry.get("strike_target", {}),
            source_root_heading_w_rad=float(canonical_entry["source_root_anchor_yaw_rad"]),
        )
        output_entry = copy.deepcopy(source_entry)
        output_entry["motion_id"] = int(canonical_entry["motion_id"])
        output_entry["source_motion_npz_before_canonicalization"] = canonical_entry[
            "source_motion_npz"
        ]
        output_entry["canonical_motion_npz"] = str(canonical_npz)
        output_entry["canonical_motion_sha256"] = sha256_file(canonical_npz)
        output_entry["strike_target_before_scene_placement"] = copy.deepcopy(
            source_entry.get("strike_target", {})
        )
        output_entry["strike_target"] = target
        output_entry["motion_npz"] = str(output_npz)
        output_entry["library_motion_npz"] = str(output_npz)
        output_entry["scene_placed_motion_sha256"] = sha256_file(output_npz)
        output_entry["scene_placement"] = {
            "contract_version": SCENE_PLACEMENT_CONTRACT_VERSION,
            "scene_frame_version": scene_frame_version,
            "root_anchor_w_m": anchor.tolist(),
            "root_heading_w_rad": float(root_heading_w_rad),
            "joint_trajectory_changed": False,
            "timing_changed": False,
        }
        # Preserve explicit P4B/P4C audit layers when the canonical input is a
        # deterministic repair candidate.  The legacy source entry does not
        # contain these fields and must not erase them during rigid placement.
        for optional in ("goal_state_layers", "p4b_repair", "repair_provenance"):
            if optional in canonical_entry:
                output_entry[optional] = copy.deepcopy(canonical_entry[optional])
        placed_entries.append(output_entry)

    output_manifest = copy.deepcopy(source_manifest)
    output_manifest["manifest_name"] = "p1_scene_placed_canonical_prior_qualification"
    output_manifest["status"] = "read_only_dynamic_qualification_only"
    output_manifest["training_role"] = "not_training_approved"
    output_manifest["motions"] = placed_entries
    output_manifest["scene_placement_contract"] = {
        "contract_version": SCENE_PLACEMENT_CONTRACT_VERSION,
        "canonical_contract_version": MOTION_PRIOR_CONTRACT_VERSION,
        "scene_frame_version": scene_frame_version,
        "canonical_manifest": str(canonical_manifest_path),
        "canonical_manifest_sha256": sha256_file(canonical_manifest_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "root_anchor_w_m": anchor.tolist(),
        "root_heading_w_rad": float(root_heading_w_rad),
        "rigid_transform_only": True,
        "joint_trajectory_changed": False,
        "timing_changed": False,
        "training_approved": False,
        "qualification_note": (
            "This package only makes the canonical prior consumable by the existing policy "
            "in the stated scene frame. Dynamic safety remains unqualified."
        ),
    }
    if "p4b_repair_contract" in canonical_manifest:
        output_manifest["p4b_repair_contract"] = copy.deepcopy(
            canonical_manifest["p4b_repair_contract"]
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_dir / "manifest.json"
    temporary = output_manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(output_manifest_path)
    return output_manifest_path
