#!/usr/bin/env python3
"""Prepare, but never approve, one base-frame A3 strike target candidate.

Historical tracking manifests store task vectors in a table/world frame while
the standalone contract deliberately evaluates in the pelvis base frame.  This
tool makes that conversion auditable: it binds a source target JSON, a specific
motion entry, and that entry's declared prepositioned base pose into a review
packet.  Its output is *not* an immutable ``target_spec.json`` and is rejected
as a training or qualification input until a human reviews the packet and
creates a separate target input.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from a3_strike_contract import sha256_file


def _rotation_bw_from_yaw(yaw_rad: float) -> np.ndarray:
    """Return world-to-base rotation for a base with world yaw ``yaw_rad``."""

    if not math.isfinite(yaw_rad):
        raise ValueError("base yaw must be finite")
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.asarray([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _finite_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return vector


def derive_candidate(
    source_target_spec: dict[str, Any],
    source_manifest: dict[str, Any],
    episode_id: str,
    source_dataset: str,
    racket_mount_contract_id: str,
    source_target_path: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    """Return a review-only target packet from source evidence already on disk."""

    if not source_dataset:
        raise ValueError("source_dataset must be non-empty")
    if not racket_mount_contract_id:
        raise ValueError("racket_mount_contract_id must be non-empty")
    source_episode = str(source_target_spec.get("episode_id", ""))
    if source_episode != episode_id:
        raise ValueError(f"source target episode_id={source_episode!r} does not match {episode_id!r}")
    entries = [entry for entry in source_manifest.get("motions", []) if str(entry.get("episode_id")) == episode_id]
    if len(entries) != 1:
        raise ValueError(f"expected exactly one manifest motion for {episode_id!r}, found {len(entries)}")
    entry = entries[0]
    hit_target = source_target_spec.get("hit_target")
    if not isinstance(hit_target, dict):
        raise ValueError("source target is missing hit_target")
    stance = entry.get("stance_metadata")
    if not isinstance(stance, dict) or not isinstance(stance.get("base_pose_target_w"), dict):
        raise ValueError(
            "manifest entry is missing stance_metadata.base_pose_target_w; "
            "do not infer a base frame from a relabelled executor state"
        )
    base_pose = stance["base_pose_target_w"]
    base_position_w = _finite_vector(base_pose.get("position_m"), "base_pose_target_w.position_m")
    base_yaw = float(base_pose.get("yaw_rad"))
    rotation_bw = _rotation_bw_from_yaw(base_yaw)
    position_w = _finite_vector(hit_target.get("racket_position_m"), "hit_target.racket_position_m")
    velocity_w = _finite_vector(hit_target.get("racket_velocity_mps"), "hit_target.racket_velocity_mps")
    normal_w = _finite_vector(hit_target.get("racket_normal_w"), "hit_target.racket_normal_w")
    normal_w /= np.linalg.norm(normal_w)
    hit_event = entry.get("hit_event")
    if not isinstance(hit_event, dict):
        raise ValueError("manifest entry is missing hit_event")
    hit_time_s = float(hit_event.get("hit_time_from_start_s"))
    if not math.isfinite(hit_time_s) or hit_time_s < 0.0:
        raise ValueError("hit_event.hit_time_from_start_s must be finite and non-negative")

    source_frame = str(source_target_spec.get("coordinate_contract", {}).get("position_frame", "unknown"))
    proposed_input = {
        "schema_version": 1,
        "source_dataset": source_dataset,
        "source_episode_id": episode_id,
        "stroke_type": str(entry.get("stroke_type", "")).lower(),
        "hit_time_s": hit_time_s,
        "racket_position_b_m": (rotation_bw @ (position_w - base_position_w)).tolist(),
        "racket_velocity_b_mps": (rotation_bw @ velocity_w).tolist(),
        "racket_normal_b": (rotation_bw @ normal_w).tolist(),
        "racket_mount_contract_id": racket_mount_contract_id,
        "racket_position_w_m": position_w.tolist(),
        "racket_velocity_w_mps": velocity_w.tolist(),
        "racket_normal_w": normal_w.tolist(),
        "source_frame": source_frame,
        "normal_semantics": "source_racket_normal_w_rotated_into_declared_prepositioned_base",
        "hit_time_interpolation": {
            "source_hit_index": hit_event.get("source_hit_index"),
            "source_fps": hit_event.get("source_fps"),
            "motion_hit_frame": hit_event.get("motion_hit_frame"),
            "motion_fps": hit_event.get("motion_fps"),
        },
    }
    motion_path = Path(str(entry.get("motion_npz", ""))).expanduser()
    native_calibration = entry.get("native_calibration")
    native_relabelled = isinstance(native_calibration, dict) and isinstance(native_calibration.get("original_strike_target"), dict)
    return {
        "candidate_schema_version": 1,
        "status": "requires_human_review_not_an_immutable_target",
        "proposed_target_input": proposed_input,
        "source_evidence": {
            "source_target_spec": str(source_target_path),
            "source_target_spec_sha256": sha256_file(source_target_path),
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_motion_npz": str(motion_path),
            "source_motion_npz_sha256": sha256_file(motion_path) if motion_path.is_file() else None,
            "historical_target_status": (
                "native_executor_relabel_present_original_target_recovered"
                if native_relabelled else entry.get("actuator_aware_pilot", {}).get("status", "not_declared")
            ),
            "native_calibration_target_source": native_calibration.get("target_source") if native_relabelled else None,
        },
        "declared_base_transform": {
            "base_position_w_m": base_position_w.tolist(),
            "base_yaw_rad": base_yaw,
            "world_to_base_rotation": rotation_bw.tolist(),
        },
        "review_required": [
            "Confirm the source episode and hit event represent the intended pilot task.",
            "Confirm the declared prepositioned base pose matches the standalone replay initial condition.",
            "Confirm racket_mount_contract_id refers to the official right_racket site local +Y/red-face convention.",
            "Confirm the command trajectory's timestamp zero is the same pre-hit origin used by hit_time_s.",
            "Only then copy proposed_target_input into a separate reviewed JSON and run create_a3_target_spec.py.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-target-spec", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--racket-mount-contract-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target_path = args.source_target_spec.expanduser().resolve()
    manifest_path = args.source_manifest.expanduser().resolve()
    candidate = derive_candidate(
        json.loads(target_path.read_text(encoding="utf-8")),
        json.loads(manifest_path.read_text(encoding="utf-8")),
        args.episode_id,
        args.source_dataset,
        args.racket_mount_contract_id,
        target_path,
        manifest_path,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
