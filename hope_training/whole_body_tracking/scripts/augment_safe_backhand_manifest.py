"""Create a conservative family of backhand references from two reviewed anchors.

This is an offline candidate generator.  It interpolates the complete pose
trajectory (not only the strike metadata) between two human-reviewed,
physically stable references, then recomputes translational velocities.  The
output is explicitly marked as *not yet training approved*; PhysX replay must
select the final subset before it is used for PPO.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


POSE_ARRAYS = (
    "joint_pos",
    "body_pos_b0",
    "body_pos_w",
    "body_quat_b0_wxyz",
    "body_quat_w",
)
VECTOR_ARRAYS = (
    "joint_vel",
    "body_lin_vel_b0",
    "body_lin_vel_w",
    "body_ang_vel_b0",
    "body_ang_vel_w",
)


def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * np.asarray(a, dtype=np.float64) + alpha * np.asarray(b, dtype=np.float64)


def _quat_lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64).copy()
    # Quaternions q and -q are identical.  Align signs before interpolation so
    # an equivalent pair never creates a long rotation through zero.
    dot = np.sum(qa * qb, axis=-1, keepdims=True)
    qb = np.where(dot < 0.0, -qb, qb)
    q = _lerp(qa, qb, alpha)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return (q / np.clip(norm, 1.0e-12, None)).astype(np.float32)


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(values.astype(np.float64), dt, axis=0, edge_order=1).astype(np.float32)


def _blend_target(a: dict, b: dict, key: str, alpha: float) -> None:
    if key in a and key in b and isinstance(a[key], list) and isinstance(b[key], list):
        a[key] = _lerp(np.asarray(a[key], dtype=np.float64), np.asarray(b[key], dtype=np.float64), alpha).tolist()


def _normalize_list(value: list[float]) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    return (array / max(norm, 1.0e-12)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-a", type=int, default=1, help="stable reviewed manifest index")
    parser.add_argument("--source-b", type=int, default=2, help="stable reviewed manifest index")
    parser.add_argument("--count", type=int, default=33)
    args = parser.parse_args()
    if args.count < 2:
        parser.error("count must be at least 2")

    source_manifest = args.source_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    motions = payload.get("motions", [])
    if not (0 <= args.source_a < len(motions) and 0 <= args.source_b < len(motions)):
        raise IndexError("source motion index outside manifest")

    a_entry = motions[args.source_a]
    b_entry = motions[args.source_b]
    a_npz = np.load(Path(a_entry["motion_npz"]), allow_pickle=False)
    b_npz = np.load(Path(b_entry["motion_npz"]), allow_pickle=False)
    if a_npz["joint_pos"].shape != b_npz["joint_pos"].shape:
        raise ValueError("source anchors must have identical trajectory shapes")
    fps = float(np.asarray(a_npz["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    alphas = np.linspace(0.0, 1.0, args.count)

    out = copy.deepcopy(payload)
    out["manifest_name"] = "p5d2_safe_backhand_augmented_v1"
    out["status"] = "candidate_only_physx_replay_required"
    out["training_role"] = "candidate_not_training_approved"
    out["augmentation_contract"] = {
        "method": "full_trajectory_quaternion_safe_interpolation",
        "source_manifest": str(source_manifest),
        "source_indices": [args.source_a, args.source_b],
        "source_episode_ids": [a_entry["episode_id"], b_entry["episode_id"]],
        "count": int(args.count),
        "alpha_range": [0.0, 1.0],
        "note": "Keep only candidates that pass PhysX stability and arm-speed gates.",
    }

    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, alpha in enumerate(alphas):
        alpha = float(alpha)
        data: dict[str, np.ndarray] = {}
        for key in a_npz.files:
            if key in ("fps", "hit_frame", "physics_qualified"):
                data[key] = np.asarray(a_npz[key]).copy()
            elif key in ("body_quat_b0_wxyz", "body_quat_w"):
                data[key] = _quat_lerp(a_npz[key], b_npz[key], alpha)
            elif key in POSE_ARRAYS or key in VECTOR_ARRAYS:
                data[key] = _lerp(a_npz[key], b_npz[key], alpha).astype(np.float32)
            else:
                data[key] = np.asarray(a_npz[key]).copy()

        # Recompute finite-difference velocities from the blended trajectory;
        # this avoids a hidden velocity jump at the strike point.
        data["joint_vel"] = _finite_difference(data["joint_pos"], dt)
        for pos_key, vel_key in (
            ("body_pos_b0", "body_lin_vel_b0"),
            ("body_pos_w", "body_lin_vel_w"),
        ):
            data[vel_key] = _finite_difference(data[pos_key], dt)

        filename = f"safe_interp_{rank:03d}_a{alpha:0.4f}.npz"
        npz_path = output_dir / filename
        np.savez_compressed(npz_path, **data)

        # Always interpolate metadata from A to B.  Choosing B as the base
        # entry at alpha>=0.5 would silently collapse the latter half of the
        # strike-point grid to one endpoint.
        entry = copy.deepcopy(a_entry)
        entry["episode_id"] = f"p5d2_safe_interp_{rank:03d}_a{alpha:0.4f}"
        entry["motion_npz"] = str(npz_path)
        # Candidate manifests use one authoritative payload.  Rewrite both
        # legacy aliases so the strict motion loader cannot accidentally
        # select the original anchor trajectory.
        entry["canonical_motion_npz"] = str(npz_path)
        entry["library_motion_npz"] = str(npz_path)
        entry["motion_npz_original"] = [str(a_entry["motion_npz"]), str(b_entry["motion_npz"])]
        entry["joint_pos_shape"] = list(data["joint_pos"].shape)
        entry["body_pos_w_shape"] = list(data["body_pos_w"].shape)
        entry["augmentation"] = {
            "alpha": alpha,
            "source_a_index": int(args.source_a),
            "source_b_index": int(args.source_b),
            "source_a_episode_id": a_entry["episode_id"],
            "source_b_episode_id": b_entry["episode_id"],
            "physx_replay_status": "pending",
        }
        _blend_target(entry.setdefault("strike_target", {}), b_entry.get("strike_target", {}), "racket_position_m", alpha)
        _blend_target(entry.setdefault("strike_target", {}), b_entry.get("strike_target", {}), "racket_velocity_mps", alpha)
        _blend_target(entry.setdefault("strike_target", {}), b_entry.get("strike_target", {}), "racket_normal_w", alpha)
        _blend_target(entry.setdefault("strike_target", {}), b_entry.get("strike_target", {}), "racket_tangent_w", alpha)
        for direction_key in ("racket_normal_w", "racket_tangent_w"):
            if direction_key in entry["strike_target"]:
                entry["strike_target"][direction_key] = _normalize_list(entry["strike_target"][direction_key])
        for goal_key in ("canonical_goal_position_b0_m", "canonical_goal_normal_b0", "canonical_goal_linear_velocity_b0_mps"):
            if goal_key in entry and goal_key in b_entry:
                entry[goal_key] = _lerp(np.asarray(entry[goal_key]), np.asarray(b_entry[goal_key]), alpha).tolist()
        for container_key in ("canonical_goal_10d", "p5d2_dataset"):
            container = entry.get(container_key)
            other = b_entry.get(container_key)
            if isinstance(container, dict) and isinstance(other, dict):
                nested = container.get("canonical_goal_10d") if container_key == "p5d2_dataset" else container
                other_nested = other.get("canonical_goal_10d") if container_key == "p5d2_dataset" else other
                if isinstance(nested, dict) and isinstance(other_nested, dict):
                    for goal_key in ("position_b0_m", "normal_b0", "linear_velocity_b0_mps"):
                        if goal_key in nested and goal_key in other_nested:
                            nested[goal_key] = _lerp(np.asarray(nested[goal_key]), np.asarray(other_nested[goal_key]), alpha).tolist()
        rows.append(entry)

    out["motions"] = rows
    out["motion_count"] = len(rows)
    out["selected_count"] = len(rows)
    output_manifest = output_dir / "p5d2_safe_backhand_augmented_manifest.json"
    output_manifest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[augment] wrote {output_manifest}")
    print(f"[augment] wrote {len(rows)} full-trajectory candidates from indices {args.source_a},{args.source_b}")


if __name__ == "__main__":
    main()
