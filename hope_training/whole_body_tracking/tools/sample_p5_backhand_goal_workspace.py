#!/usr/bin/env python3
"""Create the P5 position-only goal workspace before any teacher solving.

This is deliberately a sampler, not a trajectory generator.  It fixes the
canonical normal, velocity and nominal hit time from one manifest motion and
creates a structured position lattice around that target.  Dataset regions
are assigned *before* IK/trajectory optimization so a later random split
cannot turn millimetre-neighbours into a misleading generalization result.

Every emitted row is PENDING.  A row becomes a teacher only after the P5
offline and PhysX qualification stages; this tool never writes a control
anchor or actual execution state into a canonical goal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "eval_outputs/strike_goal_p4/p4d_motion3_canonical_candidate_v3/manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be one finite 3-vector")
    return vector


def _goal_from_motion(manifest: dict[str, Any], motion_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = [entry for entry in manifest["motions"] if int(entry["motion_id"]) == motion_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one motion_id={motion_id}, got {len(matches)}")
    entry = matches[0]
    label = entry["strike_target_b0"]
    position = _finite_vector(label["racket_position_b0_m"], "racket_position_b0_m")
    normal = _finite_vector(label["racket_normal_b0"], "racket_normal_b0")
    velocity = _finite_vector(label["racket_velocity_b0_mps"], "racket_velocity_b0_mps")
    normal /= np.linalg.norm(normal)
    layers = entry.get("goal_state_layers", {})
    planner = layers.get("canonical_planner_goal_ball_center_impact_v1", {})
    nominal_time = planner.get("nominal_motion_time_from_frame0_s")
    if nominal_time is None or float(nominal_time) <= 0.0:
        raise ValueError("seed manifest does not provide a positive nominal motion hit time")
    return (
        {
            "position_b0_m": position.tolist(),
            "normal_b0": normal.tolist(),
            "linear_velocity_b0_mps": velocity.tolist(),
            "time_to_hit_s": float(nominal_time),
        },
        {"motion_id": motion_id, "episode_id": str(entry["episode_id"])},
    )


def _region(ix: int, iy: int, iz: int, shape: tuple[int, int, int]) -> str:
    """Partition a lattice with a real contiguous, never-trained block."""
    nx, ny, nz = shape
    edge = min(ix, nx - 1 - ix, iy, ny - 1 - iy, iz, nz - 1 - iz)
    if edge == 0:
        return "boundary_holdout"
    # A positive-x / positive-y interior block is contiguous by construction.
    if ix >= (nx + 1) // 2 and iy >= (ny + 1) // 2:
        return "workspace_holdout"
    # A complete middle z slab tests interpolation across a withheld bridge.
    if nz >= 5 and iz == nz // 2:
        return "bridge_holdout"
    # Deterministic checkerboard split only inside the remaining trainable area.
    return "validation" if (ix + 2 * iy + 3 * iz) % 5 == 0 else "training"


def build_workspace(
    goal: dict[str, Any], half_range_m: np.ndarray, shape: tuple[int, int, int]
) -> list[dict[str, Any]]:
    if any(size < 3 for size in shape):
        raise ValueError("grid shape must be at least 3 in every dimension")
    if np.any(half_range_m <= 0.0) or not np.isfinite(half_range_m).all():
        raise ValueError("position half range must be finite and positive")
    centre = _finite_vector(goal["position_b0_m"], "goal.position_b0_m")
    axes = [np.linspace(-half_range_m[i], half_range_m[i], shape[i]) for i in range(3)]
    rows: list[dict[str, Any]] = []
    sample_id = 0
    for ix, dx in enumerate(axes[0]):
        for iy, dy in enumerate(axes[1]):
            for iz, dz in enumerate(axes[2]):
                rows.append(
                    {
                        "sample_id": f"p5_pos_{sample_id:05d}",
                        "split": _region(ix, iy, iz, shape),
                        "lattice_index": [ix, iy, iz],
                        "canonical_goal_10d": {
                            **goal,
                            "position_b0_m": (centre + np.array((dx, dy, dz))).tolist(),
                        },
                        "qualification": "PENDING_IK",
                        "teacher_quality": None,
                        "seed_attempts": [],
                    }
                )
                sample_id += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--motion-id", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-shape", type=int, nargs=3, default=(7, 7, 5))
    parser.add_argument("--position-half-range-m", type=float, nargs=3, default=(0.06, 0.06, 0.04))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    goal, seed = _goal_from_motion(manifest, args.motion_id)
    shape = tuple(int(value) for value in args.grid_shape)
    rows = build_workspace(goal, np.asarray(args.position_half_range_m, dtype=np.float64), shape)
    counts = {name: sum(row["split"] == name for row in rows) for name in (
        "training", "validation", "bridge_holdout", "workspace_holdout", "boundary_holdout"
    )}
    payload = {
        "schema_version": "p5_backhand_goal_workspace/v1",
        "purpose": "pre-solver position-only canonical-goal split",
        "training_approved": False,
        "teacher_data": False,
        "goal_contract": "canonical_goal_10d/v1",
        "style": "backhand_canonical_v1",
        "fixed_goal_fields": ["normal_b0", "linear_velocity_b0_mps", "time_to_hit_s"],
        "seed_source": seed,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": _sha256(args.manifest),
        "grid_shape": list(shape),
        "position_half_range_m": list(args.position_half_range_m),
        "split_counts": counts,
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "split_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
