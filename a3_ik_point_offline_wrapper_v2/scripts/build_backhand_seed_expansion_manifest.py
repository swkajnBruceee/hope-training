#!/usr/bin/env python3
"""Build controlled backhand target variants from the current-contract 96 bank."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np


def _rotate(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return c * vector + s * np.cross(axis, vector) + (1.0 - c) * axis * np.dot(axis, vector)


def _goal_yaml(goal: dict) -> str:
    pos = ", ".join(f"{float(x):.8f}" for x in goal["position_m"])
    vel = ", ".join(f"{float(x):.8f}" for x in goal["linear_velocity_mps"])
    normal = ", ".join(f"{float(x):.10f}" for x in goal["racket_normal"])
    return "\n".join(
        [
            "schema_version: a3_canonical_strike_goal/v1",
            f"goal_id: {goal['goal_id']}",
            "frame: initial_base_heading",
            "swing_type: backhand",
            f"position_m: [{pos}]",
            f"linear_velocity_mps: [{vel}]",
            f"racket_normal: [{normal}]",
            f"time_to_strike_s: {float(goal['time_to_strike_s']):.4f}",
            f"sequence: {int(goal['sequence'])}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants-per-seed", type=int, default=8)
    args = parser.parse_args()
    if args.variants_per_seed != 8:
        raise ValueError("the controlled expansion contract currently defines exactly 8 variants per seed")

    seed_path = args.seed_manifest.expanduser().resolve()
    source = json.loads(seed_path.read_text(encoding="utf-8"))
    seeds = source.get("motions", [])
    if len(seeds) != 96:
        raise ValueError(f"expected the fully admitted 96-motion seed bank, got {len(seeds)}")

    output_root = args.output_root.expanduser().resolve()
    goals_dir = output_root / "goals"
    goals_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        # Position offsets are deliberately small; the original seed remains
        # the anchor and the target tuple is changed as one synchronized unit.
        {"dp": (-0.008, -0.012, -0.004), "dv": (0.00, 0.00, 0.00), "normal_axis": (0.0, 0.0, 1.0), "normal_deg": -4.0, "time": 0.55},
        {"dp": (0.008, -0.012, 0.004), "dv": (0.00, 0.00, 0.00), "normal_axis": (0.0, 0.0, 1.0), "normal_deg": 4.0, "time": 0.60},
        {"dp": (-0.008, 0.012, 0.004), "dv": (0.00, 0.00, 0.00), "normal_axis": (0.0, 1.0, 0.0), "normal_deg": 3.0, "time": 0.65},
        {"dp": (0.008, 0.012, -0.004), "dv": (0.00, 0.00, 0.00), "normal_axis": (0.0, 1.0, 0.0), "normal_deg": -3.0, "time": 0.70},
        {"dp": (-0.004, 0.000, 0.006), "dv": (0.04, 0.04, 0.06), "normal_axis": (0.0, 1.0, 0.0), "normal_deg": 3.0, "time": 0.80},
        {"dp": (0.004, 0.000, -0.006), "dv": (-0.04, -0.04, -0.06), "normal_axis": (0.0, 1.0, 0.0), "normal_deg": -3.0, "time": 0.85},
        {"dp": (0.000, -0.006, 0.000), "dv": (0.00, 0.08, 0.00), "normal_axis": (0.0, 0.0, 1.0), "normal_deg": -2.0, "time": 0.90},
        {"dp": (0.000, 0.006, 0.000), "dv": (0.00, -0.08, 0.00), "normal_axis": (0.0, 0.0, 1.0), "normal_deg": 2.0, "time": 0.95},
    ]

    goals: list[dict] = []
    for seed_index, entry in enumerate(seeds):
        base = entry["canonical_goal_10d"]
        position = np.asarray(base["position_m"], dtype=np.float64)
        velocity = np.asarray(base["linear_velocity_mps"], dtype=np.float64)
        normal = np.asarray(base["normal_w"], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        for variant_index, variant in enumerate(variants):
            goal_id = f"legacy96_seed{seed_index:03d}_v{variant_index:02d}"
            target_position = position + np.asarray(variant["dp"], dtype=np.float64)
            target_velocity = velocity + np.asarray(variant["dv"], dtype=np.float64)
            target_normal = _rotate(normal, np.asarray(variant["normal_axis"], dtype=np.float64), variant["normal_deg"])
            target_normal /= np.linalg.norm(target_normal)
            split = "validation" if seed_index % 5 == 0 else "training"
            goal = {
                "goal_id": goal_id,
                "goal_path": f"goals/{goal_id}.yaml",
                "swing_type": "backhand",
                "split": split,
                "sequence": 400000 + seed_index * len(variants) + variant_index,
                "seed_episode_id": entry["episode_id"],
                "variant_index": variant_index,
                "position_m": target_position.tolist(),
                "linear_velocity_mps": target_velocity.tolist(),
                "racket_normal": target_normal.tolist(),
                "time_to_strike_s": variant["time"],
                "expansion_contract": {
                    "position_delta_m": list(variant["dp"]),
                    "velocity_delta_mps": list(variant["dv"]),
                    "normal_rotation_deg": variant["normal_deg"],
                    "time_to_strike_s": variant["time"],
                    "source_seed_manifest": str(seed_path),
                    "waist_contract": source.get("waist_contract"),
                },
            }
            (goals_dir / f"{goal_id}.yaml").write_text(_goal_yaml(goal), encoding="utf-8")
            goals.append(goal)

    manifest = {
        "schema_version": "a3_backhand_seed_expansion_goals/v1",
        "status": "raw_ik_generation_pending",
        "coordinate_contract": "current_root_relative_initial_heading",
        "goal_fields": ["position_m", "linear_velocity_mps", "racket_normal", "time_to_strike_s"],
        "synchronization_contract": "position, velocity, normal, hit frame, joint velocity, and strike time are one target tuple",
        "seed_manifest": str(seed_path),
        "seed_count": len(seeds),
        "variant_count_per_seed": len(variants),
        "goal_count": len(goals),
        "split_counts": {"training": sum(g["split"] == "training" for g in goals), "validation": sum(g["split"] == "validation" for g in goals)},
        "waist_contract": source.get("waist_contract"),
        "goals": goals,
    }
    output_path = output_root / "expansion_manifest.json"
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output_path), "count": len(goals), "training": manifest["split_counts"]["training"], "validation": manifest["split_counts"]["validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
