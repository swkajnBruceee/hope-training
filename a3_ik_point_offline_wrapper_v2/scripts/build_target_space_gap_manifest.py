#!/usr/bin/env python3
"""Build deterministic joint target-space gap candidates for IK generation.

The candidates stay inside the empirically observed envelopes, but combine
position, velocity, normal orientation, and strike time in new tuples.  They
are deliberately only IK inputs; no candidate is training-approved here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ENVELOPES = {
    "backhand": {
        "position": ((0.382, 0.562), (-0.535, 0.337), (-0.005, 0.168)),
        "velocity": ((0.832, 1.350), (-0.340, 0.070), (0.345, 1.315)),
        "normal_pitch_deg": (-27.0, 3.0),
        "normal_yaw_deg": (-11.0, 20.0),
        "time": (0.58, 2.15),
    },
    "forehand": {
        "position": ((0.383, 0.527), (-0.785, -0.565), (0.064, 0.197)),
        "velocity": ((1.325, 1.665), (-0.095, 0.095), (0.365, 0.595)),
        "normal_pitch_deg": (10.5, 29.5),
        "normal_yaw_deg": (-10.0, 10.0),
        "time": (1.22, 2.15),
    },
}


def _normal(stroke: str, pitch_deg: float, yaw_deg: float) -> list[float]:
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    sign = -1.0 if stroke == "backhand" else 1.0
    value = [sign * math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    norm = math.sqrt(sum(x * x for x in value))
    return [x / norm for x in value]


def _scaled(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    values = (np.arange(size, dtype=np.float64) + rng.random(size)) / size
    rng.shuffle(values)
    return low + (high - low) * values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-stroke", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.per_stroke < 10:
        raise SystemExit("--per-stroke must be at least 10")

    root = args.output_root.expanduser().resolve()
    goals_dir = root / "goals"
    goals_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    goals: list[dict] = []
    for stroke_index, stroke in enumerate(("backhand", "forehand")):
        spec = ENVELOPES[stroke]
        count = args.per_stroke
        px = _scaled(rng, *spec["position"][0], count)
        py = _scaled(rng, *spec["position"][1], count)
        pz = _scaled(rng, *spec["position"][2], count)
        vx = _scaled(rng, *spec["velocity"][0], count)
        vy = _scaled(rng, *spec["velocity"][1], count)
        vz = _scaled(rng, *spec["velocity"][2], count)
        pitch = _scaled(rng, *spec["normal_pitch_deg"], count)
        yaw = _scaled(rng, *spec["normal_yaw_deg"], count)
        strike_time = _scaled(rng, *spec["time"], count)
        for index in range(count):
            split = "validation" if index % 5 == 0 else "training"
            goal_id = f"gap_{stroke[:2]}_{index:03d}"
            normal = _normal(stroke, float(pitch[index]), float(yaw[index]))
            item = {
                "goal_id": goal_id,
                "goal_path": f"goals/{goal_id}.yaml",
                "swing_type": stroke,
                "split": split,
                "sequence": 500000 + stroke_index * 10000 + index,
                "position_m": [float(px[index]), float(py[index]), float(pz[index])],
                "linear_velocity_mps": [float(vx[index]), float(vy[index]), float(vz[index])],
                "racket_normal": normal,
                "pitch_deg": float(pitch[index]),
                "yaw_deg": float(yaw[index]),
                "time_to_strike_s": float(strike_time[index]),
                "generation_role": "joint_target_space_gap_candidate",
            }
            goals.append(item)
            goals_dir.joinpath(f"{goal_id}.yaml").write_text(
                "\n".join([
                    "schema_version: a3_canonical_strike_goal/v1",
                    f"goal_id: {goal_id}",
                    "frame: initial_base_heading",
                    f"swing_type: {stroke}",
                    "position_m: [" + ", ".join(f"{x:.8f}" for x in item["position_m"]) + "]",
                    "linear_velocity_mps: [" + ", ".join(f"{x:.8f}" for x in item["linear_velocity_mps"]) + "]",
                    "racket_normal: [" + ", ".join(f"{x:.10f}" for x in normal) + "]",
                    f"time_to_strike_s: {item['time_to_strike_s']:.6f}",
                    f"sequence: {item['sequence']}",
                    "",
                ]),
                encoding="utf-8",
            )

    manifest = {
        "schema_version": "a3_target_space_gap_goals/v1",
        "status": "raw_ik_generation_pending",
        "coordinate_contract": "current_root_relative_initial_heading",
        "goal_fields": ["position_m", "linear_velocity_mps", "racket_normal", "time_to_strike_s"],
        "synchronization_contract": "position, velocity, normal, hit frame, joint velocity, and strike time are one target tuple",
        "split_contract": "explicit_target_tuple_holdout",
        "generation_note": "Deterministic Latin-hypercube tuples inside observed FH/BH envelopes; IK and all audits remain pending.",
        "waist_contract": {
            "waist_pitch": "forward_only_nonnegative_joint_pitch",
            "backward_tilt_allowed": False,
            "forward_tilt_limit_deg": 20.0,
            "waist_roll_abs_limit_deg": 20.0,
        },
        "seed": args.seed,
        "per_stroke": args.per_stroke,
        "split_counts": {
            "training": sum(x["split"] == "training" for x in goals),
            "validation": sum(x["split"] == "validation" for x in goals),
        },
        "goals": goals,
    }
    output = root / "gap_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "count": len(goals), **manifest["split_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
