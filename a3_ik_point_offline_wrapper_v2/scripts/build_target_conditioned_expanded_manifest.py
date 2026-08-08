#!/usr/bin/env python3
"""Build a deterministic target-space coverage manifest for the 10-D pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _normal(stroke: str, pitch_deg: float, yaw_deg: float) -> list[float]:
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    sign = -1.0 if stroke == "backhand" else 1.0
    values = [sign * math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def _make_goal(stroke: str, split: str, index: int, position: tuple[float, float, float]) -> dict:
    if stroke == "backhand":
        speed = (1.26 + 0.035 * (index % 4), -0.08 + 0.04 * ((index + 1) % 5), 0.34 + 0.04 * (index % 4))
        pitch = -7.0 - 2.0 * (index % 7)
        strike_time = 1.30 + 0.05 * (index % 6)
        sequence_base = 1000 if split == "training" else 1100
    else:
        speed = (1.38 + 0.06 * (index % 5), -0.08 + 0.04 * ((index + 2) % 5), 0.38 + 0.04 * (index % 5))
        pitch = 12.0 + 2.0 * (index % 7)
        strike_time = 1.25 + 0.05 * (index % 7)
        sequence_base = 2000 if split == "training" else 2100
    yaw = -8.0 + 4.0 * (index % 5)
    goal_id = f"{stroke[:2]}_{split[:4]}_{index:02d}"
    return {
        "goal_id": goal_id,
        "goal_path": f"goals/{goal_id}.yaml",
        "swing_type": stroke,
        "split": split,
        "sequence": sequence_base + index,
        "position_m": list(position),
        "linear_velocity_mps": list(speed),
        "racket_normal": _normal(stroke, pitch, yaw),
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "time_to_strike_s": strike_time,
    }


def _positions(stroke: str, split: str) -> list[tuple[float, float, float]]:
    if stroke == "backhand":
        if split == "training":
            return [
                (0.420, -0.400, 0.030), (0.450, -0.280, 0.070), (0.480, -0.150, 0.110),
                (0.500, 0.000, 0.140), (0.430, -0.350, 0.100), (0.460, -0.200, 0.140),
                (0.490, -0.050, 0.050), (0.500, 0.080, 0.120), (0.420, -0.180, 0.040),
                (0.440, 0.020, 0.080), (0.470, -0.380, 0.130), (0.490, -0.250, 0.090),
            ]
        return [
            (0.435, -0.320, 0.055), (0.465, -0.080, 0.085), (0.485, 0.040, 0.115),
            (0.425, -0.220, 0.120), (0.475, -0.300, 0.045), (0.495, -0.120, 0.150),
        ]
    if split == "training":
        return [
            (0.420, -0.750, 0.080), (0.450, -0.720, 0.120), (0.480, -0.680, 0.160),
            (0.500, -0.620, 0.180), (0.430, -0.760, 0.140), (0.460, -0.700, 0.090),
            (0.490, -0.640, 0.130), (0.515, -0.590, 0.170), (0.425, -0.660, 0.060),
            (0.455, -0.610, 0.110), (0.485, -0.740, 0.150), (0.505, -0.670, 0.100),
        ]
    return [
        (0.435, -0.735, 0.100), (0.465, -0.675, 0.145), (0.495, -0.605, 0.155),
        (0.425, -0.705, 0.075), (0.475, -0.745, 0.170), (0.505, -0.625, 0.125),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    goals_dir = root / "goals"
    goals_dir.mkdir(parents=True, exist_ok=True)
    goals = []
    for stroke in ("backhand", "forehand"):
        for split in ("training", "validation"):
            for index, position in enumerate(_positions(stroke, split)):
                item = _make_goal(stroke, split, index, position)
                goals.append(item)
                normal = ", ".join(f"{value:.10f}" for value in item["racket_normal"])
                position_text = ", ".join(f"{value:.8f}" for value in item["position_m"])
                velocity_text = ", ".join(f"{value:.8f}" for value in item["linear_velocity_mps"])
                (goals_dir / f"{item['goal_id']}.yaml").write_text(
                    "\n".join(
                        [
                            "schema_version: a3_canonical_strike_goal/v1",
                            f"goal_id: {item['goal_id']}",
                            "frame: initial_base_heading",
                            f"swing_type: {item['swing_type']}",
                            f"position_m: [{position_text}]",
                            f"linear_velocity_mps: [{velocity_text}]",
                            f"racket_normal: [{normal}]",
                            f"time_to_strike_s: {item['time_to_strike_s']:.3f}",
                            f"sequence: {item['sequence']}",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
    manifest = {
        "schema_version": "a3_target_conditioned_goal_expanded/v1",
        "status": "raw_ik_generation_pending",
        "coordinate_contract": "initial_base_heading/root-relative",
        "goal_fields": ["position_m", "linear_velocity_mps", "racket_normal", "time_to_strike_s"],
        "synchronization_contract": "position, velocity, normal, hit frame, joint velocity, and strike time are one target tuple",
        "split_contract": "explicit_target_tuple_holdout",
        "generation_note": "Expanded positions stay inside reviewed FH/BH envelopes; validation targets are never copied into training.",
        "goals": goals,
    }
    output = root / "expanded_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "count": len(goals), "training": sum(x["split"] == "training" for x in goals), "validation": sum(x["split"] == "validation" for x in goals)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
