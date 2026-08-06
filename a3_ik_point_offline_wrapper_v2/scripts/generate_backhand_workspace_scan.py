#!/usr/bin/env python3
"""Create an explicit single-stroke workspace scan manifest.

The scan keeps the stroke label explicit and samples position plus racket
normal pitch. It does not run the generator; use the generated manifest with
generate_dual_stroke_dataset.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def linspace(lo: float, hi: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("grid counts must be at least 2")
    step = (hi - lo) / (count - 1)
    return [lo + i * step for i in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--swing-type", choices=("forehand", "backhand"), default="backhand")
    parser.add_argument("--x-count", type=int, default=5)
    parser.add_argument("--y-count", type=int, default=7)
    parser.add_argument("--z-count", type=int, default=4)
    parser.add_argument("--pitch-deg", type=float, nargs="+")
    parser.add_argument("--yaw-deg", type=float, nargs="+")
    parser.add_argument("--velocity", type=float, nargs=3)
    parser.add_argument("--time-to-strike-s", type=float, default=1.20)
    args = parser.parse_args()

    if args.swing_type == "forehand":
        bounds = ((0.38, 0.53), (-0.80, -0.56), (0.06, 0.20))
        default_pitch = [10.0, 20.0, 30.0]
        default_velocity = [1.50, 0.00, 0.48]
    else:
        bounds = ((0.38, 0.51), (-0.55, 0.14), (-0.01, 0.16))
        default_pitch = [-22.0, -10.0, 2.0]
        default_velocity = [1.35, -0.04, 0.42]
    pitch_values = args.pitch_deg if args.pitch_deg is not None else default_pitch
    yaw_values = args.yaw_deg if args.yaw_deg is not None else [0.0]
    velocity = args.velocity if args.velocity is not None else default_velocity
    xs = linspace(*bounds[0], args.x_count)
    ys = linspace(*bounds[1], args.y_count)
    zs = linspace(*bounds[2], args.z_count)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    goals: list[dict[str, str]] = []
    sequence = 300000
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            for zi, z in enumerate(zs):
                for pi, pitch_deg in enumerate(pitch_values):
                    for ai, yaw_deg in enumerate(yaw_values):
                        pitch = math.radians(pitch_deg)
                        yaw = math.radians(yaw_deg)
                        goal_id = f"{args.swing_type}_scan_x{xi:02d}_y{yi:02d}_z{zi:02d}_p{pi:02d}_a{ai:02d}"
                        goal_path = args.output_root / "goals" / f"{goal_id}.yaml"
                        goal_path.parent.mkdir(parents=True, exist_ok=True)
                        base_x = math.cos(pitch) if args.swing_type == "forehand" else -math.cos(pitch)
                        base_y = (
                            math.cos(pitch) * math.sin(yaw)
                            if args.swing_type == "forehand"
                            else -math.cos(pitch) * math.sin(yaw)
                        )
                        normal = [base_x * math.cos(yaw), base_y, math.sin(pitch)]
                        goal_path.write_text(
                            "\n".join(
                                [
                                    "schema_version: a3_canonical_strike_goal/v1",
                                    f"goal_id: {goal_id}",
                                    "frame: initial_base_heading",
                                    f"swing_type: {args.swing_type}",
                                    f"position_m: [{x:.8f}, {y:.8f}, {z:.8f}]",
                                    "linear_velocity_mps: "
                                    f"[{velocity[0]:.8f}, {velocity[1]:.8f}, {velocity[2]:.8f}]",
                                    "racket_normal: "
                                    f"[{normal[0]:.10f}, {normal[1]:.10f}, {normal[2]:.10f}]",
                                    f"time_to_strike_s: {args.time_to_strike_s:.4f}",
                                    f"sequence: {sequence}",
                                    "",
                                ]
                            ),
                            encoding="utf-8",
                        )
                        goals.append({"goal_path": str(goal_path.relative_to(args.manifest.parent)), "swing_type": args.swing_type})
                        sequence += 1

    args.manifest.write_text(
        json.dumps(
            {
                "schema_version": "a3_dual_stroke_generation_manifest/v1",
                "scan_type": f"{args.swing_type}_position_and_normal_pitch_grid",
                "position_bounds_b_m": {
                    "min": [bounds[0][0], bounds[1][0], bounds[2][0]],
                    "max": [bounds[0][1], bounds[1][1], bounds[2][1]],
                },
                "normal_pitch_deg": pitch_values,
                "normal_yaw_deg": yaw_values,
                "linear_velocity_b_mps": velocity,
                "goals": goals,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"generated_goals={len(goals)}")
    print(f"manifest={args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
