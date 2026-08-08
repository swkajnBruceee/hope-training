#!/usr/bin/env python3
"""Build a small, deterministic 10-D target-conditioned pilot manifest.

The pilot deliberately changes position, velocity, racket normal, and strike
time together.  It is a raw IK-generation input only; generated trajectories
must still pass the existing FK/TCP and hard-limit checks before admission to
the training bank.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _normal(stroke: str, pitch_deg: float, yaw_deg: float) -> list[float]:
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    c = math.cos(pitch)
    sign = -1.0 if stroke == "backhand" else 1.0
    value = [sign * c * math.cos(yaw), c * math.sin(yaw), math.sin(pitch)]
    norm = math.sqrt(sum(x * x for x in value))
    return [x / norm for x in value]


def _goal(
    *,
    goal_id: str,
    stroke: str,
    split: str,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    pitch_deg: float,
    yaw_deg: float,
    strike_time_s: float,
    sequence: int,
) -> dict:
    return {
        "goal_id": goal_id,
        "goal_path": f"goals/{goal_id}.yaml",
        "swing_type": stroke,
        "split": split,
        "sequence": sequence,
        "position_m": list(position),
        "linear_velocity_mps": list(velocity),
        "racket_normal": _normal(stroke, pitch_deg, yaw_deg),
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "time_to_strike_s": strike_time_s,
    }


def _pilot_goals() -> list[dict]:
    # Training points are deliberately interior and varied. Validation points
    # are held out by target tuple, not by randomly splitting trajectory rows.
    return [
        _goal(goal_id="bh_train_near_fast", stroke="backhand", split="training", position=(0.405, -0.30, 0.040), velocity=(1.53, 0.06, 0.54), pitch_deg=-4.0, yaw_deg=5.0, strike_time_s=1.20, sequence=1001),
        _goal(goal_id="bh_train_mid_nominal", stroke="backhand", split="training", position=(0.460, -0.10, 0.100), velocity=(1.35, -0.04, 0.42), pitch_deg=-10.0, yaw_deg=0.0, strike_time_s=1.40, sequence=1002),
        _goal(goal_id="bh_train_far_slow", stroke="backhand", split="training", position=(0.495, 0.08, 0.140), velocity=(1.17, -0.14, 0.30), pitch_deg=-16.0, yaw_deg=-5.0, strike_time_s=1.60, sequence=1003),
        _goal(goal_id="bh_train_low_lateral", stroke="backhand", split="training", position=(0.430, -0.46, 0.005), velocity=(1.35, 0.06, 0.54), pitch_deg=-22.0, yaw_deg=10.0, strike_time_s=1.30, sequence=1004),
        _goal(goal_id="bh_val_unseen_1", stroke="backhand", split="validation", position=(0.438, -0.245, 0.075), velocity=(1.44, -0.10, 0.36), pitch_deg=-7.0, yaw_deg=7.0, strike_time_s=1.50, sequence=1101),
        _goal(goal_id="bh_val_unseen_2", stroke="backhand", split="validation", position=(0.478, 0.045, 0.120), velocity=(1.26, 0.02, 0.48), pitch_deg=-19.0, yaw_deg=-8.0, strike_time_s=1.25, sequence=1102),
        _goal(goal_id="fh_train_near_fast", stroke="forehand", split="training", position=(0.405, -0.740, 0.080), velocity=(1.68, 0.10, 0.60), pitch_deg=26.0, yaw_deg=5.0, strike_time_s=1.20, sequence=2001),
        _goal(goal_id="fh_train_mid_nominal", stroke="forehand", split="training", position=(0.470, -0.650, 0.140), velocity=(1.50, 0.00, 0.48), pitch_deg=20.0, yaw_deg=0.0, strike_time_s=1.40, sequence=2002),
        _goal(goal_id="fh_train_far_slow", stroke="forehand", split="training", position=(0.515, -0.580, 0.180), velocity=(1.32, -0.10, 0.36), pitch_deg=14.0, yaw_deg=-5.0, strike_time_s=1.60, sequence=2003),
        _goal(goal_id="fh_train_low_lateral", stroke="forehand", split="training", position=(0.430, -0.780, 0.060), velocity=(1.50, 0.10, 0.60), pitch_deg=10.0, yaw_deg=10.0, strike_time_s=1.30, sequence=2004),
        _goal(goal_id="fh_val_unseen_1", stroke="forehand", split="validation", position=(0.445, -0.705, 0.110), velocity=(1.59, -0.06, 0.42), pitch_deg=23.0, yaw_deg=7.0, strike_time_s=1.50, sequence=2101),
        _goal(goal_id="fh_val_unseen_2", stroke="forehand", split="validation", position=(0.495, -0.615, 0.160), velocity=(1.41, 0.06, 0.54), pitch_deg=17.0, yaw_deg=-8.0, strike_time_s=1.25, sequence=2102),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    goals_dir = output_root / "goals"
    goals_dir.mkdir(parents=True, exist_ok=True)

    goals = _pilot_goals()
    for item in goals:
        goal_path = goals_dir / f"{item['goal_id']}.yaml"
        normal = ", ".join(f"{x:.10f}" for x in item["racket_normal"])
        position = ", ".join(f"{x:.8f}" for x in item["position_m"])
        velocity = ", ".join(f"{x:.8f}" for x in item["linear_velocity_mps"])
        goal_path.write_text(
            "\n".join(
                [
                    "schema_version: a3_canonical_strike_goal/v1",
                    f"goal_id: {item['goal_id']}",
                    "frame: initial_base_heading",
                    f"swing_type: {item['swing_type']}",
                    f"position_m: [{position}]",
                    f"linear_velocity_mps: [{velocity}]",
                    f"racket_normal: [{normal}]",
                    f"time_to_strike_s: {item['time_to_strike_s']:.3f}",
                    f"sequence: {item['sequence']}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    manifest = {
        "schema_version": "a3_target_conditioned_goal_pilot/v1",
        "status": "raw_ik_generation_pending",
        "coordinate_contract": "initial_base_heading/root-relative",
        "goal_fields": ["position_m", "linear_velocity_mps", "racket_normal", "time_to_strike_s"],
        "synchronization_contract": "IK trajectory, hit frame, joint velocity, canonical velocity, normal, and strike time must agree",
        "split_contract": "target_tuple_holdout",
        "goals": goals,
    }
    manifest_path = output_root / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "count": len(goals), "training": sum(x["split"] == "training" for x in goals), "validation": sum(x["split"] == "validation" for x in goals)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
