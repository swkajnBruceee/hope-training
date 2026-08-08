#!/usr/bin/env python3
"""Create a deterministic second-round edge-biased target manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "whole_space_gap_fill_20260807" / "gap_fill_manifest.json"
OUT = ROOT / "whole_space_gap_fill_round2_20260807"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_stroke = {"backhand": [g for g in source["goals"] if g["swing_type"] == "backhand"], "forehand": [g for g in source["goals"] if g["swing_type"] == "forehand"]}
    specs = {
        "backhand": [("time", 2.2), ("time", 0.55), ("vz", 1.33), ("vz", 0.34), ("nz", -0.46), ("nz", 0.03)] * 4,
        "forehand": [("vx", 1.32), ("vx", 1.68), ("vy", -0.10), ("vy", 0.10), ("vz", 0.36), ("vz", 0.60), ("ny", -0.17), ("ny", 0.17), ("time", 1.2), ("time", 2.2)] * 3,
    }
    rng = np.random.default_rng(20260809)
    goals = []
    for stroke, items in specs.items():
        for idx, (kind, value) in enumerate(items):
            base = by_stroke[stroke][(idx * 7 + 3) % len(by_stroke[stroke])]
            position = [float(x) for x in base["position_m"]]
            velocity = [float(x) for x in base["linear_velocity_mps"]]
            normal = np.asarray(base["racket_normal"], dtype=np.float64)
            if kind == "time": t = value
            else: t = float(base["time_to_strike_s"])
            if kind == "vx": velocity[0] = value
            if kind == "vy": velocity[1] = value
            if kind == "vz": velocity[2] = value
            if kind == "ny": normal[1] = value
            if kind == "nz": normal[2] = value
            normal = normal / np.linalg.norm(normal)
            goal_id = f"r2_{'ba' if stroke == 'backhand' else 'fo'}_{idx:03d}"
            goal = {"goal_id": goal_id, "goal_path": f"goals/{goal_id}.yaml", "swing_type": stroke, "split": "validation" if idx % 5 == 0 else "training", "sequence": 620000 + len(goals), "position_m": position, "linear_velocity_mps": velocity, "racket_normal": normal.tolist(), "pitch_deg": float(base.get("pitch_deg", 0.0)), "yaw_deg": float(base.get("yaw_deg", 0.0)), "time_to_strike_s": t, "generation_role": "whole_space_edge_gap_fill_round2"}
            goals.append(goal)
    OUT.joinpath("goals").mkdir(parents=True, exist_ok=True)
    for g in goals:
        lines = ["schema_version: a3_canonical_strike_goal/v1", f"goal_id: {g['goal_id']}", "frame: initial_base_heading", f"swing_type: {g['swing_type']}", f"position_m: {[round(x, 8) for x in g['position_m']]}", f"linear_velocity_mps: {[round(x, 8) for x in g['linear_velocity_mps']]}", f"racket_normal: {[round(x, 8) for x in g['racket_normal']]}", f"time_to_strike_s: {g['time_to_strike_s']:.6f}", f"sequence: {g['sequence']}"]
        OUT.joinpath(g["goal_path"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"schema_version": "a3_whole_space_gap_fill_goals/v2", "status": "raw_ik_generation_pending", "source_coverage": str((ROOT / "whole_space_gap_fill_20260807" / "coverage_after_fk.json").resolve()), "generation_note": "Second round targets sparse velocity/normal/time dimensions using edge substitutions over prior feasible goals.", "per_stroke": 24, "seed": 20260809, "split_counts": {"training": sum(g["split"] == "training" for g in goals), "validation": sum(g["split"] == "validation" for g in goals)}, "goals": goals}
    OUT.joinpath("gap_fill_round2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"goals": len(goals), "training": manifest["split_counts"]["training"], "validation": manifest["split_counts"]["validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
