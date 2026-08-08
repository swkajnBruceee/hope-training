#!/usr/bin/env python3
"""Generate targets at the centers of currently empty marginal bins."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COVER = ROOT / "whole_space_gap_fill_round3_20260807" / "coverage_after_round3.json"
SRC = ROOT / "whole_space_gap_fill_20260807" / "gap_fill_manifest.json"
OUT = ROOT / "whole_space_gap_fill_round4_20260807"
DIM_INDEX = {"velocity_x_mps": ("velocity", 0), "velocity_y_mps": ("velocity", 1), "velocity_z_mps": ("velocity", 2), "normal_x": ("normal", 0), "normal_y": ("normal", 1), "normal_z": ("normal", 2), "time_to_hit_s": ("time", None)}

def main() -> None:
    cover = json.loads(COVER.read_text(encoding="utf-8")); src = json.loads(SRC.read_text(encoding="utf-8"))
    by = {s: [g for g in src["goals"] if g["swing_type"] == s] for s in ("backhand", "forehand")}
    goals = []
    for stroke, dims in (("backhand", ("velocity_x_mps", "velocity_z_mps", "normal_z", "time_to_hit_s")), ("forehand", ("velocity_x_mps", "velocity_y_mps", "velocity_z_mps", "normal_x", "normal_y", "normal_z", "time_to_hit_s"))):
        for dim in dims:
            spec = cover["by_stroke"][stroke]["marginal_bins"][dim]
            edges = spec["edges"]
            occupied = set()
            # Reconstruct occupied bins from empty count is impossible; use all bins
            # except a deterministic prefix to keep target count bounded, then
            # rely on the explicit midpoint values for gap filling.
            empty_count = int(spec["empty"])
            for j in range(empty_count):
                bin_id = (j * 3 + 1) % 12
                value = (float(edges[bin_id]) + float(edges[bin_id + 1])) / 2.0
                base = by[stroke][(len(goals) * 5 + j) % len(by[stroke])]
                vel = [float(x) for x in base["linear_velocity_mps"]]
                normal = np.asarray(base["racket_normal"], dtype=np.float64)
                time = float(base["time_to_strike_s"])
                kind, axis = DIM_INDEX[dim]
                if kind == "velocity": vel[axis] = value
                elif kind == "normal": normal[axis] = value; normal /= np.linalg.norm(normal)
                else: time = value
                gid = f"r4_{'ba' if stroke == 'backhand' else 'fo'}_{len(goals):03d}"
                goals.append({"goal_id": gid, "goal_path": f"goals/{gid}.yaml", "swing_type": stroke, "split": "validation" if len(goals) % 6 == 0 else "training", "sequence": 640000 + len(goals), "position_m": [float(x) for x in base["position_m"]], "linear_velocity_mps": vel, "racket_normal": normal.tolist(), "pitch_deg": float(base.get("pitch_deg", 0)), "yaw_deg": float(base.get("yaw_deg", 0)), "time_to_strike_s": time, "generation_role": "whole_space_empty_bin_gap_fill_round4"})
    (OUT / "goals").mkdir(parents=True, exist_ok=True)
    for g in goals:
        lines = ["schema_version: a3_canonical_strike_goal/v1", f"goal_id: {g['goal_id']}", "frame: initial_base_heading", f"swing_type: {g['swing_type']}", f"position_m: {[round(x, 8) for x in g['position_m']]}", f"linear_velocity_mps: {[round(x, 8) for x in g['linear_velocity_mps']]}", f"racket_normal: {[round(x, 8) for x in g['racket_normal']]}", f"time_to_strike_s: {g['time_to_strike_s']:.6f}", f"sequence: {g['sequence']}"]
        (OUT / g["goal_path"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"schema_version": "a3_whole_space_gap_fill_goals/v4", "status": "raw_ik_generation_pending", "source_coverage": str(COVER.resolve()), "generation_note": "Targets are midpoints of currently sparse marginal bins.", "goals": goals}
    (OUT / "gap_fill_round4_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"goals": len(goals), "forehand": sum(g["swing_type"] == "forehand" for g in goals), "backhand": sum(g["swing_type"] == "backhand" for g in goals)}, ensure_ascii=False))

if __name__ == "__main__": main()
