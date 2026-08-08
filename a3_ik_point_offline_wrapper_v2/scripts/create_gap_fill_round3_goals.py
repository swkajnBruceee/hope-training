#!/usr/bin/env python3
"""Create round-three target goals focused on persistent forehand holes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "whole_space_gap_fill_20260807" / "gap_fill_manifest.json"
OUT = ROOT / "whole_space_gap_fill_round3_20260807"

def main() -> None:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    by = {s: [g for g in src["goals"] if g["swing_type"] == s] for s in ("backhand", "forehand")}
    fh = [("vx", v) for v in (1.32, 1.68)] + [("vy", v) for v in (-.10, .10)] + [("vz", v) for v in (.36, .60)] + [("ny", v) for v in (-.17, .17)] + [("time", v) for v in (1.20, 2.20)]
    bh = [("vz", v) for v in (.34, 1.33)] + [("nz", v) for v in (-.46, .03)] + [("time", v) for v in (.55, 2.20)]
    specs = [("forehand", x) for x in fh for _ in range(3)] + [("backhand", x) for x in bh for _ in range(3)]
    goals = []
    for i, (stroke, (kind, value)) in enumerate(specs):
        base = by[stroke][(i * 11 + 5) % len(by[stroke])]
        vel = [float(x) for x in base["linear_velocity_mps"]]
        normal = np.asarray(base["racket_normal"], dtype=np.float64)
        t = float(base["time_to_strike_s"])
        if kind == "vx": vel[0] = value
        elif kind == "vy": vel[1] = value
        elif kind == "vz": vel[2] = value
        elif kind == "ny": normal[1] = value
        elif kind == "nz": normal[2] = value
        elif kind == "time": t = value
        normal /= np.linalg.norm(normal)
        prefix = "fo" if stroke == "forehand" else "ba"
        gid = f"r3_{prefix}_{i:03d}"
        goal = {"goal_id": gid, "goal_path": f"goals/{gid}.yaml", "swing_type": stroke, "split": "validation" if i % 6 == 0 else "training", "sequence": 630000 + i, "position_m": [float(x) for x in base["position_m"]], "linear_velocity_mps": vel, "racket_normal": normal.tolist(), "pitch_deg": float(base.get("pitch_deg", 0)), "yaw_deg": float(base.get("yaw_deg", 0)), "time_to_strike_s": t, "generation_role": "whole_space_edge_gap_fill_round3"}
        goals.append(goal)
    (OUT / "goals").mkdir(parents=True, exist_ok=True)
    for g in goals:
        (OUT / g["goal_path"]).write_text("\n".join(["schema_version: a3_canonical_strike_goal/v1", f"goal_id: {g['goal_id']}", "frame: initial_base_heading", f"swing_type: {g['swing_type']}", f"position_m: {[round(x, 8) for x in g['position_m']]}", f"linear_velocity_mps: {[round(x, 8) for x in g['linear_velocity_mps']]}", f"racket_normal: {[round(x, 8) for x in g['racket_normal']]}", f"time_to_strike_s: {g['time_to_strike_s']:.6f}", f"sequence: {g['sequence']}"]) + "\n", encoding="utf-8")
    manifest = {"schema_version": "a3_whole_space_gap_fill_goals/v3", "status": "raw_ik_generation_pending", "source_coverage": str((ROOT / "whole_space_gap_fill_round2_20260807" / "coverage_after_round2.json").resolve()), "generation_note": "Persistent forehand velocity/normal/time and backhand edge retargeting.", "goals": goals}
    (OUT / "gap_fill_round3_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"goals": len(goals), "forehand": sum(g["swing_type"] == "forehand" for g in goals), "backhand": sum(g["swing_type"] == "backhand" for g in goals)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
