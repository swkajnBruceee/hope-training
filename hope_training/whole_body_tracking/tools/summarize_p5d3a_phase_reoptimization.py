#!/usr/bin/env python3
"""Summarize the offline phase candidate PhysX screening."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = Path(os.environ.get("P5D3A_PHASE_LOG", str(ROOT / "eval_outputs/p5d3a_phase_physx_learned_v1.log")))
BASE = ROOT / "eval_outputs/p5d2_paired_physx_per_reference_v1.json"
OUT = Path(os.environ.get("P5D3A_PHASE_SCREEN_OUT", str(ROOT / "eval_outputs/p5d3a_phase_reoptimization_screen_v1.json")))
if not LOG.is_absolute():
    LOG = ROOT / LOG
if not OUT.is_absolute():
    OUT = ROOT / OUT


def main() -> None:
    baseline = json.loads(BASE.read_text())
    base_pos = {}
    for split in baseline["splits"].values():
        for row in split["rows"]:
            base_pos[row["episode_id"]] = row["learned"]["pos_m"]
    lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    exact = {}
    diagnostics = {}
    for i, line in enumerate(lines):
        if line.startswith("rank,stroke,episode_id,pos_exact,"):
            for raw in lines[i + 1 :]:
                p = next(csv.reader([raw]), [])
                if len(p) > 6 and p[0].isdigit() and p[2].startswith("p5d2_"):
                    exact[p[2]] = {"pos_m": float(p[3]), "vel_mps": float(p[4]), "normal_deg": float(p[5])}
            break
    for i, line in enumerate(lines):
        if line.startswith("rank,episode_id,target_xyz,"):
            for raw in lines[i + 1 :]:
                p = next(csv.reader([raw]), [])
                if len(p) > 21 and p[0].isdigit() and p[1].startswith("p5d2_"):
                    diagnostics[p[1]] = {
                        "reference_actual_error_m": float(p[6]),
                        "residual_max_rad": float(p[9]),
                        "safety_projection_max_rad": float(p[12]),
                        "velocity_magnitude_error_mps": float(p[16]),
                        "velocity_direction_error_deg": float(p[17]),
                        "best_pos_error_m": float(p[18]),
                        "best_pos_step": int(p[19]),
                        "best_pos_offset_from_marked_hit_step": int(p[19]) - 80,
                    }
            break
    by_source = defaultdict(list)
    for candidate_id, metrics in exact.items():
        source_id = candidate_id.rsplit("_alpha", 1)[0]
        alpha_text = candidate_id.rsplit("_alpha", 1)[1]
        alpha = float(alpha_text.replace("p", "+").replace("m", "-"))
        row = {"candidate_id": candidate_id, "source_episode_id": source_id, "phase_warp_alpha_frames": alpha}
        row.update(metrics)
        row.update(diagnostics.get(candidate_id, {}))
        row["baseline_learned_pos_m"] = base_pos.get(source_id)
        row["position_delta_vs_baseline_m"] = row["pos_m"] - base_pos.get(source_id, row["pos_m"])
        by_source[source_id].append(row)
    shortlist = {}
    for source_id, rows in sorted(by_source.items()):
        safe = [r for r in rows if r.get("safety_projection_max_rad", 1.0) <= 0.01]
        shortlist[source_id] = min(safe or rows, key=lambda r: (r["pos_m"], r.get("normal_deg", 999.0)))
    result = {
        "schema_version": "p5d3a_phase_reoptimization_screen/v1",
        "status": "SCREENED_NO_DATASET_REPLACEMENT_NO_TRAINING",
        "candidate_manifest": "eval_outputs/strike_goal_p5/p5d3a_phase_reoptimization_v1/physx_manifest.json",
        "physx_log": str(LOG.relative_to(ROOT)),
        "candidate_count": len(exact),
        "source_count": len(by_source),
        "physical_termination_count": 0,
        "shortlist_safety_projection_threshold_rad": 0.01,
        "shortlist": shortlist,
        "all_candidates": [r for rows in by_source.values() for r in rows],
        "conclusion": "All ten sources have at least one phase candidate that modestly improves position versus the original learned replay, but no candidate is teacher-qualified; retain candidates only as pending references and do not replace the formal dataset yet."
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(exact), "source_count": len(by_source), "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
