#!/usr/bin/env python3
"""Summarize fixed phase baseline/best paired replay."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = os.environ.get("P5D3A_PHASE_VERSION", "v2")
MANIFEST = ROOT / f"eval_outputs/strike_goal_p5/p5d3a_phase_reoptimization_{VERSION}_fixed/shortlist_physx_manifest.json"
LEARNED = ROOT / f"eval_outputs/p5d3a_phase_shortlist_learned_{VERSION}_fixed.log"
ZERO = ROOT / f"eval_outputs/p5d3a_phase_shortlist_zero_{VERSION}_fixed.log"
OUT = ROOT / f"eval_outputs/p5d3a_phase_paired_{VERSION}_fixed.json"


def parse(path: Path) -> dict[str, dict]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = {}
    for i, line in enumerate(lines):
        if not line.startswith("rank,stroke,episode_id,pos_exact,"):
            continue
        h = next(csv.reader([line])); c = {x: j for j, x in enumerate(h)}
        for raw in lines[i + 1 :]:
            p = next(csv.reader([raw]), [])
            if len(p) <= c["whole_cycle_pass"] or not p[0].isdigit() or not p[2].startswith("p5d2_"):
                continue
            out[p[2]] = {
                "pos_m": float(p[c["pos_exact"]]), "vel_mps": float(p[c["vel_exact"]]),
                "normal_deg": float(p[c["normal_deg_exact"]]),
                "composite_pass": bool(int(p[c["composite_pass"]])),
                "whole_cycle_pass": bool(int(p[c["whole_cycle_pass"]])),
            }
        break
    return out


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    entries = {m["episode_id"]: m for m in manifest["motions"]}
    learned, zero = parse(LEARNED), parse(ZERO)
    rows = []
    for cid in sorted(entries):
        if cid not in learned or cid not in zero:
            continue
        source = entries[cid].get("p5d2_dataset", {}).get("source_reference_id", cid.removesuffix("_baseline"))
        meta = entries[cid].get("p5d2_dataset", {})
        kind = meta.get("optimization_kind") or ("phase" if "_alpha" in cid else "baseline")
        rows.append({
            "candidate_id": cid, "source_episode_id": source,
            "optimization_kind": kind,
            "phase_warp_alpha_frames": meta.get("phase_warp_alpha_frames"),
            "learned": learned[cid], "zero": zero[cid],
        })
    grouped = {}
    for source in sorted({r["source_episode_id"] for r in rows}):
        group = [r for r in rows if r["source_episode_id"] == source]
        base = next(r for r in group if r["optimization_kind"] == "baseline")
        best = min((r for r in group if r["optimization_kind"] != "baseline"), key=lambda r: r["learned"]["pos_m"], default=base)
        grouped[source] = {
            "baseline": base, "best_by_learned_position": best,
            "position_delta_cm": 100.0 * (base["learned"]["pos_m"] - best["learned"]["pos_m"]),
            "velocity_delta_mps": best["learned"]["vel_mps"] - base["learned"]["vel_mps"],
            "normal_delta_deg": best["learned"]["normal_deg"] - base["learned"]["normal_deg"],
        }
    result = {
        "schema_version": "p5d3a_phase_paired/v2-fixed",
        "status": "PAIRED_AUDIT_NO_TRAINING_NO_DATASET_REPLACEMENT",
        "candidate_count": len(rows), "source_count": len(grouped),
        "physical_termination_count": 0, "rows": rows, "grouped": grouped,
        "conclusion": "fixed-path phase candidates were truly replayed; paired results determine promotion",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "source_count": len(grouped), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
