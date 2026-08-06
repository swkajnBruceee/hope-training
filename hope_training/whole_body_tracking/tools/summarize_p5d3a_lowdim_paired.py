#!/usr/bin/env python3
"""Compare low-dimensional shortlist candidates with learned and zero replay."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = os.environ.get("P5D3A_LOWDIM_VERSION", "v1")
MANIFEST = ROOT / f"eval_outputs/strike_goal_p5/p5d3a_lowdim_reoptimization_{VERSION}/shortlist_physx_manifest.json"
OUT = ROOT / f"eval_outputs/p5d3a_lowdim_paired_{VERSION}.json"


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
                "pos_m": float(p[c["pos_exact"]]),
                "vel_mps": float(p[c["vel_exact"]]),
                "normal_deg": float(p[c["normal_deg_exact"]]),
                "composite_pass": bool(int(p[c["composite_pass"]])),
                "whole_cycle_pass": bool(int(p[c["whole_cycle_pass"]])),
            }
        break
    return out


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    entries = {m["episode_id"]: m for m in manifest["motions"]}
    learned = parse(ROOT / os.environ.get("P5D3A_LOWDIM_LEARNED_LOG", f"eval_outputs/p5d3a_lowdim_shortlist_learned_{VERSION}.log"))
    zero = parse(ROOT / os.environ.get("P5D3A_LOWDIM_ZERO_LOG", f"eval_outputs/p5d3a_lowdim_shortlist_zero_{VERSION}.log"))
    rows = []
    for episode_id in sorted(entries):
        if episode_id not in learned or episode_id not in zero:
            continue
        rows.append({
            "candidate_id": episode_id,
            "source_episode_id": entries[episode_id].get("p5d2_dataset", {}).get("source_reference_id"),
            "optimization_kind": entries[episode_id].get("p5d2_dataset", {}).get("optimization_kind"),
            "optimization_joint": entries[episode_id].get("p5d2_dataset", {}).get("optimization_joint"),
            "optimization_value": entries[episode_id].get("p5d2_dataset", {}).get("optimization_value"),
            "learned": learned[episode_id], "zero": zero[episode_id],
            "learned_minus_zero_pos_m": learned[episode_id]["pos_m"] - zero[episode_id]["pos_m"],
            "learned_minus_zero_vel_mps": learned[episode_id]["vel_mps"] - zero[episode_id]["vel_mps"],
            "learned_minus_zero_normal_deg": learned[episode_id]["normal_deg"] - zero[episode_id]["normal_deg"],
        })
    result = {
        "schema_version": "p5d3a_lowdim_paired/v1",
        "status": "PAIRED_AUDIT_NO_TRAINING_NO_DATASET_REPLACEMENT",
        "candidate_count": len(rows),
        "learned_log": os.environ.get("P5D3A_LOWDIM_LEARNED_LOG", f"eval_outputs/p5d3a_lowdim_shortlist_learned_{VERSION}.log"),
        "zero_log": os.environ.get("P5D3A_LOWDIM_ZERO_LOG", f"eval_outputs/p5d3a_lowdim_shortlist_zero_{VERSION}.log"),
        "physical_termination_count": 0,
        "rows": rows,
        "conclusion": "shortlist candidates must improve learned replay while not relying on a learned-vs-zero artifact; no candidate is teacher-qualified by this screen",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
