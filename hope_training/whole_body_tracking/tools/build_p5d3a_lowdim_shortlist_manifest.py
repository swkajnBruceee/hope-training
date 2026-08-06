#!/usr/bin/env python3
"""Materialize the deterministic baseline/best paired replay shortlist."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = os.environ.get("P5D3A_LOWDIM_VERSION", "v1")
SCREEN = ROOT / os.environ.get("P5D3A_LOWDIM_SCREEN", f"eval_outputs/p5d3a_lowdim_screen_{VERSION}.json")
PHYSX = ROOT / f"eval_outputs/strike_goal_p5/p5d3a_lowdim_reoptimization_{VERSION}/physx_manifest.json"
OUT = ROOT / f"eval_outputs/strike_goal_p5/p5d3a_lowdim_reoptimization_{VERSION}/shortlist_physx_manifest.json"


def main() -> None:
    screen = json.loads(SCREEN.read_text())
    physx = json.loads(PHYSX.read_text())
    wanted = set()
    for row in screen["shortlist"].values():
        wanted.add(row["baseline"]["candidate_id"])
        wanted.add(row["best"]["candidate_id"])
    motions = [m for m in physx["motions"] if m["episode_id"] in wanted]
    if len(motions) != len(wanted):
        raise RuntimeError(f"shortlist mismatch: wanted={len(wanted)} found={len(motions)}")
    out = {
        "schema_version": "p5d3a_lowdim_shortlist_physx_manifest/v1",
        "status": "PAIRED_REPLAY_PENDING",
        "canonical_goal_relabelled": False,
        "motion4_excluded": True,
        "source_screen": str(SCREEN.relative_to(ROOT)),
        "motions": motions,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"count": len(motions), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
