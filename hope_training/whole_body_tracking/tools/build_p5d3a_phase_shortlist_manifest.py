#!/usr/bin/env python3
"""Build baseline/best paired replay manifest for a fixed phase screen."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = os.environ.get("P5D3A_PHASE_VERSION", "v2")
SCREEN = ROOT / os.environ.get("P5D3A_PHASE_SCREEN", f"eval_outputs/p5d3a_phase_reoptimization_screen_{VERSION}_fixed.json")
PHYSX = ROOT / f"eval_outputs/strike_goal_p5/p5d3a_phase_reoptimization_{VERSION}_fixed/physx_manifest.json"
OUT = ROOT / f"eval_outputs/strike_goal_p5/p5d3a_phase_reoptimization_{VERSION}_fixed/shortlist_physx_manifest.json"


def main() -> None:
    screen = json.loads(SCREEN.read_text())
    physx = json.loads(PHYSX.read_text())
    wanted = set()
    for row in screen["shortlist"].values():
        wanted.add(row["candidate_id"])
        source = row["source_episode_id"]
        wanted.add(f"{source}_alpha+0".replace("+", "p"))
    # Candidate IDs are generated as alpha p1/m1; the baseline is the source
    # motion itself and is not included in the phase candidate bank.  Resolve
    # it from the canonical source manifest and append it explicitly.
    source_manifest = json.loads((ROOT / "eval_outputs/strike_goal_p5/p5d2_safety_reoptimized_v1/manifest.json").read_text())
    source_entries = {e["episode_id"]: e for e in source_manifest["motions"]}
    motions = [m for m in physx["motions"] if m["episode_id"] in wanted]
    for source in screen["shortlist"]:
        base = dict(source_entries[source])
        base["p5d2_dataset"] = dict(base.get("p5d2_dataset", {}))
        base["p5d2_dataset"].update({"reference_id": source + "_baseline", "source_reference_id": source, "optimization_kind": "baseline"})
        base["episode_id"] = source + "_baseline"
        motions.append(base)
    # Remove accidental alpha+0 requests; there is no such candidate in the
    # generator, and the explicit source baseline above is authoritative.
    motions = [m for m in motions if "_alpha+0" not in m["episode_id"]]
    if len(motions) != 20:
        raise RuntimeError(f"expected 20 phase shortlist motions, got {len(motions)}")
    out = {
        "schema_version": "p5d3a_phase_shortlist_physx_manifest/v1",
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
