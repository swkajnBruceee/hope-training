#!/usr/bin/env python3
"""Rank the bounded low-dimensional PhysX candidate screening.

The ranking is diagnostic and deterministic; it does not approve teachers or
replace the formal dataset.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = os.environ.get("P5D3A_LOWDIM_VERSION", "v1")
MANIFEST = ROOT / f"eval_outputs/strike_goal_p5/p5d3a_lowdim_reoptimization_{VERSION}/manifest.json"
LOG = ROOT / os.environ.get("P5D3A_LOWDIM_LOG", f"eval_outputs/p5d3a_lowdim_physx_learned_{VERSION}.log")
OUT = ROOT / os.environ.get("P5D3A_LOWDIM_SCREEN_OUT", f"eval_outputs/p5d3a_lowdim_screen_{VERSION}.json")


def vec(value: str) -> list[float]:
    return [float(x) for x in value.split("/")]


def parse_log() -> dict[str, dict]:
    lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    exact: dict[str, dict] = {}
    diagnostic: dict[str, dict] = {}
    for i, line in enumerate(lines):
        if line.startswith("rank,stroke,episode_id,pos_exact,"):
            header = next(csv.reader([line]))
            col = {name: j for j, name in enumerate(header)}
            for raw in lines[i + 1 :]:
                p = next(csv.reader([raw]), [])
                if len(p) <= col["whole_cycle_pass"] or not p[0].isdigit() or not p[2].startswith("p5d2_"):
                    continue
                exact[p[2]] = {
                    "pos_m": float(p[col["pos_exact"]]),
                    "vel_mps": float(p[col["vel_exact"]]),
                    "normal_deg": float(p[col["normal_deg_exact"]]),
                    "composite_pass": bool(int(p[col["composite_pass"]])),
                    "whole_cycle_pass": bool(int(p[col["whole_cycle_pass"]])),
                }
            break
    for i, line in enumerate(lines):
        if line.startswith("rank,episode_id,target_xyz,"):
            header = next(csv.reader([line]))
            col = {name: j for j, name in enumerate(header)}
            for raw in lines[i + 1 :]:
                p = next(csv.reader([raw]), [])
                if len(p) <= col["root_translation_error_m"] or not p[0].isdigit() or not p[1].startswith("p5d2_"):
                    continue
                target_vel = vec(p[col["target_vel_xyz"]])
                actual_vel = vec(p[col["actual_vel_xyz"]])
                diagnostic[p[1]] = {
                    "reference_actual_error_m": float(p[col["reference_minus_actual_m"]]),
                    "velocity_magnitude_error_mps": float(p[col["velocity_magnitude_error_mps"]]),
                    "velocity_direction_error_deg": float(p[col["velocity_direction_error_deg"]]),
                    "velocity_vector_error_mps": math.sqrt(sum((a - b) ** 2 for a, b in zip(actual_vel, target_vel))),
                    "best_pos_error_m": float(p[col["best_pos_error_m"]]),
                    "best_pos_step": int(p[col["best_pos_step"]]),
                    "safety_projection_max_rad": float(p[col["safety_projection_max_rad"]]),
                    "root_translation_error_m": float(p[col["root_translation_error_m"]]),
                }
            break
    out = {}
    for cid in set(exact) & set(diagnostic):
        out[cid] = {**exact[cid], **diagnostic[cid]}
    return out


def loss(row: dict) -> float:
    # Fixed, documented diagnostic objective: position, normal, velocity
    # magnitude/direction, plus a strong safety projection penalty.
    return (
        row["pos_m"]
        + 0.002 * row["normal_deg"]
        + 0.020 * row["velocity_magnitude_error_mps"]
        + 0.0005 * row["velocity_direction_error_deg"]
        + 10.0 * row["safety_projection_max_rad"]
    )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    metrics = parse_log()
    entries = {r["candidate_id"]: r for r in manifest["candidates"]}
    by_source: dict[str, list[dict]] = defaultdict(list)
    for cid, metric in metrics.items():
        entry = entries[cid]
        row = {"candidate_id": cid, "source_episode_id": entry["episode_id"], **entry, **metric}
        row["screen_loss"] = loss(row)
        by_source[row["source_episode_id"]].append(row)
    shortlist = {}
    for source, rows in sorted(by_source.items()):
        baseline = next(r for r in rows if r["optimization_kind"] == "baseline")
        best = min(rows, key=lambda r: r["screen_loss"])
        shortlist[source] = {
            "baseline": baseline,
            "best": best,
            "position_improvement_cm": 100.0 * (baseline["pos_m"] - best["pos_m"]),
            "velocity_magnitude_improvement_mps": baseline["velocity_magnitude_error_mps"] - best["velocity_magnitude_error_mps"],
            "velocity_direction_improvement_deg": baseline["velocity_direction_error_deg"] - best["velocity_direction_error_deg"],
            "loss_improvement": baseline["screen_loss"] - best["screen_loss"],
        }
    result = {
        "schema_version": "p5d3a_lowdim_screen/v1",
        "status": "SCREENED_NO_TRAINING_NO_DATASET_REPLACEMENT",
        "candidate_manifest": str(MANIFEST.relative_to(ROOT)),
        "physx_log": str(LOG.relative_to(ROOT)),
        "candidate_count": len(metrics),
        "source_count": len(by_source),
        "physical_termination_count": 0,
        "objective": {
            "formula": "pos_m + 0.002*normal_deg + 0.020*velocity_magnitude_error_mps + 0.0005*velocity_direction_error_deg + 10*safety_projection_max_rad",
            "canonical_goal_changed": False,
            "model_900_frozen": True,
            "model_3396_frozen": True,
            "model_2198_frozen": True,
        },
        "shortlist": shortlist,
        "all_candidates": [r for rows in by_source.values() for r in rows],
        "conclusion": "bounded one-factor low-dimensional search completed; candidates are diagnostic pending continuity and paired zero-residual validation",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(metrics), "source_count": len(by_source), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
