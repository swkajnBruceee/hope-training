#!/usr/bin/env python3
"""Summarize v6 learned action-chain vectors by joint and difficulty class."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_outputs/p5d3a_joint_responsibility_v1.json"
JOINTS = (
    "waist_yaw", "waist_roll", "waist_pitch", "right_shoulder_roll",
    "right_shoulder_pitch", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)


def parse(path: Path) -> dict[str, dict]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = {}
    for i, line in enumerate(lines):
        if not line.startswith("rank,episode_id,target_xyz,"):
            continue
        header = next(csv.reader([line]))
        col = {name: j for j, name in enumerate(header)}
        for raw in lines[i + 1 :]:
            p = next(csv.reader([raw]), [])
            if len(p) <= col["tracker_residual_vector_rad"] or not p[0].isdigit() or not p[1].startswith("p5d2_"):
                continue
            out[p[1]] = {
                "reference_actual_error_m": float(p[col["reference_minus_actual_m"]]),
                "best_pos_error_m": float(p[col["best_pos_error_m"]]),
                "prior": [float(x) for x in p[col["prior_contribution_vector_rad"]].split("/")],
                "tracker": [float(x) for x in p[col["tracker_residual_vector_rad"]].split("/")],
            }
        break
    return out


def stats(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    prior = [r["prior"] for r in rows]
    tracker = [r["tracker"] for r in rows]
    return {
        "count": len(rows),
        "prior_abs_mean_rad_by_joint": [sum(abs(v[j]) for v in prior) / len(prior) for j in range(10)],
        "tracker_abs_mean_rad_by_joint": [sum(abs(v[j]) for v in tracker) / len(tracker) for j in range(10)],
        "prior_abs_max_rad_by_joint": [max(abs(v[j]) for v in prior) for j in range(10)],
        "tracker_abs_max_rad_by_joint": [max(abs(v[j]) for v in tracker) for j in range(10)],
        "tcp_error_mean_m": sum(r["reference_actual_error_m"] for r in rows) / len(rows),
        "best_pos_error_mean_m": sum(r["best_pos_error_m"] for r in rows) / len(rows),
    }


def main() -> None:
    specs = {
        "train": "eval_outputs/p5d2_formal_train_learned_diagnostic_v6.log",
        "validation": "eval_outputs/p5d2_formal_validation_learned_diagnostic_v6.log",
        "holdout": "eval_outputs/p5d2_formal_holdout_learned_diagnostic_v6.log",
    }
    audit = json.loads((ROOT / "eval_outputs/p5d3a_difficulty_audit_v2.json").read_text())
    groups = {g: set(ids) for g, ids in audit["groups"].items()}
    all_rows = {}
    result = {
        "schema_version": "p5d3a_joint_responsibility/v1",
        "status": "AUDIT_ONLY_NO_NEW_TRAINING",
        "joint_order": list(JOINTS),
        "source": "P5D-2 learned PhysX v6 exact-hit action-chain replay",
        "splits": {},
    }
    for split, rel in specs.items():
        rows = parse(ROOT / rel)
        all_rows.update(rows)
        result["splits"][split] = stats(list(rows.values()))
    result["groups"] = {}
    for group, ids in groups.items():
        selected = [all_rows[e] for e in ids if e in all_rows]
        result["groups"][group] = stats(selected)
    result["aggregate"] = stats(list(all_rows.values()))
    result["interpretation"] = {
        "prior_vector": "frozen model_900 contribution relative to the safe reference",
        "tracker_vector": "P5D learned coordinator contribution; this is distinct from total command-reference residual",
        "next_action": "use the per-joint pattern to choose a bounded low-dimensional time/trajectory optimization; do not start PPO from this audit",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
