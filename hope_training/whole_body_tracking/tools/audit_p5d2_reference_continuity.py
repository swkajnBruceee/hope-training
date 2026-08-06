#!/usr/bin/env python3
"""Audit continuity of the P5D-2 reference bank without changing labels.

This is deliberately a metric report, not an eligibility promotion step.  It
compares nearby canonical goals with their complete joint trajectories and
records the nearest-neighbour geometry for every split/category.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: str):
    data = np.load(path)
    return {
        "q": np.asarray(data["joint_pos"], dtype=np.float64),
        "dq": np.asarray(data["joint_vel"], dtype=np.float64),
        "goal": np.concatenate(
            [
                np.asarray(data["canonical_goal_position_b0_m"], dtype=np.float64),
                np.asarray(data["canonical_goal_normal_b0"], dtype=np.float64),
                np.asarray(data["canonical_goal_linear_velocity_b0_mps"], dtype=np.float64),
                np.asarray(data["canonical_goal_time_to_hit_s"], dtype=np.float64).reshape(-1),
            ]
        ),
        "hit": int(np.asarray(data["hit_frame"]).reshape(-1)[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    motions = manifest["motions"]
    records = []
    for item in motions:
        arrays = _load(item["motion_npz"])
        records.append({"item": item, **arrays})

    goals = np.stack([r["goal"] for r in records])
    q_hit = np.stack([r["q"][r["hit"]] for r in records])
    q_all = np.stack([r["q"] for r in records])
    dq_all = np.stack([r["dq"] for r in records])
    rows = []
    for i, r in enumerate(records):
        d_goal = np.linalg.norm(goals - goals[i], axis=1)
        d_goal[i] = np.inf
        j = int(np.argmin(d_goal))
        # Generated seed/phase variants may differ by floating-point noise;
        # treat goals within 1e-5 in the full 10-D contract as the same target
        # for multimodality/coverage auditing.
        duplicate_goal = bool(d_goal[j] <= 1.0e-5)
        d_distinct = d_goal.copy()
        d_distinct[d_distinct <= 1.0e-5] = np.inf
        j_distinct = int(np.argmin(d_distinct)) if np.isfinite(d_distinct).any() else None
        q_delta = q_all[j] - q_all[i]
        dq_delta = dq_all[j] - dq_all[i]
        row = {
                "episode_id": r["item"]["episode_id"],
                "category": r["item"].get("p5d2_bank", {}).get("category"),
                "split": r["item"].get("p5d2_bank", {}).get("split"),
                "nearest_episode_id": records[j]["item"]["episode_id"],
                "nearest_goal_distance_l2": float(d_goal[j]),
                "nearest_hit_q_rms_rad": float(np.sqrt(np.mean((q_hit[j] - q_hit[i]) ** 2))),
                "nearest_hit_q_max_abs_rad": float(np.max(np.abs(q_hit[j] - q_hit[i]))),
                "nearest_trajectory_q_rms_rad": float(np.sqrt(np.mean(q_delta**2))),
                "nearest_trajectory_q_max_abs_rad": float(np.max(np.abs(q_delta))),
                "nearest_trajectory_dq_rms_rad_s": float(np.sqrt(np.mean(dq_delta**2))),
                "nearest_goal_is_duplicate": duplicate_goal,
            }
        if j_distinct is not None:
            q_delta_distinct = q_all[j_distinct] - q_all[i]
            dq_delta_distinct = dq_all[j_distinct] - dq_all[i]
            row.update(
                {
                    "nearest_distinct_episode_id": records[j_distinct]["item"]["episode_id"],
                    "nearest_distinct_goal_distance_l2": float(d_distinct[j_distinct]),
                    "nearest_distinct_hit_q_rms_rad": float(np.sqrt(np.mean((q_hit[j_distinct] - q_hit[i]) ** 2))),
                    "nearest_distinct_trajectory_q_rms_rad": float(np.sqrt(np.mean(q_delta_distinct**2))),
                    "nearest_distinct_trajectory_dq_rms_rad_s": float(np.sqrt(np.mean(dq_delta_distinct**2))),
                }
            )
        rows.append(row)

    def stats(subset):
        if not subset:
            return {"count": 0}
        keys = [
            "nearest_goal_distance_l2",
            "nearest_hit_q_rms_rad",
            "nearest_hit_q_max_abs_rad",
            "nearest_trajectory_q_rms_rad",
            "nearest_trajectory_q_max_abs_rad",
            "nearest_trajectory_dq_rms_rad_s",
            "nearest_distinct_goal_distance_l2",
            "nearest_distinct_hit_q_rms_rad",
            "nearest_distinct_trajectory_q_rms_rad",
            "nearest_distinct_trajectory_dq_rms_rad_s",
        ]
        return {
            "count": len(subset),
            **{
                key: {
                    "min": float(min(x[key] for x in subset if key in x)),
                    "mean": float(np.mean([x[key] for x in subset if key in x])),
                    "max": float(max(x[key] for x in subset if key in x)),
                }
                for key in keys
            },
        }

    out = {
        "schema_version": "p5d2_reference_continuity_audit/v1",
        "manifest": str(manifest_path),
        "reference_count": len(records),
        "continuity_audit_completed": True,
        "promotion_performed": False,
        "thresholds": None,
        "threshold_note": "Metrics are reported fail-closed; no eligibility promotion occurs without an approved continuity contract.",
        "duplicate_canonical_goal_count": int(sum(r["nearest_goal_is_duplicate"] for r in rows)),
        "unique_canonical_goal_count_at_1e-5": int(np.unique(np.round(goals, 5), axis=0).shape[0]),
        "overall": stats(rows),
        "by_split": {k: stats([r for r in rows if r["split"] == k]) for k in sorted({r["split"] for r in rows})},
        "by_category": {k: stats([r for r in rows if r["category"] == k]) for k in sorted({r["category"] for r in rows})},
        "rows": rows,
    }
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_path), "reference_count": len(records), "overall": out["overall"]}, indent=2))


if __name__ == "__main__":
    main()
