#!/usr/bin/env python3
"""Cluster canonical goals, choose one deterministic reference per goal, split data."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


def _goal(entry):
    g = entry["canonical_goal_10d"]
    return np.asarray(g["position_b0_m"] + g["normal_b0"] + g["linear_velocity_b0_mps"] + [g["time_to_hit_s"]], dtype=float)


def _region(entry):
    p = np.asarray(entry["canonical_goal_10d"]["position_b0_m"], dtype=float)
    if p[1] < -0.03:
        return "workspace_low_y"
    if p[1] > 0.03:
        return "workspace_high_y"
    return "workspace_mid_y_low_z" if p[2] < -0.05 else "workspace_mid_y_high_z"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--post-trace", required=True)
    ap.add_argument("--recovery-audit", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    source_path = Path(args.manifest).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    post = np.load(Path(args.post_trace).resolve(), allow_pickle=True)
    post_proj = np.max(np.abs(np.asarray(post["trace"], dtype=float)[:, :, 40:50]), axis=2)
    post_max = np.max(post_proj, axis=0)
    post_mean = np.mean(post_proj, axis=0)
    recovery = json.loads(Path(args.recovery_audit).resolve().read_text(encoding="utf-8"))
    recovery_by_id = {r["episode_id"]: r for r in recovery["rows"]}
    entries = source["motions"]
    clusters = []
    for i, entry in enumerate(entries):
        g = _goal(entry)
        for c in clusters:
            if np.linalg.norm(g - _goal(entries[c[0]])) <= 1.0e-5:
                c.append(i)
                break
        else:
            clusters.append([i])

    selected = []
    cluster_rows = []
    for cluster_id, indices in enumerate(clusters):
        ranked = sorted(
            indices,
            key=lambda i: (
                float(post_max[i]),
                -float(recovery_by_id.get(entries[i]["episode_id"], {}).get("min_root_upright", -1.0)),
                entries[i]["episode_id"],
            ),
        )
        winner = ranked[0]
        selected.append(winner)
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(indices),
                "canonical_goal_10d": entries[winner]["canonical_goal_10d"],
                "candidate_episode_ids": [entries[i]["episode_id"] for i in indices],
                "selected_episode_id": entries[winner]["episode_id"],
                "selection_score": {"post_projection_max_rad": float(post_max[winner]), "min_root_upright": recovery_by_id.get(entries[winner]["episode_id"], {}).get("min_root_upright")},
                "selection_reason": "minimum post-reoptimization safety projection, then recovery upright, then deterministic episode_id",
            }
        )

    selected_entries = {entries[i]["episode_id"]: copy.deepcopy(entries[i]) for i in selected}
    by_cat = {}
    for eid, entry in selected_entries.items():
        by_cat.setdefault(entry.get("p5d2_bank", {}).get("category"), []).append(entry)
    # Fixed spatial-block split. No near duplicate canonical goal crosses splits.
    holdout = [e for e in by_cat.get("boundary_holdout", [])] + by_cat.get("bridge_holdout", [])[:1]
    val = list(by_cat.get("validation", []))
    local = sorted(by_cat.get("local_continuation", []), key=lambda e: (float(_goal(e)[1]), e["episode_id"]))
    if local:
        val.append(local.pop(len(local) // 2))
    train_pool = [e for e in selected_entries.values() if e not in holdout and e not in val]
    anchors = [e for e in train_pool if e.get("p5d2_bank", {}).get("category") == "anchor"]
    local_pool = [e for e in train_pool if e.get("p5d2_bank", {}).get("category") == "local_continuation"]
    bridge_pool = [e for e in train_pool if e.get("p5d2_bank", {}).get("category") == "bridge_holdout"]
    train = anchors + local_pool[:10] + bridge_pool[:2]
    if len(train) < 16:
        extras = [e for e in train_pool if e not in train]
        train.extend(extras[: 16 - len(train)])
    train = train[:16]
    used = {e["episode_id"] for e in train + val + holdout}

    def decorate(entry, split):
        e = copy.deepcopy(entry)
        i = entries.index(next(src for src in entries if src["episode_id"] == e["episode_id"]))
        recovery_row = recovery_by_id.get(e["episode_id"], {})
        e["p5d2_dataset"] = {
            "reference_id": e["episode_id"],
            "split": split,
            "eligibility": "TRACKER_TRAINING_ELIGIBLE",
            "region": _region(e),
            "canonical_goal_10d": e["canonical_goal_10d"],
            "seed_motion_id_audit_only": e.get("p5d2_bank", {}).get("source_seed_motion_id"),
            "hit_frame": e["reference_contract"]["hit_frame"],
            "control_dt": 1.0 / float(e["fps"]),
            "tcp_definition": e["reference_contract"]["tcp_contract"],
            "frame_definition": e["reference_contract"]["coordinate_frame"],
            "projection_max_rad_after_reoptimization": float(post_max[i]),
            "projection_mean_rad_after_reoptimization": float(post_mean[i]),
            "recovery_pass": not bool(recovery_row.get("physical_terminated", True)),
            "generator_version": e["reference_contract"]["schema"],
            "qualification_version": "p5d2_tracker_reference_eligibility/v2",
            "actual_tracking_error_is_not_rejection": True,
            "motion_id_to_actor": False,
        }
        e["p5d2_bank"]["split"] = split
        e["p5d2_bank"]["continuity_audited"] = True
        e["p5d2_bank"]["physics_qualified"] = False
        e["p5d2_bank"]["teacher_approved"] = False
        return e

    all_selected = [decorate(e, "train") for e in train] + [decorate(e, "validation") for e in val] + [decorate(e, "holdout") for e in holdout]
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {"schema_version": "p5d2_deduplicated_dataset/v1", "status": "tracker_training_eligible_pending_teacher", "source_manifest": str(source_path), "canonical_goal_cluster_tolerance_10d": 1.0e-5, "canonical_goal_cluster_count": len(clusters), "selected_count": len(all_selected), "actor_receives_reference_id": False, "teacher_approved": False, "training_started": False, "cluster_audit": cluster_rows}
    for name, rows in (("train", [e for e in all_selected if e["p5d2_dataset"]["split"] == "train"]), ("validation", [e for e in all_selected if e["p5d2_dataset"]["split"] == "validation"]), ("holdout", [e for e in all_selected if e["p5d2_dataset"]["split"] == "holdout"]), ("all", all_selected)):
        payload = copy.deepcopy(base); payload["motions"] = rows; payload["split"] = name; payload["count"] = len(rows); payload["region_counts"] = {r: sum(1 for e in rows if e["p5d2_dataset"]["region"] == r) for r in sorted({_region(e) for e in rows})}
        (out_dir / f"p5d2_{name}_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {"schema_version": "p5d2_dedup_split_report/v1", "source": str(source_path), "cluster_count": len(clusters), "selected_count": len(all_selected), "train_count": len(train), "validation_count": len(val), "holdout_count": len(holdout), "omitted_cluster_representatives": sorted(set(selected_entries) - used), "region_counts": {s: {r: sum(1 for e in rows if _region(e) == r) for r in sorted({_region(e) for e in rows})} for s, rows in (("train", train), ("validation", val), ("holdout", holdout))}, "clusters": cluster_rows, "training_started": False}
    (out_dir / "p5d2_dedup_split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "cluster_count": len(clusters), "selected_count": len(all_selected), "train": len(train), "validation": len(val), "holdout": len(holdout), "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
