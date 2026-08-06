#!/usr/bin/env python3
"""Prepare repaired canonical seed packages and a balanced P5D-2 workspace.

This is an offline data-preparation step only.  It does not approve teachers
and it does not start training.  Motion 4 is intentionally excluded until its
negative racket/pelvis collision is repaired.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "eval_outputs/strike_goal_p5/p5_oracle_iter_candidate_00041_p1"
P1_MANIFEST = P1 / "manifest.json"
REPAIRS = ROOT / "eval_outputs/strike_goal_p5/p5d2_repairs_multianchor"
WORKSPACE = ROOT / "eval_outputs/strike_goal_p5/backhand_position_workspace_v1.json"
OUT = ROOT / "eval_outputs/strike_goal_p5/p5d2_repaired_seed_v1"
SEEDS = (0, 1, 2, 3, 5)


def main() -> None:
    source = json.loads(P1_MANIFEST.read_text())
    by_id = {int(x["motion_id"]): x for x in source["motions"]}
    out_motion = OUT / "canonical_motion_npz"
    out_motion.mkdir(parents=True, exist_ok=True)
    entries = []
    for mid in SEEDS:
        entry = copy.deepcopy(by_id[mid])
        src_path = Path(entry.get("canonical_motion_npz") or entry["motion_npz"]).expanduser()
        src = {k: np.asarray(v).copy() for k, v in np.load(src_path, allow_pickle=False).items()}
        if mid != 3:
            repair = np.load(REPAIRS / f"motion_{mid:02d}/repair_candidate.npz", allow_pickle=False)
            src["joint_pos"] = np.asarray(repair["projected_joint_pos"], dtype=np.float32)
            src["joint_vel"] = np.asarray(repair["projected_joint_vel"], dtype=np.float32)
        dst = (out_motion / f"seed_motion_{mid:02d}.npz").resolve()
        np.savez_compressed(dst, **src)
        entry["motion_id"] = mid
        entry["episode_id"] = f"p5d2_seed_motion_{mid:02d}"
        entry["canonical_motion_npz"] = str(dst)
        entry["motion_npz"] = str(dst)
        entry["library_motion_npz"] = str(dst)
        entry["p5d2_seed_contract"] = {
            "role": "offline_optimizer_initialization_only",
            "repaired": mid != 3,
            "physics_qualified": False,
        }
        entries.append(entry)

    seed_manifest = {
        "manifest_name": "p5d2_repaired_canonical_seed_manifest",
        "schema_version": "p5d2_offline_seed_manifest/v1",
        "training_role": "offline_optimizer_seed_only",
        "teacher_data": False,
        "physics_qualified": False,
        "excluded_motion_ids": {"4": "negative_racket_pelvis_collision_distance"},
        "motions": entries,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "seed_manifest.json").write_text(json.dumps(seed_manifest, ensure_ascii=False, indent=2) + "\n")

    workspace = json.loads(WORKSPACE.read_text())
    buckets: dict[str, list[dict]] = {}
    for row in workspace["samples"]:
        buckets.setdefault(str(row["split"]), []).append(row)
    # Balance the first batch instead of taking the lexicographic prefix,
    # which is almost entirely a boundary slice of the 3-D lattice.
    target_counts = {
        "training": 14,
        "validation": 10,
        "bridge_holdout": 10,
        "workspace_holdout": 10,
        "boundary_holdout": 10,
    }
    selected = []
    for split, count in target_counts.items():
        selected.extend(buckets.get(split, [])[:count])
    balanced = dict(workspace)
    balanced["purpose"] = "p5d2_balanced_multianchor_offline_batch"
    balanced["training_approved"] = False
    balanced["teacher_data"] = False
    balanced["selected_counts"] = {k: sum(x["split"] == k for x in selected) for k in target_counts}
    balanced["samples"] = selected
    (OUT / "balanced_workspace.json").write_text(json.dumps(balanced, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"seed_manifest": str(OUT / "seed_manifest.json"), "workspace": str(OUT / "balanced_workspace.json"), "seed_ids": list(SEEDS), "sample_count": len(selected), "split_counts": balanced["selected_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
