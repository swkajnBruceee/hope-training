#!/usr/bin/env python3
"""Package the first P5D-2 multi-anchor reference distribution.

Motion 0/1/2/5 are the same reviewed P1 anchors after deterministic offline
waist repair. Motion 3 is the already repaired P5 anchor. Motion 4 is kept
out deliberately: its offline collision audit remains negative. A few P5
position continuations are appended to the train/holdout split so the first
multi-anchor run is not just five isolated IDs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "eval_outputs/strike_goal_p5/p5_oracle_iter_candidate_00041_p1"
P1_MANIFEST = P1 / "manifest.json"
OFFLINE = ROOT / "eval_outputs/strike_goal_p5/offline_smoke_velocity_v2"
REPAIRS = ROOT / "eval_outputs/strike_goal_p5/p5d2_repairs_multianchor"
OUT = ROOT / "eval_outputs/strike_goal_p5/p5d2_multianchor_manifests"
ANCHOR = np.asarray((-0.5, -0.7625, 1.04), dtype=np.float64)


def main() -> None:
    source = json.loads(P1_MANIFEST.read_text(encoding="utf-8"))
    by_id = {int(x["motion_id"]): x for x in source["motions"]}
    # Motion 4 is not silently accepted: its P5 repair has a negative
    # right-racket/pelvis collision distance in the offline audit.
    offline_audit = json.loads((REPAIRS / "motion_04/repair_audit.json").read_text()) if (REPAIRS / "motion_04/repair_audit.json").exists() else None
    if offline_audit and offline_audit.get("all_offline_gates_pass"):
        raise RuntimeError("motion 4 unexpectedly passed; update the explicit exclusion contract")

    samples = json.loads((OFFLINE / "manifest.json").read_text())
    sample_by_id = {s["sample_id"]: s for s in samples["samples"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "motion_npz").mkdir(exist_ok=True)

    def package_anchor(original_id: int) -> Path:
        entry = by_id[original_id]
        # The runtime loader consumes the scene-placed P1 package (body_*_w),
        # while the repair tool consumes the canonical b0 package.
        src_path = Path(entry.get("library_motion_npz") or entry["motion_npz"]).expanduser()
        if not src_path.exists():
            src_path = Path(entry["canonical_motion_npz"]).expanduser()
        src = {k: np.asarray(v).copy() for k, v in np.load(src_path, allow_pickle=False).items()}
        if original_id == 3:
            return src_path.resolve()
        candidate = REPAIRS / f"motion_{original_id:02d}/repair_candidate.npz"
        c = np.load(candidate, allow_pickle=False)
        src["joint_pos"] = np.asarray(c["projected_joint_pos"], dtype=np.float32)
        src["joint_vel"] = np.asarray(c["projected_joint_vel"], dtype=np.float32)
        out = OUT / "motion_npz" / f"anchor_motion_{original_id:02d}.npz"
        np.savez_compressed(out, **src)
        return out.resolve()

    def package_continuation(sample_id: str) -> Path:
        src_path = by_id[3].get("library_motion_npz") or by_id[3]["motion_npz"]
        src = {k: np.asarray(v).copy() for k, v in np.load(src_path, allow_pickle=False).items()}
        c = np.load(OFFLINE / f"{sample_id}_seed03.npz", allow_pickle=False)
        src["joint_pos"] = np.asarray(c["joint_pos"], dtype=np.float32)
        src["joint_vel"] = np.asarray(c["joint_vel"], dtype=np.float32)
        out = OUT / "motion_npz" / f"continuation_{sample_id}.npz"
        np.savez_compressed(out, **src)
        return out.resolve()

    packaged = {i: package_anchor(i) for i in (0, 1, 2, 3, 5)}
    for sid in ("p5_pos_00041", "p5_pos_00043", "p5_pos_00046", "p5_pos_00048"):
        packaged[sid] = package_continuation(sid)

    def target_for(sample_id: str, entry: dict) -> dict:
        if sample_id not in sample_by_id:
            return copy.deepcopy(entry["strike_target"])
        goal = sample_by_id[sample_id]["canonical_goal_10d"]
        target = copy.deepcopy(entry["strike_target"])
        old_pos = np.asarray(target["racket_position_m"], dtype=np.float64)
        old_ball = np.asarray(target["ball_position_m"], dtype=np.float64)
        target["racket_position_m"] = (ANCHOR + np.asarray(goal["position_b0_m"])).tolist()
        target["ball_position_m"] = (np.asarray(target["racket_position_m"]) + old_ball - old_pos).tolist()
        target["racket_normal_w"] = list(goal["normal_b0"])
        target["racket_velocity_mps"] = list(goal["linear_velocity_b0_mps"])
        v = np.asarray(target["racket_velocity_mps"])
        target["racket_velocity_direction_w"] = (v / np.linalg.norm(v)).tolist()
        return target

    def make_entry(sample_id: str, original_id: int, new_id: int) -> dict:
        e = copy.deepcopy(by_id[original_id])
        e["episode_id"] = f"p5d2_{sample_id}"
        e["motion_id"] = new_id
        e["motion_npz"] = str(packaged[sample_id] if sample_id in packaged else packaged[original_id])
        e["library_motion_npz"] = e["motion_npz"]
        e["canonical_motion_npz"] = e["motion_npz"]
        e["strike_target"] = target_for(sample_id, e)
        e["p5d2_reference"] = {
            "sample_id": sample_id,
            "role": "tracker_reference_only",
            "teacher_approved": False,
            "physics_qualified": False,
            "source_anchor_motion_id": original_id,
        }
        return e

    train_specs = [(f"anchor_motion_{i:02d}", i) for i in (0, 1, 2, 3, 5)]
    train_specs += [(sid, 3) for sid in ("p5_pos_00041", "p5_pos_00043", "p5_pos_00046")]
    holdout_specs = [("p5_pos_00048", 3)]

    def write(name: str, specs: list[tuple[str, int]], split: str) -> None:
        entries = [make_entry(sid, original_id, i) for i, (sid, original_id) in enumerate(specs)]
        payload = {
            "manifest_name": name,
            "schema_version": "p5d2_multianchor_reference_manifest/v1",
            "status": "reference_tracker_only_not_teacher_qualified",
            "training_role": "p5d2_dynamic_tracker_reference_only",
            "teacher_data": False,
            "physics_qualified": False,
            "canonical_goal_contract": "canonical_goal_10d/v1",
            "split": split,
            "selection_rule": "five repaired anchors plus P5 continuation candidates; motion 4 explicitly excluded by collision gate",
            "excluded_anchors": {"motion_4": "offline_negative_right_racket_pelvis_collision_distance"},
            "motions": entries,
        }
        (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    write("p5d2_multianchor_train_manifest.json", train_specs, "training")
    write("p5d2_multianchor_holdout_manifest.json", holdout_specs, "contiguous_holdout")
    print(json.dumps({"train": len(train_specs), "holdout": len(holdout_specs), "excluded": [4]}, indent=2))


if __name__ == "__main__":
    main()
