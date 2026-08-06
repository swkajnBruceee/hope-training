#!/usr/bin/env python3
"""Materialize a 40--60 item P5D-2 reference bank without training.

The bank is split before any PPO run: 43 train references, 9 validation
references, and 10 contiguous bridge holdout references.  An additional 10
boundary references are written as an optional OOD audit set.  The dynamic
diversity slice uses distinct repaired seed solutions for the same nearby
targets; it is explicitly labelled as multi-seed diversity, not hidden as
time scaling.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "eval_outputs/strike_goal_p5/p5_oracle_iter_candidate_00041_p1/manifest.json"
SEED_MANIFEST = ROOT / "eval_outputs/strike_goal_p5/p5d2_repaired_seed_v1/seed_manifest.json"
OFFLINE = ROOT / "eval_outputs/strike_goal_p5/p5d2_multianchor_offline_repaired_v2/manifest.json"
ANCHOR_BANK = ROOT / "eval_outputs/strike_goal_p5/p5d2_multianchor_manifests/motion_npz"
OUT = ROOT / "eval_outputs/strike_goal_p5/p5d2_runtime_reference_bank"
WORLD_ANCHOR = np.asarray((-0.5, -0.7625, 1.04), dtype=np.float64)
SEEDS = (0, 1, 2, 3, 5)


def main() -> None:
    p1 = json.loads(P1.read_text())
    p1_by_id = {int(x["motion_id"]): x for x in p1["motions"]}
    offline = json.loads(OFFLINE.read_text())
    samples = {s["sample_id"]: s for s in offline["samples"]}
    accepted = {}
    for sample in offline["samples"]:
        for a in sample["seed_attempts"]:
            if a.get("qualification") == "PENDING_PHYSX" and a.get("candidate_npz"):
                accepted[(sample["sample_id"], int(a["seed_motion_id"]))] = Path(OFFLINE).parent / a["candidate_npz"]
    out_motion = OUT / "motion_npz"
    out_motion.mkdir(parents=True, exist_ok=True)

    def target_for(sample_id: str, seed: int) -> dict:
        source = copy.deepcopy(p1_by_id[seed]["strike_target"])
        goal = samples[sample_id]["canonical_goal_10d"]
        old_pos = np.asarray(source["racket_position_m"], dtype=np.float64)
        old_ball = np.asarray(source["ball_position_m"], dtype=np.float64)
        source["racket_position_m"] = (WORLD_ANCHOR + np.asarray(goal["position_b0_m"])).tolist()
        source["ball_position_m"] = (np.asarray(source["racket_position_m"]) + old_ball - old_pos).tolist()
        source["racket_normal_w"] = list(goal["normal_b0"])
        source["racket_velocity_mps"] = list(goal["linear_velocity_b0_mps"])
        velocity = np.asarray(source["racket_velocity_mps"])
        source["racket_velocity_direction_w"] = (velocity / np.linalg.norm(velocity)).tolist()
        return source

    def canonical_goal_for(sample_id: str, seed: int) -> dict:
        if sample_id in samples:
            return copy.deepcopy(samples[sample_id]["canonical_goal_10d"])
        label = p1_by_id[seed].get("goal_state_layers", {}).get("canonical_motion_label_b0_before_repair", {})
        target = p1_by_id[seed]["strike_target"]
        position = np.asarray(target["racket_position_m"], dtype=np.float64) - WORLD_ANCHOR
        return {
            "position_b0_m": position.tolist(),
            "normal_b0": list(label.get("racket_normal_b0", target["racket_normal_w"])),
            "linear_velocity_b0_mps": list(label.get("racket_velocity_b0_mps", target["racket_velocity_mps"])),
            "time_to_hit_s": float(p1_by_id[seed].get("hit_event", {}).get("hit_time_from_start_s", 0.6)),
        }

    def package(sample_id: str, seed: int, ordinal: int) -> Path:
        source_path = Path(p1_by_id[seed].get("library_motion_npz") or p1_by_id[seed]["motion_npz"])
        source = {k: np.asarray(v).copy() for k, v in np.load(source_path, allow_pickle=False).items()}
        candidate_path = accepted[(sample_id, seed)]
        candidate = np.load(candidate_path, allow_pickle=False)
        source["joint_pos"] = np.asarray(candidate["joint_pos"], dtype=np.float32)
        source["joint_vel"] = np.asarray(candidate["joint_vel"], dtype=np.float32)
        dst = (out_motion / f"reference_{ordinal:03d}_{sample_id}_seed{seed:02d}.npz").resolve()
        np.savez_compressed(dst, **source)
        return dst

    entries = []
    ordinal = 0

    def add_anchor(seed: int) -> None:
        nonlocal ordinal
        source = copy.deepcopy(p1_by_id[seed])
        if seed == 3:
            path = Path(source["library_motion_npz"] or source["motion_npz"])
        else:
            path = ANCHOR_BANK / f"anchor_motion_{seed:02d}.npz"
        e = source
        e.update({"episode_id": f"p5d2_anchor_{seed:02d}", "motion_id": ordinal, "motion_npz": str(path.resolve()), "library_motion_npz": str(path.resolve()), "canonical_motion_npz": str(path.resolve())})
        e["canonical_goal_10d"] = canonical_goal_for(f"anchor_motion_{seed:02d}", seed)
        e["p5d2_bank"] = {"category": "anchor", "source_seed_motion_id": seed, "teacher_approved": False}
        entries.append(e); ordinal += 1

    for seed in SEEDS:
        add_anchor(seed)

    train_samples = [s["sample_id"] for s in offline["samples"] if s["split"] == "training"]
    val_samples = [s["sample_id"] for s in offline["samples"] if s["split"] == "validation"]
    bridge_samples = [s["sample_id"] for s in offline["samples"] if s["split"] == "bridge_holdout"]
    boundary_samples = [s["sample_id"] for s in offline["samples"] if s["split"] == "boundary_holdout"]

    def add_candidate(sample_id: str, seed: int, category: str, split: str) -> None:
        nonlocal ordinal
        path = package(sample_id, seed, ordinal)
        entry = copy.deepcopy(p1_by_id[seed])
        entry.update({"episode_id": f"p5d2_{sample_id}_seed{seed:02d}", "motion_id": ordinal, "motion_npz": str(path), "library_motion_npz": str(path), "canonical_motion_npz": str(path), "strike_target": target_for(sample_id, seed)})
        entry["canonical_goal_10d"] = canonical_goal_for(sample_id, seed)
        entry["p5d2_bank"] = {"category": category, "split": split, "sample_id": sample_id, "source_seed_motion_id": seed, "teacher_approved": False, "physics_qualified": False}
        entries.append(entry); ordinal += 1

    # 28 local references: two different seed solutions per training target.
    for i, sample_id in enumerate(train_samples):
        for seed in (SEEDS[i % len(SEEDS)], SEEDS[(i + 2) % len(SEEDS)]):
            add_candidate(sample_id, seed, "local_continuation", "training")
    # 10 extra multi-seed dynamic variants, explicitly separated from local
    # position coverage; these are the first dynamics-diversity slice.
    for i, sample_id in enumerate(train_samples[:5]):
        for seed in (SEEDS[(i + 1) % len(SEEDS)], SEEDS[(i + 3) % len(SEEDS)]):
            add_candidate(sample_id, seed, "multi_seed_dynamics_variant", "training")
    for i, sample_id in enumerate(val_samples):
        add_candidate(sample_id, SEEDS[(i + 3) % len(SEEDS)], "validation", "validation")
    for i, sample_id in enumerate(bridge_samples[:10]):
        add_candidate(sample_id, SEEDS[i % len(SEEDS)], "bridge_holdout", "contiguous_holdout")
    boundary_entries_start = len(entries)
    for i, sample_id in enumerate(boundary_samples[:10]):
        add_candidate(sample_id, SEEDS[(i + 1) % len(SEEDS)], "boundary_holdout", "ood_audit")

    def write(name: str, selected: list[dict], split: str) -> None:
        payload = {
            "manifest_name": name,
            "schema_version": "p5d2_runtime_reference_bank/v1",
            "status": "reference_tracker_only_not_teacher_qualified",
            "training_role": "p5d2_dynamic_tracker_reference_only",
            "teacher_data": False,
            "physics_qualified": False,
            "canonical_goal_contract": "canonical_goal_10d/v1",
            "split": split,
            "excluded_motion_ids": {"4": "offline_negative_right_racket_pelvis_collision_distance"},
            "motion_count": len(selected),
            "motions": selected,
        }
        (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    write("p5d2_train_manifest.json", [e for e in entries if e["p5d2_bank"].get("split", "training") == "training" or e["p5d2_bank"].get("category") == "anchor"], "training")
    write("p5d2_validation_manifest.json", [e for e in entries if e["p5d2_bank"].get("split") == "validation"], "validation")
    write("p5d2_bridge_holdout_manifest.json", [e for e in entries if e["p5d2_bank"].get("split") == "contiguous_holdout"], "contiguous_holdout")
    write("p5d2_boundary_ood_manifest.json", [e for e in entries if e["p5d2_bank"].get("split") == "ood_audit"], "ood_audit")
    write("p5d2_all_reference_bank_manifest.json", entries, "all_splits")
    print(json.dumps({"all": len(entries), "train": len([e for e in entries if e["p5d2_bank"].get("split", "training") == "training" or e["p5d2_bank"].get("category") == "anchor"]), "validation": len([e for e in entries if e["p5d2_bank"].get("split") == "validation"]), "bridge_holdout": len([e for e in entries if e["p5d2_bank"].get("split") == "contiguous_holdout"]), "boundary_ood": len([e for e in entries if e["p5d2_bank"].get("split") == "ood_audit"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
