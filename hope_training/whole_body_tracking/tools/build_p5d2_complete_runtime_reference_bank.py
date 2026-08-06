#!/usr/bin/env python3
"""Build the P5D-2 runtime bank from complete, offline-audited references.

This is a packaging/audit-preparation step only.  It never starts PPO or a
PhysX training job.  Candidates are selected only after the complete
READY-to-hit-to-recovery offline gates pass.  Runtime packages are scene
placed from canonical base-heading coordinates using the frozen P1 root
anchor; canonical goals are copied from the candidate and are never relabelled
from an actual trajectory.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "eval_outputs/strike_goal_p5/p5_oracle_iter_candidate_00041_p1/manifest.json"
SEEDS = ROOT / "eval_outputs/strike_goal_p5/p5d2_repaired_seed_v1/seed_manifest.json"
COMPLETE = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_offline_v1/manifest.json"
ANCHORS = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_offline_v1/anchor_manifest.json"
PROGRAMMATIC = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_offline_v1/programmatic_variants/manifest.json"
OUT = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_runtime_reference_bank"
WORLD_ANCHOR = np.asarray((-0.5, -0.7625, 1.04), dtype=np.float64)
SEED_IDS = (0, 1, 2, 3, 5)  # motion 4 is intentionally excluded


def goal_from_npz(z: np.lib.npyio.NpzFile) -> dict:
    return {
        "position_b0_m": np.asarray(z["canonical_goal_position_b0_m"], dtype=float).tolist(),
        "normal_b0": np.asarray(z["canonical_goal_normal_b0"], dtype=float).tolist(),
        "linear_velocity_b0_mps": np.asarray(z["canonical_goal_linear_velocity_b0_mps"], dtype=float).tolist(),
        "time_to_hit_s": float(np.asarray(z["canonical_goal_time_to_hit_s"]).reshape(-1)[0]),
    }


def target_from_goal(seed_target: dict, goal: dict) -> dict:
    out = copy.deepcopy(seed_target)
    old_pos = np.asarray(seed_target["racket_position_m"], dtype=float)
    old_ball = np.asarray(seed_target["ball_position_m"], dtype=float)
    pos_w = WORLD_ANCHOR + np.asarray(goal["position_b0_m"], dtype=float)
    out["racket_position_m"] = pos_w.tolist()
    out["ball_position_m"] = (pos_w + old_ball - old_pos).tolist()
    out["racket_normal_w"] = list(goal["normal_b0"])
    out["racket_velocity_mps"] = list(goal["linear_velocity_b0_mps"])
    vel = np.asarray(out["racket_velocity_mps"], dtype=float)
    if np.linalg.norm(vel) > 1e-9:
        out["racket_velocity_direction_w"] = (vel / np.linalg.norm(vel)).tolist()
    return out


def package(src_path: Path, dst_path: Path, goal: dict | None = None) -> None:
    z = np.load(src_path, allow_pickle=False)
    required = {"fps", "joint_pos", "joint_vel", "body_pos_b0", "body_quat_b0_wxyz", "body_lin_vel_b0", "body_ang_vel_b0", "hit_frame"}
    missing = required.difference(z.files)
    if missing:
        raise ValueError(f"{src_path} missing {sorted(missing)}")
    payload = {k: np.asarray(z[k]).copy() for k in z.files}
    payload["body_pos_w"] = np.asarray(z["body_pos_b0"], dtype=np.float32) + WORLD_ANCHOR.astype(np.float32)
    payload["body_quat_w"] = np.asarray(z["body_quat_b0_wxyz"], dtype=np.float32)
    payload["body_lin_vel_w"] = np.asarray(z["body_lin_vel_b0"], dtype=np.float32)
    payload["body_ang_vel_w"] = np.asarray(z["body_ang_vel_b0"], dtype=np.float32)
    payload["scene_root_anchor_w_m"] = WORLD_ANCHOR.astype(np.float32)
    payload["scene_root_heading_w_rad"] = np.asarray([0.0], dtype=np.float32)
    if goal is not None:
        payload.setdefault("canonical_goal_position_b0_m", np.asarray(goal["position_b0_m"], dtype=np.float32))
        payload.setdefault("canonical_goal_normal_b0", np.asarray(goal["normal_b0"], dtype=np.float32))
        payload.setdefault("canonical_goal_linear_velocity_b0_mps", np.asarray(goal["linear_velocity_b0_mps"], dtype=np.float32))
        payload.setdefault("canonical_goal_time_to_hit_s", np.asarray([goal["time_to_hit_s"]], dtype=np.float32))
    payload["physics_qualified"] = np.asarray([False])
    np.savez_compressed(dst_path, **payload)


def main() -> None:
    p1 = json.loads(P1.read_text())
    p1_by_id = {int(m["motion_id"]): m for m in p1["motions"]}
    seed_manifest = json.loads(SEEDS.read_text())
    seed_by_id = {int(m["motion_id"]): m for m in seed_manifest["motions"]}
    complete = json.loads(COMPLETE.read_text())
    programmatic = json.loads(PROGRAMMATIC.read_text()) if PROGRAMMATIC.exists() else {"samples": []}
    eligible = [r for r in complete["samples"] if r.get("qualification") == "TRACKER_TRAINING_ELIGIBLE" and int(r["seed_motion_id"]) in SEED_IDS]
    by_key = {(r["sample_id"], int(r["seed_motion_id"])): r for r in eligible}
    by_split = {}
    for r in eligible:
        by_split.setdefault(r["split"], []).append(r)
    for rows in by_split.values():
        rows.sort(key=lambda r: (r["sample_id"], int(r["seed_motion_id"])))

    anchor_manifest = json.loads(ANCHORS.read_text())
    anchor_rows = {int(r["seed_motion_id"]): r for r in anchor_manifest["rows"] if r.get("qualification") == "TRACKER_TRAINING_ELIGIBLE" and int(r["seed_motion_id"]) in SEED_IDS}
    OUT.mkdir(parents=True, exist_ok=True)
    out_motion = OUT / "motion_npz"; out_motion.mkdir(parents=True, exist_ok=True)

    selected = []
    used = set()
    ordinal = 0

    def add(src_path: Path, seed: int, category: str, split: str, sample_id: str, anchor: bool = False) -> None:
        nonlocal ordinal
        z = np.load(src_path, allow_pickle=False)
        if "canonical_goal_position_b0_m" in z.files:
            goal = goal_from_npz(z)
        else:
            # Original repaired seed anchors carry the canonical strike
            # contract in the manifest rather than duplicating it in NPZ.
            target = seed_by_id[seed]["strike_target"]
            velocity = np.asarray(target["racket_velocity_mps"], dtype=float)
            # Repaired seed strike_target is stored in the P1 world contract;
            # the complete anchor NPZ is canonical base-heading.  Convert
            # only the position here.  Do not treat the world target as b0.
            goal = {"position_b0_m": (np.asarray(target["racket_position_m"], dtype=float) - WORLD_ANCHOR).tolist(), "normal_b0": np.asarray(target["racket_normal_w"], dtype=float).tolist(), "linear_velocity_b0_mps": velocity.tolist(), "time_to_hit_s": float(seed_by_id[seed].get("hit_event", {}).get("hit_time_from_start_s", 0.6))}
        stem = f"reference_{ordinal:03d}_{sample_id}_seed{seed:02d}"
        dst = (out_motion / f"{stem}.npz").resolve()
        package(src_path, dst, goal)
        base = copy.deepcopy(p1_by_id[seed])
        seed_target = copy.deepcopy(seed_by_id[seed]["strike_target"])
        base.update({"episode_id": f"p5d2_complete_{sample_id}_seed{seed:02d}", "motion_id": ordinal, "motion_npz": str(dst), "library_motion_npz": str(dst), "canonical_motion_npz": str(dst), "strike_target": target_from_goal(seed_target, goal), "canonical_goal_10d": goal})
        base["p5d2_bank"] = {"category": category, "split": split, "sample_id": sample_id, "source_seed_motion_id": seed, "teacher_approved": False, "physics_qualified": False, "runtime_safety_replayed": False, "prior_compatibility_audited": False, "continuity_audited": False, "anchor": anchor}
        base["reference_contract"] = {"schema": "p5_complete_reference/v1", "coordinate_frame": "base_heading_b0", "tcp_contract": "P1 canonical TCP", "hit_frame": int(np.asarray(z["hit_frame"]).reshape(-1)[0]), "frames": int(z["joint_pos"].shape[0]), "post_hit_recovery_frames": int(z["joint_pos"].shape[0] - int(np.asarray(z["hit_frame"]).reshape(-1)[0]) - 1), "actual_trajectory_as_reference": False}
        selected.append(base); ordinal += 1

    # Anchor set: motion 4 is excluded; seed 1 is included only if its full
    # trajectory passes.  If it fails, fail closed and report the missing
    # anchor rather than substituting an unverified trajectory.
    for seed in SEED_IDS:
        ar = anchor_rows.get(seed)
        if ar:
            add(Path(ar["candidate_npz"]), seed, "anchor", "training", f"anchor_motion_{seed:02d}", anchor=True)

    # Two distinct seed solutions per training target = 28 local references.
    train_rows = by_split.get("training", [])
    groups = {}
    for r in train_rows:
        groups.setdefault(r["sample_id"], []).append(r)
    for sample_id in sorted(groups):
        picked = []
        for r in groups[sample_id]:
            if len(picked) >= 2: break
            key = (r["sample_id"], int(r["seed_motion_id"]))
            if key not in used:
                picked.append(r); used.add(key)
        for r in picked:
            add(Path(r["candidate_npz"]), int(r["seed_motion_id"]), "local_continuation", "training", sample_id)

    # Five explicitly programmatic, endpoint-flat phase-warp references.  They
    # preserve the hit state but exercise a different timing profile.
    for r in programmatic.get("samples", []):
        if r.get("qualification") != "TRACKER_TRAINING_ELIGIBLE":
            continue
        add(Path(r["candidate_npz"]), int(r["seed_motion_id"]), "programmatic_phase_warp", "training", r["sample_id"])

    # Five additional seed/dynamics variants, disjoint from the local slice.
    dynamics_count = 0
    for r in train_rows:
        if dynamics_count >= 5: break
        key = (r["sample_id"], int(r["seed_motion_id"]))
        if key in used: continue
        used.add(key)
        add(Path(r["candidate_npz"]), int(r["seed_motion_id"]), "multi_seed_dynamics_variant", "training", r["sample_id"])
        dynamics_count += 1

    def one_per_sample(split: str, category: str, out_split: str, limit: int) -> None:
        count = 0
        for r in by_split.get(split, []):
            if count >= limit: break
            key = (r["sample_id"], int(r["seed_motion_id"]))
            if key in used: continue
            used.add(key); add(Path(r["candidate_npz"]), int(r["seed_motion_id"]), category, out_split, r["sample_id"]); count += 1

    one_per_sample("validation", "validation", "validation", 9)
    one_per_sample("bridge_holdout", "bridge_holdout", "contiguous_holdout", 10)
    one_per_sample("boundary_holdout", "boundary_holdout", "ood_audit", 10)

    excluded = {"4": "user_requested_motion4_excluded", "1": "no_complete_full_trajectory_soft_margin_pass" if 1 not in anchor_rows else None}
    excluded = {k: v for k, v in excluded.items() if v}
    payload = {"schema_version": "p5d2_complete_runtime_reference_bank/v1", "status": "offline_complete_pending_runtime_audit", "training_role": "p5d2_dynamic_tracker_reference_only", "teacher_data": False, "physics_qualified": False, "canonical_goal_contract": "canonical_goal_10d/v1", "coordinate_contract": {"source": "canonical_base_heading_b0", "runtime": "P1_world", "root_anchor_w_m": WORLD_ANCHOR.tolist(), "root_heading_w_rad": 0.0}, "source_complete_manifest": str(COMPLETE.resolve()), "source_anchor_manifest": str(ANCHORS.resolve()), "source_programmatic_manifest": str(PROGRAMMATIC.resolve()) if PROGRAMMATIC.exists() else None, "selection_counts": {"all": len(selected), "training": sum(1 for x in selected if x["p5d2_bank"]["split"] == "training"), "validation": sum(1 for x in selected if x["p5d2_bank"]["split"] == "validation"), "bridge_holdout": sum(1 for x in selected if x["p5d2_bank"]["split"] == "contiguous_holdout"), "boundary_ood": sum(1 for x in selected if x["p5d2_bank"]["split"] == "ood_audit")}, "excluded_motion_ids": excluded, "motions": selected}
    (OUT / "p5d2_complete_all_reference_bank_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    for split, fn in [("training", "p5d2_complete_train_manifest.json"), ("validation", "p5d2_complete_validation_manifest.json"), ("contiguous_holdout", "p5d2_complete_bridge_holdout_manifest.json"), ("ood_audit", "p5d2_complete_boundary_ood_manifest.json")]:
        part = copy.deepcopy(payload); part["split"] = split; part["motions"] = [x for x in selected if x["p5d2_bank"]["split"] == split or (split == "training" and x["p5d2_bank"].get("anchor"))]; part["motion_count"] = len(part["motions"])
        (OUT / fn).write_text(json.dumps(part, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selection_counts": payload["selection_counts"], "anchor_pass_ids": sorted(anchor_rows), "excluded_motion_ids": excluded, "output": str(OUT / "p5d2_complete_all_reference_bank_manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
