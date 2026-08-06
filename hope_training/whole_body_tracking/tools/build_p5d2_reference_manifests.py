#!/usr/bin/env python3
"""Build the explicit P5D-2 tracker-reference train/holdout manifests.

The P5 offline solver exports joint trajectories only.  The runtime motion
library intentionally consumes the complete P1 NPZ contract, so this tool
packages each solver result with the reviewed motion-3 body-state payload,
while replacing only joint position/velocity and the canonical strike target.
These are *reference-tracker* samples, not teacher-approved executions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "eval_outputs/strike_goal_p5/p5_oracle_iter_candidate_00041_p1"
SOURCE_MANIFEST = P1 / "manifest.json"
SOURCE_MOTION = P1 / "motion_npz/motion_03_T_010_gao01_8p47_10p47.npz"
OFFLINE_MANIFEST = ROOT / "eval_outputs/strike_goal_p5/offline_smoke_velocity_v2/manifest.json"
OFFLINE_DIR = OFFLINE_MANIFEST.parent
OUT_DIR = ROOT / "eval_outputs/strike_goal_p5/p5d2_reference_manifests"
ANCHOR = np.asarray((-0.5, -0.7625, 1.04), dtype=np.float64)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def accepted_candidates() -> dict[str, tuple[dict, Path]]:
    payload = json.loads(OFFLINE_MANIFEST.read_text(encoding="utf-8"))
    result: dict[str, tuple[dict, Path]] = {}
    for sample in payload.get("samples", []):
        accepted = [
            x for x in sample.get("seed_attempts", [])
            if x.get("qualification") == "PENDING_PHYSX" and x.get("candidate_npz")
        ]
        if len(accepted) != 1:
            raise ValueError(f"{sample['sample_id']}: expected one PENDING_PHYSX candidate")
        path = OFFLINE_DIR / str(accepted[0]["candidate_npz"])
        if not path.exists():
            raise FileNotFoundError(path)
        result[str(sample["sample_id"])] = (sample, path)
    return result


def target_from_sample(sample: dict, source_target: dict) -> dict:
    goal = sample["canonical_goal_10d"]
    out = copy.deepcopy(source_target)
    out["racket_position_m"] = (ANCHOR + np.asarray(goal["position_b0_m"], dtype=np.float64)).tolist()
    out["racket_velocity_mps"] = list(goal["linear_velocity_b0_mps"])
    out["racket_normal_w"] = list(goal["normal_b0"])
    velocity = np.asarray(out["racket_velocity_mps"], dtype=np.float64)
    out["racket_velocity_direction_w"] = (velocity / np.linalg.norm(velocity)).tolist()
    # Preserve the canonical racket-to-ball offset from P1; this does not
    # relabel the task target from the executed state.
    old_pos = np.asarray(source_target["racket_position_m"], dtype=np.float64)
    old_ball = np.asarray(source_target["ball_position_m"], dtype=np.float64)
    out["ball_position_m"] = (np.asarray(out["racket_position_m"]) + old_ball - old_pos).tolist()
    return out


def package_motion(sample_id: str, candidate: Path, source_data: dict[str, np.ndarray]) -> Path:
    target = OUT_DIR / "motion_npz" / f"{sample_id}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate_data = np.load(candidate, allow_pickle=False)
    data = {k: np.asarray(v).copy() for k, v in source_data.items()}
    data["joint_pos"] = np.asarray(candidate_data["joint_pos"], dtype=np.float32)
    data["joint_vel"] = np.asarray(candidate_data["joint_vel"], dtype=np.float32)
    data["p5d2_reference_sample_id"] = np.frombuffer(sample_id.encode("utf-8"), dtype=np.uint8)
    data["p5d2_physics_qualified"] = np.asarray((0,), dtype=np.int8)
    np.savez_compressed(target, **data)
    return target


def main() -> None:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    source_entry = next(x for x in source["motions"] if int(x["motion_id"]) == 3)
    source_data = {k: np.asarray(v) for k, v in np.load(SOURCE_MOTION, allow_pickle=False).items()}
    candidates = accepted_candidates()
    # Same-target reviewed motion-3 anchor plus three generated continuations.
    train_ids = ["motion3_anchor", "p5_pos_00041", "p5_pos_00043", "p5_pos_00046"]
    holdout_ids = ["p5_pos_00048"]
    all_ids = train_ids[1:] + holdout_ids
    packaged: dict[str, Path] = {}
    for sample_id in all_ids:
        packaged[sample_id] = package_motion(sample_id, candidates[sample_id][1], source_data)
    packaged["motion3_anchor"] = SOURCE_MOTION.resolve()

    def make_entry(sample_id: str, motion_id: int) -> dict:
        if sample_id == "motion3_anchor":
            entry = copy.deepcopy(source_entry)
            sample = None
        else:
            sample = candidates[sample_id][0]
            entry = copy.deepcopy(source_entry)
            entry["strike_target"] = target_from_sample(sample, source_entry["strike_target"])
        entry["episode_id"] = f"p5d2_{sample_id}"
        entry["motion_id"] = motion_id
        entry["motion_npz"] = str(packaged[sample_id])
        entry["library_motion_npz"] = str(packaged[sample_id])
        entry["canonical_motion_npz"] = str(packaged[sample_id])
        entry["source_motion_npz_before_canonicalization"] = str(packaged[sample_id])
        entry["p5d2_reference"] = {
            "sample_id": sample_id,
            "role": "tracker_reference_only",
            "teacher_approved": False,
            "physics_qualified": False,
            "source": "p5_offline_velocity_v2",
        }
        if sample is not None:
            entry["strike_target_b0"] = {
                "racket_position_b0_m": sample["canonical_goal_10d"]["position_b0_m"],
                "racket_normal_b0": sample["canonical_goal_10d"]["normal_b0"],
                "racket_velocity_b0_mps": sample["canonical_goal_10d"]["linear_velocity_b0_mps"],
            }
        return entry

    def manifest(name: str, ids: list[str]) -> dict:
        return {
            "manifest_name": name,
            "schema_version": "p5d2_tracker_reference_manifest/v1",
            "status": "reference_tracker_only_not_teacher_qualified",
            "training_role": "p5d2_dynamic_tracker_reference_only",
            "teacher_data": False,
            "physics_qualified": False,
            "canonical_goal_contract": "canonical_goal_10d/v1",
            "source_manifest": str(SOURCE_MANIFEST.resolve()),
            "source_manifest_sha256": sha256(SOURCE_MANIFEST),
            "offline_manifest": str(OFFLINE_MANIFEST.resolve()),
            "offline_manifest_sha256": sha256(OFFLINE_MANIFEST),
            "split": "training" if name.endswith("train") else "contiguous_holdout",
            "selection_rule": "motion3 anchor plus P5 continuation candidates; no actual-state relabeling",
            "motions": [make_entry(sample_id, i) for i, sample_id in enumerate(ids)],
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "p5d2_train_manifest.json"
    holdout_path = OUT_DIR / "p5d2_holdout_manifest.json"
    train_path.write_text(json.dumps(manifest("p5d2_tracker_reference_train", train_ids), indent=2) + "\n", encoding="utf-8")
    holdout_path.write_text(json.dumps(manifest("p5d2_tracker_reference_holdout", holdout_ids), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": str(train_path), "holdout": str(holdout_path), "train_count": len(train_ids), "holdout_count": len(holdout_ids)}, indent=2))


if __name__ == "__main__":
    main()
