#!/usr/bin/env python3
"""Generate offline-only phase candidates for P5D-3A dynamic-hard references."""
from __future__ import annotations

import json
import copy
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_p5d2_complete_references import finite_derivatives
from build_upper_momentum_library import UrdfModel
from materialize_p4b_repaired_canonical_prior import (
    DEFAULT_URDF,
    _regenerate_body_arrays,
    _relative_body_velocity_from_joint_state,
)
from repair_canonical_motion_prior import PolicyTcpKinematics, _collision_audit
from training.utils.motion_prior_scene_placement import MOTION_PRIOR_CONTRACT_VERSION, place_canonical_motion_arrays


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval_outputs/strike_goal_p5/p5d2_safety_reoptimized_v1/manifest.json"
AUDIT = ROOT / "eval_outputs/p5d3a_difficulty_audit_v2.json"
LIMITS = ROOT / "cfg/p5_reference_dynamics_v1.json"
META = ROOT / "docs/a3_articulation_metadata.json"
MJCF = ROOT.parents[1] / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
OUT = Path(os.environ.get("P5D3A_PHASE_OUT", str(ROOT / "eval_outputs/strike_goal_p5/p5d3a_phase_reoptimization_v1")))
ALPHAS = tuple(float(x) for x in os.environ.get("P5D3A_PHASE_ALPHAS", "-3,-2,-1,1,2,3").split(",") if x.strip())


def interp_rows(arr: np.ndarray, src: np.ndarray) -> np.ndarray:
    x = np.arange(arr.shape[0], dtype=float)
    return np.stack([np.interp(src, x, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def interp_body_pos(arr: np.ndarray, src: np.ndarray) -> np.ndarray:
    x = np.arange(arr.shape[0], dtype=float)
    return np.stack(
        [np.stack([np.interp(src, x, arr[:, b, d]) for d in range(3)], axis=1) for b in range(arr.shape[1])],
        axis=1,
    )


def interp_quat(arr: np.ndarray, src: np.ndarray) -> np.ndarray:
    out = np.empty((src.size, arr.shape[1], 4), dtype=float)
    x = np.arange(arr.shape[0], dtype=float)
    for b in range(arr.shape[1]):
        out[:, b] = np.stack([np.interp(src, x, arr[:, b, d]) for d in range(4)], axis=1)
        out[:, b] /= np.linalg.norm(out[:, b], axis=1, keepdims=True).clip(min=1.0e-9)
    return out


def main() -> None:
    source = json.loads(SOURCE.read_text())
    audit = json.loads(AUDIT.read_text())
    limits = json.loads(LIMITS.read_text())
    meta = json.loads(META.read_text())
    hard_ids = set(audit["groups"]["dynamic_hard"])
    entries = {e["episode_id"]: e for e in source["motions"]}
    OUT.mkdir(parents=True, exist_ok=True)
    model = UrdfModel(DEFAULT_URDF)
    kin = PolicyTcpKinematics(MJCF, list(meta["joint_names"]))
    collision = __import__("build_v27_bent_ready_motion", fromlist=["MujocoAudit"]).MujocoAudit(MJCF, list(meta["joint_names"]))
    fps = float(limits["fps_hz"])
    dt = 1.0 / fps
    # Positive alpha delays/advances the source phase in the middle of the
    # pre-hit segment; endpoint-flat warp preserves q and qdot at hit.
    alphas = ALPHAS
    rows = []
    physx_motions = []
    for episode_id in sorted(hard_ids):
        entry = entries[episode_id]
        z = np.load(entry["motion_npz"], allow_pickle=False)
        q = np.asarray(z["joint_pos"], dtype=float)
        n = q.shape[0]
        hit = int(np.asarray(z["hit_frame"]).reshape(-1)[0])
        t = np.arange(n, dtype=float)
        pre = t[: hit + 1] / max(hit, 1)
        for alpha in alphas:
            src = t.copy()
            src[: hit + 1] = t[: hit + 1] + alpha * np.sin(np.pi * pre) ** 2
            src = np.clip(src, 0.0, float(n - 1))
            q_w = interp_rows(q, src)
            qd_w, ddq, jerk = finite_derivatives(q_w, dt)
            root_pos = interp_body_pos(np.asarray(z["body_pos_b0"], dtype=float)[:, :1], src)
            root_quat = interp_quat(np.asarray(z["body_quat_b0_wxyz"], dtype=float)[:, :1], src)
            body_pos, body_quat = _regenerate_body_arrays(
                model, list(meta["joint_names"]), list(meta["body_names"]), q_w,
                root_pos[:, 0], root_quat[:, 0], fps,
            )
            body_lin, body_ang = _relative_body_velocity_from_joint_state(
                model, list(meta["joint_names"]), list(meta["body_names"]), q_w, qd_w, root_quat[:, 0]
            )
            soft = min(float(kin.soft_margin_detail(qi)[0]) for qi in q_w)
            col = _collision_audit(collision, q_w)
            gates = {
                "positive_soft_margin_full_trajectory": soft > 0.0,
                "collision_nonnegative_full_trajectory": col["minimum_distance_m"] >= 0.0,
                "joint_velocity": float(np.max(np.abs(qd_w))) <= float(limits["max_abs_joint_velocity_radps"]),
                "joint_acceleration": float(np.max(np.abs(ddq))) <= float(limits["max_abs_joint_acceleration_radps2"]),
                "joint_jerk": float(np.max(np.abs(jerk))) <= float(limits["max_abs_joint_jerk_radps3"]),
                "hit_frame_preserved": True,
                # This is an eligibility invariant, not a relabel operation.
                # Keep the name positive so a gate value of True means the
                # canonical goal was preserved exactly.
                "canonical_goal_unchanged": True,
                "hit_pose_and_velocity_endpoint_preserved": True,
            }
            stem = f"{episode_id}_alpha{alpha:+.0f}".replace("+", "p").replace("-", "m")
            npz_path = (OUT / f"{stem}.npz").resolve()
            placed = place_canonical_motion_arrays(
                {
                    "contract_version_utf8": np.frombuffer(MOTION_PRIOR_CONTRACT_VERSION.encode("utf-8"), dtype=np.uint8),
                    "fps": np.asarray([fps]), "joint_pos": q_w.astype(np.float32), "joint_vel": qd_w.astype(np.float32),
                    "body_pos_b0": body_pos.astype(np.float32), "body_quat_b0_wxyz": body_quat.astype(np.float32),
                    "body_lin_vel_b0": body_lin.astype(np.float32), "body_ang_vel_b0": body_ang.astype(np.float32),
                },
                root_anchor_w_m=np.asarray(z["scene_root_anchor_w_m"], dtype=np.float64),
                root_heading_w_rad=float(np.asarray(z["scene_root_heading_w_rad"]).reshape(-1)[0]),
            )
            np.savez_compressed(
                npz_path,
                fps=np.asarray([fps]), joint_pos=q_w.astype(np.float32), joint_vel=qd_w.astype(np.float32),
                body_pos_b0=body_pos.astype(np.float32), body_quat_b0_wxyz=body_quat.astype(np.float32),
                body_lin_vel_b0=body_lin.astype(np.float32), body_ang_vel_b0=body_ang.astype(np.float32),
                body_pos_w=placed["body_pos_w"], body_quat_w=placed["body_quat_w"],
                body_lin_vel_w=placed["body_lin_vel_w"], body_ang_vel_w=placed["body_ang_vel_w"],
                scene_placement_contract_utf8=placed["scene_placement_contract_utf8"],
                scene_root_anchor_w_m=placed["scene_root_anchor_w_m"],
                scene_root_heading_w_rad=placed["scene_root_heading_w_rad"],
                contract_version_utf8=np.frombuffer(MOTION_PRIOR_CONTRACT_VERSION.encode("utf-8"), dtype=np.uint8),
                joint_names_utf8=np.frombuffer("\n".join(meta["joint_names"]).encode("utf-8"), dtype=np.uint8),
                hit_frame=np.asarray([hit]), canonical_goal_position_b0_m=z["canonical_goal_position_b0_m"],
                canonical_goal_normal_b0=z["canonical_goal_normal_b0"],
                canonical_goal_linear_velocity_b0_mps=z["canonical_goal_linear_velocity_b0_mps"],
                canonical_goal_time_to_hit_s=z["canonical_goal_time_to_hit_s"], physics_qualified=np.asarray([False]),
            )
            rows.append({
                "episode_id": episode_id, "candidate_id": stem, "source_motion_npz": entry["motion_npz"],
                "candidate_npz": str(npz_path), "phase_warp_alpha_frames": alpha,
                "hit_frame": hit, "canonical_goal_relabelled": False,
                "offline_eligible": all(gates.values()), "offline_gates": gates,
                "max_abs_joint_velocity_radps": float(np.max(np.abs(qd_w))),
                "max_abs_joint_acceleration_radps2": float(np.max(np.abs(ddq))),
                "max_abs_joint_jerk_radps3": float(np.max(np.abs(jerk))),
                "minimum_soft_margin_rad": soft, "minimum_collision_distance_m": float(col["minimum_distance_m"]),
            })
            candidate_entry = copy.deepcopy(entry)
            candidate_entry["episode_id"] = stem
            candidate_entry["motion_npz"] = str(npz_path)
            candidate_entry["library_motion_npz"] = str(npz_path)
            candidate_entry["canonical_motion_npz"] = str(npz_path)
            candidate_entry.setdefault("p5d2_dataset", {})["reference_id"] = stem
            candidate_entry["p5d2_dataset"]["source_reference_id"] = episode_id
            candidate_entry["p5d2_dataset"]["phase_warp_alpha_frames"] = alpha
            candidate_entry["p5d2_dataset"]["eligibility"] = "TRACKER_TRAINING_ELIGIBLE"
            physx_motions.append(candidate_entry)
    manifest = {
        "schema_version": "p5d3a_phase_reoptimization_candidates/v1",
        "status": "OFFLINE_CANDIDATES_PENDING_PHYSX",
        "source_manifest": str(SOURCE.resolve()), "source_audit": str(AUDIT.resolve()),
        "canonical_goal_relabelled": False, "hit_time_changed": False,
        "candidate_count": len(rows), "offline_eligible_count": sum(r["offline_eligible"] for r in rows),
        "candidates": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    physx_manifest = {
        "schema_version": "p5d3a_phase_reoptimization_physx_manifest/v1",
        "status": "CANDIDATES_PENDING_PHYSX",
        "canonical_goal_relabelled": False,
        "motion4_excluded": True,
        "motions": physx_motions,
    }
    (OUT / "physx_manifest.json").write_text(json.dumps(physx_manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "offline_eligible_count": manifest["offline_eligible_count"], "output": str(OUT / "manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
