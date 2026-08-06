#!/usr/bin/env python3
"""Build bounded, hit-endpoint-flat low-dimensional PhysX search candidates.

This is an offline candidate builder only.  It never changes the canonical
goal or hit frame and never starts training.  Each candidate modifies only a
small hit-window phase/amplitude component of the selected responsibility
joints; formal replay decides whether it is useful.
"""
from __future__ import annotations

import copy
import itertools
import json
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
OUT = Path(os.environ.get("P5D3A_LOW_DIM_OUT", str(ROOT / "eval_outputs/strike_goal_p5/p5d3a_lowdim_reoptimization_v1")))
MULTIFACTOR = os.environ.get("P5D3A_MULTIFACTOR", "0") == "1"
LIMITS = ROOT / "cfg/p5_reference_dynamics_v1.json"
META = ROOT / "docs/a3_articulation_metadata.json"
MJCF = ROOT.parents[1] / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"

SPECS = {
    # Two representative dynamic-hard references.
    "p5d2_complete_p5_pos_00053_seed00": ("dynamic_hard", ("right_elbow_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint")),
    "p5d2_complete_p5_pos_00056_seed00": ("dynamic_hard", ("right_elbow_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint")),
    # Two representative speed/phase candidates.
    "p5d2_complete_anchor_motion_00_seed00": ("speed_phase_candidate", ("waist_roll_joint", "right_elbow_joint")),
    "p5d2_complete_p5_pos_00000_seed03": ("speed_phase_candidate", ("waist_roll_joint", "right_elbow_joint")),
}


def interp_column(values: np.ndarray, src: np.ndarray) -> np.ndarray:
    return np.interp(src, np.arange(values.shape[0], dtype=float), values)


def endpoint_flat_phase(q: np.ndarray, hit: int, joint: int, alpha: float) -> np.ndarray:
    n = q.shape[0]
    t = np.arange(n, dtype=float)
    u = t[: hit + 1] / max(hit, 1)
    src = t[: hit + 1] + alpha * np.sin(np.pi * u) ** 2
    out = q.copy()
    out[: hit + 1, joint] = interp_column(q[:, joint], np.clip(src, 0.0, n - 1.0))
    return out


def endpoint_flat_amplitude(q: np.ndarray, hit: int, joint: int, amplitude: float) -> np.ndarray:
    out = q.copy()
    u = np.arange(hit + 1, dtype=float) / max(hit, 1)
    out[: hit + 1, joint] += amplitude * np.sin(np.pi * u) ** 2
    return out


def main() -> None:
    source = json.loads(SOURCE.read_text())
    limits = json.loads(LIMITS.read_text())
    meta = json.loads(META.read_text())
    entries = {e["episode_id"]: e for e in source["motions"]}
    missing = sorted(set(SPECS) - set(entries))
    if missing:
        raise RuntimeError(f"selected references missing from source manifest: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    model = UrdfModel(DEFAULT_URDF)
    kin = PolicyTcpKinematics(MJCF, list(meta["joint_names"]))
    collision = __import__("build_v27_bent_ready_motion", fromlist=["MujocoAudit"]).MujocoAudit(MJCF, list(meta["joint_names"]))
    fps = float(limits["fps_hz"])
    dt = 1.0 / fps
    joint_index = {name: i for i, name in enumerate(meta["joint_names"])}
    rows = []
    motions = []
    for episode_id, (difficulty, joints) in SPECS.items():
        entry = entries[episode_id]
        z = np.load(entry["motion_npz"], allow_pickle=False)
        q = np.asarray(z["joint_pos"], dtype=float)
        hit = int(np.asarray(z["hit_frame"]).reshape(-1)[0])
        variants = [("baseline", None, 0.0)]
        # One-factor-at-a-time bounded search.  The endpoint-flat warp keeps
        # q and qdot at the hit frame unchanged; amplitude also vanishes with
        # zero derivative at both ends of the pre-hit interval.
        if MULTIFACTOR:
            values = (-4.0, 0.0, 4.0)
            for combo in itertools.product(values, repeat=len(joints)):
                if all(abs(x) < 1.0e-9 for x in combo):
                    continue
                variants.append(("multi_phase", joints, combo))
        else:
            for name in joints:
                for alpha in (-4.0, -2.0, 2.0, 4.0):
                    variants.append(("phase", name, alpha))
                for amplitude in (-0.020, 0.020):
                    variants.append(("amplitude", name, amplitude))
        for kind, joint_name, value in variants:
            q_new = q.copy()
            if kind == "phase":
                q_new = endpoint_flat_phase(q_new, hit, joint_index[joint_name], value)
            elif kind == "amplitude":
                q_new = endpoint_flat_amplitude(q_new, hit, joint_index[joint_name], value)
            elif kind == "multi_phase":
                for name, alpha in zip(joint_name, value):
                    q_new = endpoint_flat_phase(q_new, hit, joint_index[name], alpha)
            qd, ddq, jerk = finite_derivatives(q_new, dt)
            root_pos = np.asarray(z["body_pos_b0"], dtype=float)[:, :1]
            root_quat = np.asarray(z["body_quat_b0_wxyz"], dtype=float)[:, :1]
            body_pos, body_quat = _regenerate_body_arrays(
                model, list(meta["joint_names"]), list(meta["body_names"]), q_new,
                root_pos[:, 0], root_quat[:, 0], fps,
            )
            body_lin, body_ang = _relative_body_velocity_from_joint_state(
                model, list(meta["joint_names"]), list(meta["body_names"]), q_new, qd, root_quat[:, 0]
            )
            soft = min(float(kin.soft_margin_detail(qi)[0]) for qi in q_new)
            col = _collision_audit(collision, q_new)
            gates = {
                "positive_soft_margin_full_trajectory": soft > 0.0,
                "collision_nonnegative_full_trajectory": col["minimum_distance_m"] >= 0.0,
                "joint_velocity": float(np.max(np.abs(qd))) <= float(limits["max_abs_joint_velocity_radps"]),
                "joint_acceleration": float(np.max(np.abs(ddq))) <= float(limits["max_abs_joint_acceleration_radps2"]),
                "joint_jerk": float(np.max(np.abs(jerk))) <= float(limits["max_abs_joint_jerk_radps3"]),
                "hit_frame_preserved": True,
                "canonical_goal_relabelled": False,
            }
            offline_eligible = all(v for key, v in gates.items() if key != "canonical_goal_relabelled")
            if kind == "baseline":
                tag = "baseline"
            elif kind == "multi_phase":
                tag = "multi_phase_" + "_".join(f"{name}_{alpha:+.0f}" for name, alpha in zip(joint_name, value))
            else:
                tag = f"{kind}_{joint_name}_{value:+.3f}"
            tag = tag.replace("+", "p").replace("-", "m").replace(".", "d")
            stem = f"{episode_id}_{tag}"
            npz_path = (OUT / f"{stem}.npz").resolve()
            placed = place_canonical_motion_arrays(
                {
                    "contract_version_utf8": np.frombuffer(MOTION_PRIOR_CONTRACT_VERSION.encode("utf-8"), dtype=np.uint8),
                    "fps": np.asarray([fps]), "joint_pos": q_new.astype(np.float32), "joint_vel": qd.astype(np.float32),
                    "body_pos_b0": body_pos.astype(np.float32), "body_quat_b0_wxyz": body_quat.astype(np.float32),
                    "body_lin_vel_b0": body_lin.astype(np.float32), "body_ang_vel_b0": body_ang.astype(np.float32),
                },
                root_anchor_w_m=np.asarray(z["scene_root_anchor_w_m"], dtype=np.float64),
                root_heading_w_rad=float(np.asarray(z["scene_root_heading_w_rad"]).reshape(-1)[0]),
            )
            np.savez_compressed(
                npz_path,
                fps=np.asarray([fps]), joint_pos=q_new.astype(np.float32), joint_vel=qd.astype(np.float32),
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
                "episode_id": episode_id, "candidate_id": stem, "difficulty": difficulty,
                "source_motion_npz": entry["motion_npz"], "candidate_npz": str(npz_path),
                "optimization_kind": kind, "optimization_joint": joint_name,
                "optimization_value": value, "hit_frame": hit, "canonical_goal_relabelled": False,
                "offline_eligible": offline_eligible, "offline_gates": gates,
                "minimum_soft_margin_rad": soft, "minimum_collision_distance_m": float(col["minimum_distance_m"]),
            })
            candidate_entry = copy.deepcopy(entry)
            candidate_entry["episode_id"] = stem
            candidate_entry["motion_npz"] = str(npz_path)
            # MotionLibraryLoader prioritizes library_motion_npz over
            # motion_npz.  Both must point at the candidate, otherwise a
            # screening manifest silently replays the original source clip.
            candidate_entry["library_motion_npz"] = str(npz_path)
            candidate_entry["canonical_motion_npz"] = str(npz_path)
            candidate_entry.setdefault("p5d2_dataset", {})["reference_id"] = stem
            candidate_entry["p5d2_dataset"].update({
                "source_reference_id": episode_id, "optimization_kind": kind,
                "optimization_joint": joint_name, "optimization_value": value,
                "eligibility": "TRACKER_TRAINING_ELIGIBLE",
            })
            motions.append(candidate_entry)
    manifest = {
        "schema_version": "p5d3a_lowdim_reoptimization_candidates/v1",
        "status": "OFFLINE_CANDIDATES_PENDING_PHYSX",
        "canonical_goal_relabelled": False, "hit_time_changed": False,
        "source_manifest": str(SOURCE.resolve()), "candidate_count": len(rows),
        "offline_eligible_count": sum(r["offline_eligible"] for r in rows), "candidates": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    physx = {
        "schema_version": "p5d3a_lowdim_reoptimization_physx_manifest/v1",
        "status": "CANDIDATES_PENDING_PHYSX", "canonical_goal_relabelled": False,
        "motion4_excluded": True, "motions": motions,
    }
    (OUT / "physx_manifest.json").write_text(json.dumps(physx, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "offline_eligible_count": manifest["offline_eligible_count"], "output": str(OUT / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
