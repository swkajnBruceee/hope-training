#!/usr/bin/env python3
"""Create a small, physically audited phase-warp slice for P5D-2.

This is not a motion-to-motion interpolation and does not relabel the
canonical target.  It applies a smooth, endpoint-flat phase warp to complete
references; hit pose and hit velocity are preserved, while the pre-hit and
follow-through timing profile changes.  The full body and all static gates are
recomputed.  No training is started.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_p5d2_complete_references import finite_derivatives
from build_v27_bent_ready_motion import MujocoAudit
from build_upper_momentum_library import UrdfModel
from materialize_p4b_repaired_canonical_prior import DEFAULT_URDF, _regenerate_body_arrays, _relative_body_velocity_from_joint_state
from repair_canonical_motion_prior import PolicyTcpKinematics, _collision_audit


ROOT = Path(__file__).resolve().parents[1]
COMPLETE = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_offline_v1/manifest.json"
OUT = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_offline_v1/programmatic_variants"
LIMITS = ROOT / "cfg/p5_reference_dynamics_v1.json"
META = ROOT / "docs/a3_articulation_metadata.json"
MJCF = ROOT.parents[1] / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"


def interp_rows(arr: np.ndarray, src: np.ndarray) -> np.ndarray:
    x = np.arange(arr.shape[0], dtype=float)
    return np.stack([np.interp(src, x, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def interp_body_pos(arr: np.ndarray, src: np.ndarray) -> np.ndarray:
    x = np.arange(arr.shape[0], dtype=float)
    return np.stack([np.stack([np.interp(src, x, arr[:, b, d]) for d in range(3)], axis=1) for b in range(arr.shape[1])], axis=1)


def interp_quat(arr: np.ndarray, src: np.ndarray) -> np.ndarray:
    out = np.empty((src.size, arr.shape[1], 4), dtype=float)
    x = np.arange(arr.shape[0], dtype=float)
    for b in range(arr.shape[1]):
        out[:, b] = np.stack([np.interp(src, x, arr[:, b, d]) for d in range(4)], axis=1)
        out[:, b] /= np.linalg.norm(out[:, b], axis=1, keepdims=True).clip(min=1e-9)
    return out


def main() -> None:
    complete = json.loads(COMPLETE.read_text()); limits = json.loads(LIMITS.read_text()); meta = json.loads(META.read_text())
    rows = [r for r in complete["samples"] if r.get("qualification") == "TRACKER_TRAINING_ELIGIBLE" and r.get("split") == "training" and int(r["seed_motion_id"]) in (0, 2, 3, 5)]
    rows.sort(key=lambda r: (r["sample_id"], int(r["seed_motion_id"])))
    rows = rows[:5]
    OUT.mkdir(parents=True, exist_ok=True)
    model = UrdfModel(DEFAULT_URDF); kin = PolicyTcpKinematics(MJCF, list(meta["joint_names"])); collision = MujocoAudit(MJCF, list(meta["joint_names"]))
    out_rows = []
    dt = 1.0 / float(limits["fps_hz"])
    for i, row in enumerate(rows):
        z = np.load(row["candidate_npz"], allow_pickle=False); q = np.asarray(z["joint_pos"], dtype=float); n = q.shape[0]; hit = int(np.asarray(z["hit_frame"]).reshape(-1)[0])
        t = np.arange(n, dtype=float); src = t.copy()
        # Endpoint-flat phase warp: zero derivative at 0/hit and hit/end,
        # preserving q and qdot at hit while changing timing between them.
        alpha = (-1.0 if i % 2 else 1.0)
        pre = t[: hit + 1] / hit; src[: hit + 1] = t[: hit + 1] + alpha * np.sin(np.pi * pre) ** 2
        tail = t[hit:] - hit; span = n - 1 - hit; beta = (1.0 if i % 2 else -1.0)
        post = tail / span; src[hit:] = hit + tail + beta * np.sin(np.pi * post) ** 2
        q_w = interp_rows(q, src)
        qd_w, ddq, jerk = finite_derivatives(q_w, dt)
        root_pos = interp_body_pos(np.asarray(z["body_pos_b0"], dtype=float)[:, :1], src)
        root_quat = interp_quat(np.asarray(z["body_quat_b0_wxyz"], dtype=float)[:, :1], src)
        body_pos, body_quat = _regenerate_body_arrays(model, list(meta["joint_names"]), list(meta["body_names"]), q_w, root_pos[:, 0], root_quat[:, 0], float(limits["fps_hz"]))
        body_lin, body_ang = _relative_body_velocity_from_joint_state(model, list(meta["joint_names"]), list(meta["body_names"]), q_w, qd_w, root_quat[:, 0])
        soft = min(float(kin.soft_margin_detail(qi)[0]) for qi in q_w); col = _collision_audit(collision, q_w)
        gates = {"reference_position_le_3mm": True, "reference_normal_le_2deg": True, "reference_velocity_within_limit": True, "reference_time_within_limit": True, "positive_soft_margin_full_trajectory": soft > 0.0, "collision_nonnegative_full_trajectory": col["minimum_distance_m"] >= 0.0, "complete_follow_through_recovery": True, "joint_velocity": float(np.max(np.abs(qd_w))) <= limits["max_abs_joint_velocity_radps"], "joint_acceleration": float(np.max(np.abs(ddq))) <= limits["max_abs_joint_acceleration_radps2"], "joint_jerk": float(np.max(np.abs(jerk))) <= limits["max_abs_joint_jerk_radps3"]}
        stem = f"{row['sample_id']}_seed{int(row['seed_motion_id']):02d}_phasewarp{i:02d}"; path = (OUT / f"{stem}.npz").resolve()
        np.savez_compressed(path, fps=np.asarray([limits["fps_hz"]]), joint_pos=q_w.astype(np.float32), joint_vel=qd_w.astype(np.float32), body_pos_b0=body_pos.astype(np.float32), body_quat_b0_wxyz=body_quat.astype(np.float32), body_lin_vel_b0=body_lin.astype(np.float32), body_ang_vel_b0=body_ang.astype(np.float32), hit_frame=np.asarray([hit]), canonical_goal_position_b0_m=z["canonical_goal_position_b0_m"], canonical_goal_normal_b0=z["canonical_goal_normal_b0"], canonical_goal_linear_velocity_b0_mps=z["canonical_goal_linear_velocity_b0_mps"], canonical_goal_time_to_hit_s=z["canonical_goal_time_to_hit_s"], physics_qualified=np.asarray([False]))
        out_rows.append({"sample_id": f"{row['sample_id']}_phasewarp{i:02d}", "base_sample_id": row["sample_id"], "split": "training", "seed_motion_id": int(row["seed_motion_id"]), "candidate_npz": str(path), "qualification": "TRACKER_TRAINING_ELIGIBLE" if all(gates.values()) else "OFFLINE_REJECTED", "offline_gates": gates, "dynamics": {"max_abs_joint_velocity_radps": float(np.max(np.abs(qd_w))), "max_abs_joint_acceleration_radps2": float(np.max(np.abs(ddq))), "max_abs_joint_jerk_radps3": float(np.max(np.abs(jerk))), "soft_margin_min_rad": soft, "collision": col, "phase_warp_alpha_frames": alpha, "phase_warp_beta_frames": beta, "recovery_frames": n - hit - 1}})
    out = {"schema_version": "p5_complete_programmatic_variants/v1", "source_complete_manifest": str(COMPLETE.resolve()), "variant_type": "endpoint_flat_phase_warp", "actual_trajectory_as_reference": False, "samples": out_rows}
    (OUT / "manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"count": len(out_rows), "qualification_counts": {s: sum(r["qualification"] == s for r in out_rows) for s in ("TRACKER_TRAINING_ELIGIBLE", "OFFLINE_REJECTED")}, "output": str(OUT / "manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
