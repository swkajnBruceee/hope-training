#!/usr/bin/env python3
"""Materialize complete READY-to-recover versions of the five seed anchors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_p5d2_complete_references import finite_derivatives, quintic_tail
from build_v27_bent_ready_motion import MujocoAudit
from build_upper_momentum_library import UrdfModel
from materialize_p4b_repaired_canonical_prior import DEFAULT_URDF, _regenerate_body_arrays, _relative_body_velocity_from_joint_state
from repair_canonical_motion_prior import PolicyTcpKinematics, _collision_audit


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "eval_outputs/strike_goal_p5/p5d2_repaired_seed_v1/seed_manifest.json"
OUT = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_offline_v1"
LIMITS = json.loads((ROOT / "cfg/p5_reference_dynamics_v1.json").read_text())
META = json.loads((ROOT / "docs/a3_articulation_metadata.json").read_text())
XML = ROOT.parents[1] / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"


def main() -> None:
    fps = float(LIMITS["fps_hz"]); dt = 1.0 / fps; n_tail = int(LIMITS["recovery_tail_frames"])
    model = UrdfModel(DEFAULT_URDF); kin = PolicyTcpKinematics(XML, META["joint_names"]); audit = MujocoAudit(XML, META["joint_names"])
    out = OUT / "canonical_motion_npz"; out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(SEED.read_text()); rows = []
    for seed in manifest["motions"]:
        sid = int(seed["motion_id"]); data = {k: np.asarray(v).copy() for k, v in np.load(seed["canonical_motion_npz"], allow_pickle=False).items()}
        hit = 30; q = data["joint_pos"][: hit + 1].astype(np.float64); qd = data["joint_vel"][: hit + 1].astype(np.float64)
        tail_q, tail_qd = quintic_tail(q[-1], qd[-1], (qd[-1] - qd[-2]) / dt, data["joint_pos"][0].astype(np.float64), dt, n_tail)
        full_q = np.vstack((q, tail_q)); full_qd = np.vstack((qd, tail_qd)); dq, ddq, jerk = finite_derivatives(full_q, dt)
        root_pos = np.asarray(data["body_pos_b0"][: hit + 1, 0], dtype=np.float64); root_quat = np.asarray(data["body_quat_b0_wxyz"][: hit + 1, 0], dtype=np.float64)
        root_pos = np.vstack((root_pos, np.repeat(root_pos[-1:], n_tail, axis=0))); root_quat = np.vstack((root_quat, np.repeat(root_quat[-1:], n_tail, axis=0)))
        body_pos, body_quat = _regenerate_body_arrays(model, META["joint_names"], META["body_names"], full_q, root_pos, root_quat, fps)
        body_lin, body_ang = _relative_body_velocity_from_joint_state(model, META["joint_names"], META["body_names"], full_q, full_qd, root_quat)
        soft = min(float(kin.soft_margin_detail(x)[0]) for x in full_q); coll = _collision_audit(audit, full_q)
        gates = {"positive_soft_margin_full_trajectory": soft > 0.0, "collision_nonnegative_full_trajectory": coll["minimum_distance_m"] >= 0.0, "joint_velocity": float(np.max(abs(dq))) <= LIMITS["max_abs_joint_velocity_radps"], "joint_acceleration": float(np.max(abs(ddq))) <= LIMITS["max_abs_joint_acceleration_radps2"], "joint_jerk": float(np.max(abs(jerk))) <= LIMITS["max_abs_joint_jerk_radps3"]}
        path = out / f"anchor_motion_{sid:02d}_complete.npz"
        np.savez_compressed(path, fps=np.asarray([fps]), joint_pos=full_q.astype(np.float32), joint_vel=full_qd.astype(np.float32), body_pos_b0=body_pos.astype(np.float32), body_quat_b0_wxyz=body_quat.astype(np.float32), body_lin_vel_b0=body_lin.astype(np.float32), body_ang_vel_b0=body_ang.astype(np.float32), hit_frame=np.asarray([hit]), physics_qualified=np.asarray([False]))
        rows.append({"seed_motion_id": sid, "candidate_npz": str(path.resolve()), "qualification": "TRACKER_TRAINING_ELIGIBLE" if all(gates.values()) else "OFFLINE_REJECTED", "gates": gates, "soft_margin_min_rad": soft, "collision": coll, "dynamics": {"max_abs_joint_velocity_radps": float(np.max(np.abs(dq))), "max_abs_joint_acceleration_radps2": float(np.max(np.abs(ddq))), "max_abs_joint_jerk_radps3": float(np.max(np.abs(jerk)))}})
    (OUT / "anchor_manifest.json").write_text(json.dumps({"schema_version": "p5_complete_anchor_references/v1", "teacher_data": False, "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
