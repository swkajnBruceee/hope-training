#!/usr/bin/env python3
"""Turn P5 offline hit candidates into complete recoverable references.

The old candidates stopped at the hit and appended a zero-velocity tail. This
tool keeps the canonical pre-hit trajectory, then adds a 1-second quintic
follow-through/recovery to the seed READY pose. It regenerates canonical body
FK/velocities and applies the versioned offline dynamic/safety gates. No PPO or
PhysX training is started here.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from build_v27_bent_ready_motion import MujocoAudit
from build_upper_momentum_library import UrdfModel
from materialize_p4b_repaired_canonical_prior import (
    DEFAULT_URDF,
    _regenerate_body_arrays,
    _relative_body_velocity_from_joint_state,
)
from repair_canonical_motion_prior import PolicyTcpKinematics, _collision_audit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFLINE = ROOT / "eval_outputs/strike_goal_p5/p5d2_multianchor_offline_repaired_v2/manifest.json"
DEFAULT_SEEDS = ROOT / "eval_outputs/strike_goal_p5/p5d2_repaired_seed_v1/seed_manifest.json"
DEFAULT_LIMITS = ROOT / "cfg/p5_reference_dynamics_v1.json"
DEFAULT_OUT = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_offline_v1"
DEFAULT_METADATA = ROOT / "docs/a3_articulation_metadata.json"
DEFAULT_MJCF = ROOT.parents[1] / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"


def quintic_tail(q0: np.ndarray, v0: np.ndarray, a0: np.ndarray, q1: np.ndarray, dt: float, frames: int) -> tuple[np.ndarray, np.ndarray]:
    T = float(dt * frames)
    matrix = np.asarray(((T**3, T**4, T**5), (3*T**2, 4*T**3, 5*T**4), (6*T, 12*T**2, 20*T**3)), dtype=np.float64)
    coeff = np.zeros((6, q0.size), dtype=np.float64)
    coeff[0], coeff[1], coeff[2] = q0, v0, a0 / 2.0
    rhs = np.stack((q1 - (coeff[0] + coeff[1]*T + coeff[2]*T*T), -(coeff[1] + 2*coeff[2]*T), -2*coeff[2]), axis=0)
    coeff[3:] = np.linalg.solve(matrix, rhs)
    times = np.arange(1, frames + 1, dtype=np.float64) * dt
    q = sum(coeff[k] * times[:, None] ** k for k in range(6))
    dq = sum(k * coeff[k] * times[:, None] ** (k - 1) for k in range(1, 6))
    return q, dq


def finite_derivatives(q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dq = np.empty_like(q)
    dq[0] = (q[1] - q[0]) / dt
    dq[-1] = (q[-1] - q[-2]) / dt
    dq[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    ddq = np.diff(dq, axis=0) / dt
    jerk = np.diff(ddq, axis=0) / dt
    return dq, ddq, jerk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-manifest", type=Path, default=DEFAULT_OFFLINE)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    args = parser.parse_args()
    offline = json.loads(args.offline_manifest.read_text())
    seeds = json.loads(args.seed_manifest.read_text())
    seed_rows = {int(e["motion_id"]): e for e in seeds["motions"]}
    limits = json.loads(args.limits.read_text())
    fps = float(limits["fps_hz"]); dt = 1.0 / fps; recovery_frames = int(limits["recovery_tail_frames"])
    metadata = json.loads(args.metadata.read_text())
    joint_names = list(metadata["joint_names"]); body_names = list(metadata["body_names"])
    model = UrdfModel(DEFAULT_URDF)
    kin = PolicyTcpKinematics(args.mjcf, joint_names)
    collision = MujocoAudit(args.mjcf, joint_names)
    out_motion = args.output_dir / "canonical_motion_npz"; out_motion.mkdir(parents=True, exist_ok=True)
    rows = []; accepted_count = 0; rejected_count = 0

    for sample in offline["samples"]:
        for attempt in sample.get("seed_attempts", []):
            if attempt.get("qualification") != "PENDING_PHYSX" or not attempt.get("candidate_npz"):
                continue
            seed_id = int(attempt["seed_motion_id"]); candidate_path = args.offline_manifest.parent / attempt["candidate_npz"]
            candidate = np.load(candidate_path, allow_pickle=False)
            seed_path = Path(seed_rows[seed_id]["canonical_motion_npz"])
            seed = {k: np.asarray(v).copy() for k, v in np.load(seed_path, allow_pickle=False).items()}
            q_candidate = np.asarray(candidate["joint_pos"], dtype=np.float64)
            qd_candidate = np.asarray(candidate["joint_vel"], dtype=np.float64)
            hit = int(np.asarray(candidate["hit_frame"]).reshape(-1)[0])
            q_pre = q_candidate[: hit + 1]
            qd_pre = qd_candidate[: hit + 1]
            q_hit = q_pre[-1]; qd_hit = qd_pre[-1]
            qdd_hit = (qd_pre[-1] - qd_pre[-2]) / dt
            tail_q, tail_qd = quintic_tail(q_hit, qd_hit, qdd_hit, np.asarray(seed["joint_pos"])[0], dt, recovery_frames)
            q_full = np.vstack((q_pre, tail_q))
            qd_full = np.vstack((qd_pre, tail_qd))
            dq_fd, ddq, jerk = finite_derivatives(q_full, dt)
            soft_min = min(float(kin.soft_margin_detail(q)[0]) for q in q_full)
            collision_report = _collision_audit(collision, q_full)
            dynamic_gates = {
                "joint_velocity": float(np.max(np.abs(dq_fd))) <= float(limits["max_abs_joint_velocity_radps"]),
                "joint_acceleration": float(np.max(np.abs(ddq))) <= float(limits["max_abs_joint_acceleration_radps2"]),
                "joint_jerk": float(np.max(np.abs(jerk))) <= float(limits["max_abs_joint_jerk_radps3"]),
            }
            offline_gates = dict(attempt.get("offline_gates", {}))
            offline_gates.update({"complete_follow_through_recovery": True, "positive_soft_margin_full_trajectory": soft_min > 0.0, "collision_nonnegative_full_trajectory": collision_report["minimum_distance_m"] >= 0.0, **dynamic_gates})
            passed = all(bool(v) for v in offline_gates.values())
            body_root_pos = np.asarray(seed["body_pos_b0"][:, 0], dtype=np.float64)
            body_root_quat = np.asarray(seed["body_quat_b0_wxyz"][:, 0], dtype=np.float64)
            root_pos = np.vstack((body_root_pos[: hit + 1], np.repeat(body_root_pos[hit:hit+1], recovery_frames, axis=0)))
            root_quat = np.vstack((body_root_quat[: hit + 1], np.repeat(body_root_quat[hit:hit+1], recovery_frames, axis=0)))
            body_pos, body_quat = _regenerate_body_arrays(model, joint_names, body_names, q_full, root_pos, root_quat, fps)
            body_lin, body_ang = _relative_body_velocity_from_joint_state(model, joint_names, body_names, q_full, qd_full, root_quat)
            stem = f"{sample['sample_id']}_seed{seed_id:02d}"
            out_path = out_motion / f"{stem}.npz"
            np.savez_compressed(out_path, fps=np.asarray([fps]), joint_pos=q_full.astype(np.float32), joint_vel=qd_full.astype(np.float32), body_pos_b0=body_pos.astype(np.float32), body_quat_b0_wxyz=body_quat.astype(np.float32), body_lin_vel_b0=body_lin.astype(np.float32), body_ang_vel_b0=body_ang.astype(np.float32), hit_frame=np.asarray([hit]), canonical_goal_position_b0_m=candidate["canonical_goal_position_b0_m"], canonical_goal_normal_b0=candidate["canonical_goal_normal_b0"], canonical_goal_linear_velocity_b0_mps=candidate["canonical_goal_linear_velocity_b0_mps"], canonical_goal_time_to_hit_s=candidate["canonical_goal_time_to_hit_s"], physics_qualified=np.asarray([False]))
            row = {"sample_id": sample["sample_id"], "split": sample["split"], "seed_motion_id": seed_id, "hit_frame": hit, "candidate_npz": str(out_path.resolve()), "qualification": "TRACKER_TRAINING_ELIGIBLE" if passed else "OFFLINE_REJECTED", "offline_gates": offline_gates, "dynamics": {"max_abs_joint_velocity_radps": float(np.max(np.abs(dq_fd))), "max_abs_joint_acceleration_radps2": float(np.max(np.abs(ddq))), "max_abs_joint_jerk_radps3": float(np.max(np.abs(jerk))), "soft_margin_min_rad": soft_min, "collision": collision_report, "recovery_frames": recovery_frames}}
            rows.append(row); accepted_count += int(passed); rejected_count += int(not passed)

    payload = {"schema_version": "p5_complete_offline_references/v1", "training_role": "tracker_reference_only", "teacher_data": False, "physics_qualified": False, "limits": limits, "source_offline_manifest": str(args.offline_manifest.resolve()), "source_seed_manifest": str(args.seed_manifest.resolve()), "candidate_count": len(rows), "eligible_count": accepted_count, "rejected_count": rejected_count, "samples": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "eligible_count": accepted_count, "rejected_count": rejected_count, "output": str(args.output_dir / 'manifest.json')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
