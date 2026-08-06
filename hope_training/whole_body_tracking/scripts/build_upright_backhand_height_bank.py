#!/usr/bin/env python3
"""Expand the upright backhand reference bank in racket height.

This is a pure-URDF, offline IK materializer.  It keeps the accepted smooth
seed trajectories, sets waist roll/pitch to zero, solves the exact policy TCP
position/normal at a set of vertical offsets, and applies the correction with
a C2 bump over 56 frames.  It deliberately remains a candidate bank until
the PhysX P0/P1 gates are run.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_upper_momentum_library import UrdfModel  # noqa: E402
from materialize_p4b_repaired_canonical_prior import (  # noqa: E402
    DEFAULT_URDF,
    MOUNT_OFFSET,
    _quat_matrix,
    _regenerate_body_arrays,
    _relative_body_velocity_from_joint_state,
)
from build_upright_backhand_reference_bank import _arm_indices, _load_motion, _smoothness  # noqa: E402


RIGHT_IK_NAMES = (
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _hard_limits(path: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(path).getroot()
    result = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            result[joint.attrib["name"]] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    return result


class TcpIK:
    def __init__(self, model: UrdfModel, names: list[str], limits: dict[str, tuple[float, float]]):
        self.model = model
        self.names = names
        self.index = {name: i for i, name in enumerate(names)}
        self.ik_indices = [self.index[name] for name in RIGHT_IK_NAMES]
        self.limits = limits

    def state(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wrist = self.model.fk(dict(zip(self.names, q, strict=True)))["right_wrist_yaw_Link"]
        return wrist[:3, 3] + wrist[:3, :3] @ MOUNT_OFFSET, wrist[:3, :3][:, 1].copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        result = np.zeros((6, len(self.ik_indices)), dtype=np.float64)
        epsilon = 1.0e-5
        for column, index in enumerate(self.ik_indices):
            plus = q.copy(); plus[index] += epsilon
            minus = q.copy(); minus[index] -= epsilon
            p_plus, n_plus = self.state(plus)
            p_minus, n_minus = self.state(minus)
            result[:3, column] = (p_plus - p_minus) / (2.0 * epsilon)
            result[3:, column] = (n_plus - n_minus) / (2.0 * epsilon)
        return result

    def solve(self, q_start: np.ndarray, target: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, float, float]:
        q = q_start.copy()
        for _ in range(120):
            position, face = self.state(q)
            residual = np.r_[target - position, 0.15 * (normal - face)]
            jacobian = self.jacobian(q)
            weighted = jacobian.copy(); weighted[3:] *= 0.15
            delta = weighted.T @ np.linalg.solve(weighted @ weighted.T + 2.0e-4 * np.eye(6), residual)
            q[self.ik_indices] += np.clip(delta, -0.03, 0.03)
            for name, (lower, upper) in self.limits.items():
                if name in self.index:
                    q[self.index[name]] = np.clip(q[self.index[name]], lower + 0.01, upper - 0.01)
            if np.linalg.norm(residual[:3]) <= 1.0e-4 and np.linalg.norm(residual[3:]) <= 1.0e-3:
                break
        position, face = self.state(q)
        return q, float(np.linalg.norm(position - target)), float(np.degrees(np.arccos(np.clip(face @ normal, -1.0, 1.0))))


def _bump(q: np.ndarray, hit: int, delta: np.ndarray, window: int) -> np.ndarray:
    result = q.copy()
    for frame in range(max(0, hit - window), min(len(q) - 1, hit + window) + 1):
        phase = (frame - hit) / float(window)
        weight = max(0.0, 1.0 - phase * phase) ** 3
        result[frame] += weight * delta
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height-offsets", type=float, nargs="+", default=[-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10])
    parser.add_argument("--max-arm-speed", type=float, default=3.60)
    parser.add_argument("--max-arm-acceleration", type=float, default=40.0)
    parser.add_argument("--max-post-hit-drop", type=float, default=0.30)
    parser.add_argument("--window-frames", type=int, default=28)
    parser.add_argument("--metadata", type=Path, default=ROOT / "docs/a3_articulation_metadata.json")
    args = parser.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    names = list(metadata["joint_names"]); body_names = list(metadata["body_names"])
    index = {name: i for i, name in enumerate(names)}
    arm = _arm_indices(names); model = UrdfModel(DEFAULT_URDF); limits = _hard_limits(DEFAULT_URDF); ik = TcpIK(model, names, limits)

    safe = []
    seen = set()
    for entry in source["motions"]:
        arrays = _load_motion(entry); q = np.asarray(arrays["joint_pos"], dtype=np.float64).copy()
        q[:, [index["waist_roll_joint"], index["waist_pitch_joint"]]] = 0.0
        fps = float(np.asarray(arrays["fps"]).reshape(-1)[0]); stats = _smoothness(q, fps, arm)
        goal = entry.get("canonical_goal_10d", {}); key = tuple(round(float(v), 5) for v in goal.get("position_b0_m", []))
        if stats[0] <= 3.25 and stats[1] <= 40.0 and stats[2] <= 0.30 and key not in seen:
            seen.add(key); safe.append((entry, arrays, q, stats))
    if len(safe) < 4:
        raise RuntimeError("not enough low-speed spatially distinct seeds")

    out_dir = args.output_dir.expanduser().resolve(); motion_dir = out_dir / "motion_npz"; motion_dir.mkdir(parents=True, exist_ok=True)
    rows = []; audit = []
    for seed_i, (entry, arrays, q_base, seed_stats) in enumerate(safe):
        fps = float(np.asarray(arrays["fps"]).reshape(-1)[0]); hit = int(np.asarray(arrays.get("hit_frame", [30])).reshape(-1)[0])
        goal_base = entry["canonical_goal_10d"]; normal = np.asarray(goal_base["normal_b0"], dtype=np.float64)
        for level_i, dz in enumerate(args.height_offsets):
            target = np.asarray(goal_base["position_b0_m"], dtype=np.float64) + np.array([0.0, 0.0, dz], dtype=np.float64)
            q_hit, pos_err, normal_err = ik.solve(q_base[hit], target, normal)
            if pos_err > 0.002 or normal_err > 0.25:
                raise RuntimeError(f"IK failed for {entry['episode_id']} dz={dz}: {pos_err} m, {normal_err} deg")
            q = _bump(q_base, hit, q_hit - q_base[hit], args.window_frames)
            q[:, [index["waist_roll_joint"], index["waist_pitch_joint"]]] = 0.0
            qd = np.gradient(q, 1.0 / fps, axis=0)
            root_pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)[:, 0]
            root_quat = np.asarray(arrays["body_quat_w"], dtype=np.float64)[:, 0]
            body_pos_w, body_quat_w = _regenerate_body_arrays(model, names, body_names, q, root_pos, root_quat, fps)
            body_lin, body_ang = _relative_body_velocity_from_joint_state(model, names, body_names, q, qd, root_quat)
            body_pos_b0 = body_pos_w - root_pos[:, None, :]
            tcp = body_pos_b0[hit, body_names.index("right_wrist_yaw_Link")] + _quat_matrix(body_quat_w[hit, body_names.index("right_wrist_yaw_Link")], "wxyz") @ MOUNT_OFFSET
            actual_err = float(np.linalg.norm(tcp - target))
            stats = _smoothness(q, fps, arm)
            hard_margin = min(min(q[:, i].min() - limits[n][0], limits[n][1] - q[:, i].max()) for n, i in index.items() if n in limits)
            passed = bool(stats[0] <= args.max_arm_speed and stats[1] <= args.max_arm_acceleration and stats[2] <= args.max_post_hit_drop and hard_margin >= 0.01 and actual_err <= 0.002)
            if not passed:
                raise RuntimeError(f"offline height gate failed {entry['episode_id']} dz={dz}: stats={stats} margin={hard_margin} err={actual_err}")
            stem = f"upright_backhand_height_{seed_i:02d}_{level_i:02d}_{dz:+.2f}".replace("+", "p").replace("-", "m")
            path = motion_dir / f"{stem}.npz"
            np.savez_compressed(path, fps=np.asarray(arrays["fps"]), joint_pos=q.astype(np.float32), joint_vel=qd.astype(np.float32), body_pos_w=body_pos_w.astype(np.float32), body_quat_w=body_quat_w.astype(np.float32), body_lin_vel_w=body_lin.astype(np.float32), body_ang_vel_w=body_ang.astype(np.float32), body_pos_b0=body_pos_b0.astype(np.float32), body_quat_b0_wxyz=body_quat_w.astype(np.float32), body_lin_vel_b0=body_lin.astype(np.float32), body_ang_vel_b0=body_ang.astype(np.float32), hit_frame=np.asarray([hit]), physics_qualified=np.asarray([False]), scene_root_anchor_w_m=np.asarray(arrays.get("scene_root_anchor_w_m", [-0.5, -0.7625, 1.04])), scene_root_heading_w_rad=np.asarray(arrays.get("scene_root_heading_w_rad", [0.0])), canonical_goal_position_b0_m=target.astype(np.float64), canonical_goal_normal_b0=normal.astype(np.float64), canonical_goal_linear_velocity_b0_mps=np.asarray(goal_base["linear_velocity_b0_mps"], dtype=np.float64), canonical_goal_time_to_hit_s=np.asarray([float(goal_base["time_to_hit_s"])], dtype=np.float64))
            out = copy.deepcopy(entry); out["episode_id"] = stem; out["motion_id"] = len(rows); out["motion_npz"] = str(path); out.pop("library_motion_npz", None); out.pop("canonical_motion_npz", None); out["canonical_goal_10d"] = {"position_b0_m": target.tolist(), "normal_b0": normal.tolist(), "linear_velocity_b0_mps": list(goal_base["linear_velocity_b0_mps"]), "time_to_hit_s": float(goal_base["time_to_hit_s"])}; out["reference_contract"] = {"schema": "p5_complete_reference/v1", "hit_frame": hit, "frames": int(q.shape[0]), "upright_torso": True, "height_offset_from_source_m": float(dz)}; out["upright_torso_reference_bank"] = True; out["height_expansion"] = {"source_episode_id": entry["episode_id"], "height_offset_m": float(dz), "ik_position_error_m": actual_err, "ik_normal_error_deg": normal_err}; out["teacher_approved"] = False; out["physics_qualified"] = False; rows.append(out)
            audit.append({"motion_id": len(rows) - 1, "source": entry["episode_id"], "height_offset_m": float(dz), "target_z_m": float(target[2]), "ik_position_error_m": actual_err, "ik_normal_error_deg": normal_err, "arm_speed_max_radps": stats[0], "arm_acceleration_max_radps2": stats[1], "post_hit_speed_drop_radps_per_frame": stats[2], "hard_joint_margin_rad": float(hard_margin), "waist_roll_abs_max_rad": float(np.max(np.abs(q[:, index["waist_roll_joint"]]))), "waist_pitch_abs_max_rad": float(np.max(np.abs(q[:, index["waist_pitch_joint"]]))), "passed_offline_gates": passed})

    payload = {"schema_version": "upright_backhand_height_reference_bank/v1", "status": "candidate_only_physx_replay_required", "training_role": "reference_candidates_not_training_approved", "teacher_approved": False, "physics_qualified": False, "source_manifest": str(source_path), "motion_count": len(rows), "upright_torso_contract": {"waist_yaw": "preserved", "waist_roll": 0.0, "waist_pitch": 0.0, "allow_forward_tilt": False}, "height_offsets_m": [float(x) for x in args.height_offsets], "coverage_note": "Low-speed upright backhand references expanded by exact TCP/normal IK with a C2 56-frame bridge; PhysX P0/P1 still required.", "motions": rows}
    (out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "offline_audit.json").write_text(json.dumps({"safe_seed_count": len(safe), "safe_seed_ids": [x[0]["episode_id"] for x in safe], "generated_count": len(rows), "rows": audit}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"safe_seeds": len(safe), "generated": len(rows), "manifest": str(out_dir / "manifest.json"), "audit": str(out_dir / "offline_audit.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
