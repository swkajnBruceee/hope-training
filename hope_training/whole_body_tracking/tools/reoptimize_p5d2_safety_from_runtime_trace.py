#!/usr/bin/env python3
"""Build a low-dimensional safety-aware P5D-2 reoptimization candidate.

The deployed zero-residual replay provides the exact joint safety projection
after the frozen model_900 prior is composed with each reference.  We fit that
projection as a smooth post-hit correction in the upper-body trajectory (the
hit frame itself is held fixed so canonical p/n/v/t labels are untouched),
regenerate all body FK/velocity arrays, and emit a new manifest.  This is an
offline first iteration; the emitted package must be replayed through the
same PhysX safety filter before promotion.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from build_upper_momentum_library import UrdfModel, _rotation_log
from materialize_p4b_repaired_canonical_prior import (
    _quat_matrix,
    _regenerate_body_arrays,
    _relative_body_velocity_from_joint_state,
)

ROOT = Path(__file__).resolve().parents[1]
WORLD_ANCHOR = np.asarray((-0.5, -0.7625, 1.04), dtype=np.float64)
JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
BODY_NAMES = [
    "pelvis_link", "left_hip_pitch_Link", "right_hip_pitch_Link", "waist_yaw_Link",
    "left_hip_roll_Link", "right_hip_roll_Link", "waist_roll_Link", "left_hip_yaw_Link",
    "right_hip_yaw_Link", "torso_Link", "left_knee_Link", "right_knee_Link",
    "head_yaw_Link", "left_shoulder_pitch_Link", "right_shoulder_pitch_Link",
    "left_ankle_pitch_Link", "right_ankle_pitch_Link", "head_pitch_Link",
    "left_shoulder_roll_Link", "right_shoulder_roll_Link", "left_ankle_roll_Link",
    "right_ankle_roll_Link", "left_shoulder_yaw_Link", "right_shoulder_yaw_Link",
    "left_elbow_Link", "right_elbow_Link", "left_wrist_roll_Link", "right_wrist_roll_Link",
    "left_wrist_pitch_Link", "right_wrist_pitch_Link", "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
]
UPPER_Q_INDEX = [2, 5, 8, 13, 18, 22, 24, 26, 28, 30]


def _smooth(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0:
        return values
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / float(2 * radius + 1)
    out = np.zeros_like(values)
    for j in range(values.shape[1]):
        out[:, j] = np.convolve(values[:, j], kernel, mode="same")
    return out


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = (values[1] - values[0]) * fps
    out[-1] = (values[-1] - values[-2]) * fps
    out[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-manifest", required=True)
    ap.add_argument("--urdf", default=str(ROOT / "training/assets/agibot_a3/urdf/model.urdf"))
    args = ap.parse_args()
    bank_path = Path(args.bank).resolve()
    queue = json.loads(Path(args.queue).resolve().read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    trace = np.load(Path(args.trace).resolve(), allow_pickle=True)
    trace_values = np.asarray(trace["trace"], dtype=np.float64)
    trace_times = np.asarray(trace["time_steps"], dtype=np.int64)
    projection = trace_values[:, :, 40:50]
    queued = {item["episode_id"] for item in queue["items"]}
    out_dir = Path(args.output_dir).resolve()
    motion_dir = out_dir / "motion_npz"
    motion_dir.mkdir(parents=True, exist_ok=True)
    model = UrdfModel(Path(args.urdf).resolve())
    output_entries = []
    records = []
    for env_idx, entry in enumerate(bank["motions"]):
        eid = entry["episode_id"]
        source = Path(entry["motion_npz"]).resolve()
        with np.load(source, allow_pickle=False) as z:
            arrays = {name: np.asarray(z[name]).copy() for name in z.files}
        q_old = np.asarray(arrays["joint_pos"], dtype=np.float64)
        frames = q_old.shape[0]
        hit = int(np.asarray(arrays["hit_frame"]).reshape(-1)[0])
        correction = np.zeros((frames, 10), dtype=np.float64)
        if eid in queued:
            for frame in range(frames):
                mask = trace_times[:, env_idx] == frame
                if np.any(mask):
                    correction[frame] = projection[mask, env_idx].mean(axis=0)
            # Keep the canonical hit state exactly unchanged.  The safety
            # projection is primarily post-hit for this bank; a tiny local
            # bridge keeps the deformation smooth without relabelling p/n/v/t.
            correction = _smooth(correction, radius=2)
            correction[hit] = 0.0
            correction[: max(0, hit - 2)] *= 0.90
            correction[hit + 1 :] *= 0.90
            correction = np.clip(correction, -0.08, 0.08)
        q_new = q_old.copy()
        q_new[:, UPPER_Q_INDEX] += correction
        q_new[hit] = q_old[hit]
        fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
        qd_old = np.asarray(arrays["joint_vel"], dtype=np.float64)
        qd_new = _finite_difference(q_new, fps)
        root_pos = np.asarray(arrays["body_pos_b0"][:, 0], dtype=np.float64)
        root_quat = np.asarray(arrays["body_quat_b0_wxyz"][:, 0], dtype=np.float64)
        body_pos, body_quat = _regenerate_body_arrays(
            model, JOINT_NAMES, BODY_NAMES, q_new, root_pos, root_quat, fps
        )
        old_rel_lin, old_rel_ang = _relative_body_velocity_from_joint_state(
            model, JOINT_NAMES, BODY_NAMES, q_old, qd_old, root_quat
        )
        new_rel_lin, new_rel_ang = _relative_body_velocity_from_joint_state(
            model, JOINT_NAMES, BODY_NAMES, q_new, qd_new, root_quat
        )
        body_lin = np.asarray(arrays["body_lin_vel_b0"], dtype=np.float64) + new_rel_lin - old_rel_lin
        body_ang = np.asarray(arrays["body_ang_vel_b0"], dtype=np.float64) + new_rel_ang - old_rel_ang
        arrays.update(
            joint_pos=q_new.astype(np.float32),
            joint_vel=qd_new.astype(np.float32),
            body_pos_b0=body_pos.astype(np.float32),
            body_quat_b0_wxyz=body_quat.astype(np.float32),
            body_lin_vel_b0=body_lin.astype(np.float32),
            body_ang_vel_b0=body_ang.astype(np.float32),
            body_pos_w=(body_pos + WORLD_ANCHOR).astype(np.float32),
            body_quat_w=body_quat.astype(np.float32),
            body_lin_vel_w=body_lin.astype(np.float32),
            body_ang_vel_w=body_ang.astype(np.float32),
        )
        out_path = motion_dir / Path(source).name
        np.savez_compressed(out_path, **arrays)
        out_entry = copy.deepcopy(entry)
        out_entry["motion_npz"] = str(out_path)
        out_entry["library_motion_npz"] = str(out_path)
        out_entry["canonical_motion_npz"] = str(out_path)
        p = np.max(np.abs(projection[:, env_idx]), axis=-1)
        out_entry["safety_reoptimization"] = {
            "schema": "p5d2_runtime_safety_reoptimization/v1",
            "status": "ITERATION_1_TRACE_PROJECTED_CORRECTION" if eid in queued else "UNCHANGED_TRANSPARENT_REFERENCE",
            "queued": eid in queued,
            "hit_frame_preserved": True,
            "canonical_goal_relabelled": False,
            "projection_before_max_rad": float(np.max(p)),
            "projection_before_mean_rad": float(np.mean(p)),
            "correction_max_rad": float(np.max(np.abs(correction))),
            "correction_rms_rad": float(np.sqrt(np.mean(correction**2))),
            "trigger_frames": [int(x) for x in np.where(p > 0.01)[0].tolist()],
            "trigger_joints": [int(x) for x in np.where(np.max(np.abs(projection[:, env_idx]), axis=0) > 0.01)[0].tolist()],
            "training_started": False,
        }
        output_entries.append(out_entry)
        records.append({"episode_id": eid, "queued": eid in queued, "projection_before_max_rad": float(np.max(p)), "correction_max_rad": float(np.max(np.abs(correction)))})
    out_manifest = copy.deepcopy(bank)
    out_manifest["schema_version"] = "p5d2_complete_runtime_reference_bank_safety_reoptimized/v1"
    out_manifest["status"] = "safety_reoptimized_pending_physx_replay"
    out_manifest["source_safety_reoptimization_queue"] = str(Path(args.queue).resolve())
    out_manifest["motions"] = output_entries
    out_manifest["safety_reoptimization"] = {
        "method": "runtime_trace_projected_upper_trajectory_correction_v1",
        "low_dimensional": True,
        "upper_joint_indices": UPPER_Q_INDEX,
        "canonical_goal_relabelled": False,
        "hit_time_changed": False,
        "runtime_replay_required": True,
        "queued_count": len(queued),
        "records": records,
    }
    output_manifest = Path(args.output_manifest).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_manifest": str(output_manifest), "queued_count": len(queued), "reference_count": len(output_entries), "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
