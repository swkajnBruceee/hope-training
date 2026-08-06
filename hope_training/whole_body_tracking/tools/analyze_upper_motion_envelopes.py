#!/usr/bin/env python3
"""Compare the historical K17 and current K6 upper-body reference dynamics.

This is an offline audit: it reads motion NPZ files only and never changes a
manifest or reference.  The output identifies which old motion is closest to
each current strike before attempting a legacy-compatible upper reference.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY_MANIFEST = ROOT / "sample_motions/p2_strike_stabilizer_library_k17_v1/tracking_motion_manifest.json"
CURRENT_MANIFEST = ROOT / "sample_motions/p2_data260708_backhand_strike_only_v1/manifest.json"
UPPER_JOINTS = (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
ALL_JOINTS = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
UPPER_IDS = np.asarray([ALL_JOINTS.index(name) for name in UPPER_JOINTS], dtype=np.int64)


@dataclass
class Motion:
    episode_id: str
    stroke_type: str
    fps: float
    hit_frame: int
    q: np.ndarray


def _load_manifest(path: Path) -> list[Motion]:
    data = json.loads(path.read_text())
    result = []
    for entry in data["motions"]:
        npz_path = Path(entry["motion_npz"])
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        with np.load(npz_path) as npz:
            q = np.asarray(npz["joint_pos"], dtype=np.float64)[:, UPPER_IDS]
        result.append(Motion(
            episode_id=str(entry["episode_id"]),
            stroke_type=str(entry.get("stroke_type", "unknown")),
            fps=float(entry.get("fps", 50.0)),
            hit_frame=int(entry["hit_event"]["motion_hit_frame"]),
            q=q,
        ))
    return result


def _window(motion: Motion, radius: int = 12) -> tuple[np.ndarray, int]:
    start = max(0, motion.hit_frame - radius)
    end = min(motion.q.shape[0], motion.hit_frame + radius + 1)
    return motion.q[start:end], motion.hit_frame - start


def _features(motion: Motion) -> dict:
    q, hit = _window(motion)
    dt = 1.0 / motion.fps
    qd = np.gradient(q, dt, axis=0, edge_order=1)
    qdd = np.gradient(qd, dt, axis=0, edge_order=1)
    return {
        "episode_id": motion.episode_id,
        "stroke_type": motion.stroke_type,
        "frames": int(motion.q.shape[0]),
        "hit_frame": motion.hit_frame,
        "window": {"start": motion.hit_frame - hit, "end": motion.hit_frame - hit + len(q) - 1},
        "hit_q_rad": q[hit].tolist(),
        "hit_qd_rad_s": qd[hit].tolist(),
        "excursion_rad": (q.max(axis=0) - q.min(axis=0)).tolist(),
        "peak_abs_velocity_rad_s": np.abs(qd).max(axis=0).tolist(),
        "peak_abs_acceleration_rad_s2": np.abs(qdd).max(axis=0).tolist(),
    }


def _distance(current: Motion, legacy: Motion) -> float:
    cq, ch = _window(current)
    lq, lh = _window(legacy)
    # Align by relative hit time and compare a fixed window.  The q/qd terms
    # expose both static pose and support-force timing differences.
    count = min(ch, lh, len(cq) - 1 - ch, len(lq) - 1 - lh, 10)
    c = cq[ch - count:ch + count + 1]
    l = lq[lh - count:lh + count + 1]
    cqd = np.gradient(c, 1.0 / current.fps, axis=0, edge_order=1)
    lqd = np.gradient(l, 1.0 / legacy.fps, axis=0, edge_order=1)
    return float(np.sqrt(np.mean((c - l) ** 2)) + 0.06 * np.sqrt(np.mean((cqd - lqd) ** 2)))


def _aggregate(rows: list[dict], field: str) -> dict:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return {"mean": values.mean(axis=0).tolist(), "max": values.max(axis=0).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-manifest", type=Path, default=LEGACY_MANIFEST)
    parser.add_argument("--current-manifest", type=Path, default=CURRENT_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "eval_outputs/upper_contract/motion_envelope_report.json")
    args = parser.parse_args()

    legacy = _load_manifest(args.legacy_manifest)
    current = _load_manifest(args.current_manifest)
    legacy_rows = [_features(motion) for motion in legacy]
    current_rows = [_features(motion) for motion in current]
    nearest = []
    for motion in current:
        candidates = sorted(
            ((other, _distance(motion, other)) for other in legacy), key=lambda item: item[1]
        )[:3]
        nearest.append({
            "current_episode_id": motion.episode_id,
            "nearest_legacy": [
                {"episode_id": candidate.episode_id, "stroke_type": candidate.stroke_type, "dynamic_distance": distance}
                for candidate, distance in candidates
            ],
        })

    report = {
        "purpose": "read-only upper-reference dynamic-envelope comparison",
        "upper_joint_names": list(UPPER_JOINTS),
        "legacy_manifest": str(args.legacy_manifest),
        "current_manifest": str(args.current_manifest),
        "legacy": {"motions": legacy_rows, "aggregate": {
            name: _aggregate(legacy_rows, name)
            for name in ("excursion_rad", "peak_abs_velocity_rad_s", "peak_abs_acceleration_rad_s2")
        }},
        "current": {"motions": current_rows, "aggregate": {
            name: _aggregate(current_rows, name)
            for name in ("excursion_rad", "peak_abs_velocity_rad_s", "peak_abs_acceleration_rad_s2")
        }},
        "nearest_legacy_by_current": nearest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[upper-envelope] wrote {args.output}")
    for item in nearest:
        best = item["nearest_legacy"][0]
        print(f"[upper-envelope] {item['current_episode_id']} -> {best['episode_id']} distance={best['dynamic_distance']:.4f}")


if __name__ == "__main__":
    main()
