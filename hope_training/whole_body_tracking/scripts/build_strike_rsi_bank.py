#!/usr/bin/env python3
"""Build a phase-indexed RSI bank from qualified native strike NPZ files.

This is data preparation only.  It does not alter the simulator or authorize
training.  Each entry keeps the full retargeted state so later RSI reset code
can validate continuation without reconstructing state from a partial pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PHASES = (
    "preparation",
    "swing_acceleration",
    "pre_contact",
    "contact",
    "deceleration",
    "follow_through",
    "ready_recovery",
)


def _resolve(path: str | Path, repo_root: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    # Manifest paths are rooted at the project containing ``data``.
    candidates = (repo_root / p, repo_root.parent / p, Path.cwd() / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _phase_labels(length: int, hit_frame: int) -> np.ndarray:
    # Strike phases must be defined relative to the contact frame.  A previous
    # whole-clip normalized split mislabeled post-contact frames as acceleration.
    frame = np.arange(length)
    labels = np.full(length, 6, dtype=np.int8)
    labels[frame < max(0, hit_frame - 20)] = 0
    labels[(frame >= max(0, hit_frame - 20)) & (frame < max(0, hit_frame - 10))] = 1
    labels[(frame >= max(0, hit_frame - 10)) & (frame < hit_frame)] = 2
    labels[(frame > hit_frame) & (frame <= hit_frame + 12)] = 4
    labels[(frame > hit_frame + 12) & (frame <= hit_frame + 30)] = 5
    if 0 <= hit_frame < length:
        labels[hit_frame] = 3
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = _resolve(args.manifest, repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []

    for entry in entries:
        episode_id = str(entry["episode_id"])
        motion_path = _resolve(entry["motion_npz"], repo_root)
        with np.load(motion_path, allow_pickle=False) as data:
            required = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{motion_path}: missing {missing}")
            arrays = {key: np.asarray(data[key], dtype=np.float32) for key in required}

        hit = entry.get("hit_event", {})
        hit_frame = int(hit.get("motion_hit_frame", arrays["joint_pos"].shape[0] // 2))
        labels = _phase_labels(arrays["joint_pos"].shape[0], hit_frame)
        phase = np.clip(np.arange(arrays["joint_pos"].shape[0]) - hit_frame, -hit_frame, arrays["joint_pos"].shape[0] - hit_frame - 1).astype(np.int32)
        fps = float(entry.get("fps", hit.get("motion_fps", 50.0)))
        time_to_hit = (-phase / fps).astype(np.float32)
        out_name = f"{episode_id}.npz"
        out_path = args.output_dir / out_name
        np.savez_compressed(
            out_path,
            **arrays,
            root_pos_w=arrays["body_pos_w"][:, 0],
            root_quat_w=arrays["body_quat_w"][:, 0],
            root_lin_vel_w=arrays["body_lin_vel_w"][:, 0],
            root_ang_vel_w=arrays["body_ang_vel_w"][:, 0],
            phase_id=labels,
            frame_offset_from_hit=phase,
            time_to_hit_s=time_to_hit,
        )
        index.append({
            "episode_id": episode_id,
            "state_file": out_name,
            "source_motion": str(motion_path),
            "fps": fps,
            "num_frames": int(arrays["joint_pos"].shape[0]),
            "hit_frame": hit_frame,
            "phase_names": list(PHASES),
            "strike_target": entry.get("strike_target", {}),
        })

    payload = {
        "schema_version": 1,
        "stage": "strike_conditioned_rsi_bank_v1",
        "training_eligible": False,
        "state_semantics": "full_native_retargeted_state_with_continuation_context",
        "required_arrays": ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "phase_id", "frame_offset_from_hit", "time_to_hit_s"],
        "continuation_context": {
            "present": ["root_state", "joint_state", "phase_index"],
            "not_present_and_must_be_captured_before_training": ["previous_action", "pd_target", "actuator_internal_state", "observation_history", "action_history", "phase_accumulator", "filter_state", "foot_contact_state", "ball_state", "racket_state"]
        },
        "phase_names": list(PHASES),
        "source_manifest": str(manifest_path),
        "entries": index,
    }
    (args.output_dir / "rsi_bank_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "entries": len(index), "training_eligible": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
