"""Rebuild per-motion racket strike targets from the current motion FK state.

The motion NPZ stores the current A3 body state at every frame.  This utility
uses the same wrist-to-paddle adapter as the Isaac task and rewrites only the
kinematic strike fields in a *new* manifest.  The source manifest is never
modified.

This is intentionally a data-contract tool, not a ball/planner calibration
tool.  Ball fields are preserved from the source manifest and must be reviewed
separately before physical ball-contact training.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


MOUNT_OFFSET = np.asarray(
    (0.210211399202899, 0.0320784994676765, 0.0320358706296689), dtype=np.float64
)


def _resolve(path_text: str, manifest_path: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Workspace-relative paths are common in project manifests.
    workspace = Path(__file__).resolve().parents[3]
    return (workspace / path).resolve()


def _quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    qv = np.asarray((x, y, z), dtype=np.float64)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def _quat_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _normalise(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1.0e-8:
        raise ValueError(f"invalid zero-length vector: {v}")
    return v / norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--wrist-body-index", type=int, default=31)
    parser.add_argument("--mount-normal-axis", type=int, default=1)
    parser.add_argument("--mount-normal-sign", type=float, default=1.0)
    args = parser.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    output_path = args.output_manifest.expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    source = json.loads(source_path.read_text(encoding="utf-8"))
    motions = source.get("motions", [])
    if not motions:
        raise ValueError(f"manifest has no motions: {source_path}")
    if not 0 <= args.mount_normal_axis <= 2:
        raise ValueError("--mount-normal-axis must be 0, 1, or 2")

    output = copy.deepcopy(source)
    output["dataset_status"] = "candidate_motion_fk_recalibrated_not_yet_training_approved"
    output["strike_target_recalibration"] = {
        "method": "current_motion_npz_body_fk_wrist_mount",
        "wrist_body_index": int(args.wrist_body_index),
        "mount_offset_m": [float(x) for x in MOUNT_OFFSET],
        "normal_axis": int(args.mount_normal_axis),
        "normal_sign": float(args.mount_normal_sign),
        "source_manifest": str(source_path),
        "ball_fields_preserved": True,
        "note": "Candidate for kinematic reference-contract validation; not a ball/planner target approval.",
    }

    print("index\tepisode_id\thit_frame\told_new_pos_diff_m\tnew_pos_m\tnew_vel_mps")
    for index, entry in enumerate(output["motions"]):
        motion_path = _resolve(str(entry["motion_npz"]), source_path)
        data = np.load(motion_path)
        frame = int(entry.get("hit_event", {}).get("motion_hit_frame", 0))
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)
        body_lin = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
        body_ang = np.asarray(data["body_ang_vel_w"], dtype=np.float64)
        if not 0 <= frame < body_pos.shape[0]:
            raise ValueError(f"{motion_path}: hit frame {frame} outside {body_pos.shape[0]} frames")
        if args.wrist_body_index >= body_pos.shape[1]:
            raise ValueError(f"{motion_path}: wrist body index {args.wrist_body_index} outside {body_pos.shape[1]} bodies")

        wrist_pos = body_pos[frame, args.wrist_body_index]
        wrist_quat = body_quat[frame, args.wrist_body_index]
        offset_w = _quat_apply_wxyz(wrist_quat, MOUNT_OFFSET)
        racket_pos = wrist_pos + offset_w
        racket_vel = body_lin[frame, args.wrist_body_index] + np.cross(
            body_ang[frame, args.wrist_body_index], offset_w
        )
        rotation = _quat_matrix_wxyz(wrist_quat)
        racket_normal = _normalise(rotation[:, args.mount_normal_axis] * float(args.mount_normal_sign))
        racket_tangent = rotation[:, 0]
        old_pos = np.asarray(entry.get("strike_target", {}).get("racket_position_m", racket_pos), dtype=np.float64)

        target = dict(entry.get("strike_target", {}))
        target["racket_position_m"] = [float(x) for x in racket_pos]
        target["racket_velocity_mps"] = [float(x) for x in racket_vel]
        target["racket_normal_w"] = [float(x) for x in racket_normal]
        target["racket_tangent_w"] = [float(x) for x in racket_tangent]
        target["racket_quat_xyzw"] = [
            float(wrist_quat[1]), float(wrist_quat[2]), float(wrist_quat[3]), float(wrist_quat[0])
        ]
        entry["strike_target_original"] = copy.deepcopy(entry.get("strike_target", {}))
        entry["strike_target"] = target
        entry["strike_target_recalibration"] = {
            "source": str(motion_path),
            "hit_frame": frame,
            "method": "body_state_wrist_plus_mount_offset",
            "old_position_error_m": float(np.linalg.norm(old_pos - racket_pos)),
        }
        print(
            f"{index}\t{entry.get('episode_id', index)}\t{frame}\t{np.linalg.norm(old_pos-racket_pos):.4f}\t"
            f"{np.round(racket_pos, 4).tolist()}\t{np.round(racket_vel, 4).tolist()}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[rebuild] wrote {output_path} ({len(output['motions'])} motions)")


if __name__ == "__main__":
    main()
