"""Create a uniformly time-rescaled motion-manifest candidate.

``speed_scale=1`` preserves the source duration.  For ``speed_scale<1`` the
motion is slowed down while preserving its spatial path and final pose.  The
source manifest is never modified; generated files are audit candidates only.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


def _interp_rows(values: np.ndarray, sample_index: np.ndarray) -> np.ndarray:
    src = np.arange(values.shape[0], dtype=np.float64)
    flat = values.reshape(values.shape[0], -1)
    out = np.stack([np.interp(sample_index, src, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape((sample_index.shape[0],) + values.shape[1:]).astype(np.float32)


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return (quat / np.clip(norm, 1.0e-8, None)).astype(np.float32)


def _interp_quat_wxyz(quat: np.ndarray, sample_index: np.ndarray) -> np.ndarray:
    """Normalized component interpolation; adequate for the small 50 Hz audit step."""
    out = _interp_rows(quat, sample_index)
    return _normalize_quat(out)


def _resolve_motion_path(entry: dict, manifest_dir: Path) -> Path:
    for key in ("motion_npz", "library_motion_npz"):
        value = entry.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if path.is_file():
                return path
            candidate = manifest_dir / path
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no readable motion path for {entry.get('episode_id')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--speed-scale", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 < args.speed_scale <= 1.0:
        raise SystemExit("--speed-scale must be in (0, 1]")

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    manifest_dir = args.source_manifest.parent
    output = copy.deepcopy(source_manifest)
    output["dataset_status"] = "candidate_time_scaled_execution_audit_not_training_approved"
    output["time_rescale"] = {
        "speed_scale": float(args.speed_scale),
        "duration_scale": float(1.0 / args.speed_scale),
        "spatial_targets_preserved": True,
        "strike_targets_recomputed_after_generation": False,
        "source_manifest": str(args.source_manifest),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    new_entries = []
    for entry in source_manifest.get("motions", []):
        motion_path = _resolve_motion_path(entry, manifest_dir)
        data = np.load(motion_path)
        n = int(data["joint_pos"].shape[0])
        fps = int(np.asarray(data["fps"]).reshape(-1)[0])
        hit = dict(entry.get("hit_event", {}))
        old_hit = int(hit.get("motion_hit_frame", round(0.46 * (n - 1))))
        new_n = max(2, int(round((n - 1) / args.speed_scale)) + 1)
        # Preserve the spatial strike state exactly: construct the new time
        # grid around the hit so output[new_hit] samples source[old_hit]
        # exactly. A single global linspace can move the hit by centimeters
        # after rounding, which would silently change the intended target.
        new_hit = int(round(old_hit * (new_n - 1) / max(n - 1, 1)))
        new_hit = min(max(new_hit, 1), new_n - 2)
        pre_index = np.linspace(0.0, float(old_hit), new_hit + 1, dtype=np.float64)
        post_count = new_n - new_hit
        post_index = np.linspace(float(old_hit), float(n - 1), post_count, dtype=np.float64)[1:]
        sample_index = np.concatenate([pre_index, post_index])
        dt = 1.0 / float(fps)

        joint_pos = _interp_rows(np.asarray(data["joint_pos"], dtype=np.float32), sample_index)
        body_pos = _interp_rows(np.asarray(data["body_pos_w"], dtype=np.float32), sample_index)
        body_quat = _interp_quat_wxyz(np.asarray(data["body_quat_w"], dtype=np.float32), sample_index)
        joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
        body_lin_vel = np.gradient(body_pos, dt, axis=0).astype(np.float32)
        body_ang_vel = (_interp_rows(np.asarray(data["body_ang_vel_w"], dtype=np.float32), sample_index) * args.speed_scale).astype(np.float32)

        episode_id = str(entry.get("episode_id", motion_path.stem))
        output_path = args.output_root / f"{episode_id}.npz"
        np.savez_compressed(
            output_path,
            fps=np.asarray([fps], dtype=np.int64),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=body_lin_vel,
            body_ang_vel_w=body_ang_vel,
        )

        new_entry = copy.deepcopy(entry)
        new_entry["motion_npz"] = str(output_path.resolve())
        new_entry.pop("library_motion_npz", None)
        new_entry["fps"] = fps
        new_entry["joint_pos_shape"] = list(joint_pos.shape)
        new_entry["body_pos_w_shape"] = list(body_pos.shape)
        hit["motion_hit_frame"] = new_hit
        hit["hit_time_from_start_s"] = float(new_hit / fps)
        hit["time_rescaled_from_speed_scale"] = float(args.speed_scale)
        new_entry["hit_event"] = hit
        new_entries.append(new_entry)

    output["motions"] = new_entries
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output_manifest} ({len(new_entries)} motion(s), speed_scale={args.speed_scale})")


if __name__ == "__main__":
    main()
