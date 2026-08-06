"""Build a strike-only motion candidate with an explicit zero-velocity tail.

The source motions contain a post-contact follow-through.  This tool keeps the
motion through the configured hit frame, optionally keeps a short post-hit
window, then appends repeated final poses whose exported velocities are zero.
The source manifest and source NPZ files are never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


VELOCITY_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")
POSE_KEYS = ("joint_pos", "body_pos_w", "body_quat_w")


def _resolve(path_text: str, *bases: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _append_zero_tail(data: dict[str, np.ndarray], tail_frames: int) -> dict[str, np.ndarray]:
    if tail_frames <= 0:
        return data
    result = dict(data)
    for key in POSE_KEYS:
        value = np.asarray(data[key])
        result[key] = np.concatenate(
            [value, np.repeat(value[-1:, ...], tail_frames, axis=0)], axis=0
        )
    for key in VELOCITY_KEYS:
        value = np.asarray(data[key])
        zeros = np.zeros((tail_frames, *value.shape[1:]), dtype=value.dtype)
        result[key] = np.concatenate([value, zeros], axis=0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--post-hit-frames", type=int, default=0)
    parser.add_argument("--zero-velocity-tail-frames", type=int, default=8)
    args = parser.parse_args()
    if args.post_hit_frames < 0 or args.zero_velocity_tail_frames < 0:
        parser.error("post-hit and zero-velocity tail frame counts must be non-negative")

    source_path = args.source_manifest.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    output = copy.deepcopy(source)
    output["manifest_name"] = f"{source.get('manifest_name', output_manifest.stem)}_strike_only_v1"
    output["dataset_status"] = "strike_only_visual_candidate"
    output["strike_only_contract"] = {
        "method": "truncate_at_hit_then_zero_velocity_hold",
        "source_manifest": str(source_path),
        "post_hit_frames_kept": int(args.post_hit_frames),
        "zero_velocity_tail_frames": int(args.zero_velocity_tail_frames),
        "hit_frame_unchanged": True,
        "note": "Visual candidate only; do not train until human visual review passes.",
    }

    rows = []
    for entry in output.get("motions", []):
        source_motion = _resolve(str(entry["motion_npz"]), source_path.parent, Path.cwd())
        if not source_motion.exists():
            raise FileNotFoundError(source_motion)
        source_data = np.load(source_motion, allow_pickle=False)
        missing = [key for key in ("fps", *POSE_KEYS, *VELOCITY_KEYS) if key not in source_data]
        if missing:
            raise ValueError(f"{source_motion}: missing keys {missing}")
        data = {key: np.asarray(source_data[key]) for key in source_data.files}
        hit_frame = int(entry.get("hit_event", {}).get("motion_hit_frame", -1))
        frame_count = int(data["joint_pos"].shape[0])
        if not 0 <= hit_frame < frame_count:
            raise ValueError(f"{source_motion}: hit frame {hit_frame} outside {frame_count}")

        last_kept = min(frame_count - 1, hit_frame + int(args.post_hit_frames))
        truncated = {key: value[: last_kept + 1].copy() for key, value in data.items()}
        candidate_data = _append_zero_tail(truncated, int(args.zero_velocity_tail_frames))

        output_path = output_dir / source_motion.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **candidate_data)

        entry["motion_npz_original"] = str(source_motion)
        entry["motion_npz"] = str(output_path)
        entry["joint_pos_shape"] = list(candidate_data["joint_pos"].shape)
        entry["body_pos_w_shape"] = list(candidate_data["body_pos_w"].shape)
        entry["strike_only_contract"] = {
            "hit_frame": hit_frame,
            "last_source_frame_kept": last_kept,
            "post_hit_frames_kept": int(last_kept - hit_frame),
            "zero_velocity_tail_start": int(last_kept + 1),
            "zero_velocity_tail_frames": int(args.zero_velocity_tail_frames),
            "final_joint_vel_max_abs": float(np.max(np.abs(candidate_data["joint_vel"][-args.zero_velocity_tail_frames:])) if args.zero_velocity_tail_frames else 0.0),
            "final_body_lin_vel_max_abs": float(np.max(np.abs(candidate_data["body_lin_vel_w"][-args.zero_velocity_tail_frames:])) if args.zero_velocity_tail_frames else 0.0),
            "final_body_ang_vel_max_abs": float(np.max(np.abs(candidate_data["body_ang_vel_w"][-args.zero_velocity_tail_frames:])) if args.zero_velocity_tail_frames else 0.0),
        }
        rows.append({
            "episode_id": entry["episode_id"],
            "source_motion": str(source_motion),
            "candidate_motion": str(output_path),
            "hit_frame": hit_frame,
            "source_frames": frame_count,
            "candidate_frames": int(candidate_data["joint_pos"].shape[0]),
            "last_source_frame_kept": last_kept,
            "zero_velocity_tail_start": last_kept + 1,
            "zero_velocity_tail_frames": int(args.zero_velocity_tail_frames),
            "zero_tail_joint_vel_max_abs": entry["strike_only_contract"]["final_joint_vel_max_abs"],
            "zero_tail_body_lin_vel_max_abs": entry["strike_only_contract"]["final_body_lin_vel_max_abs"],
            "zero_tail_body_ang_vel_max_abs": entry["strike_only_contract"]["final_body_ang_vel_max_abs"],
        })

    output["strike_only_contract"]["motions"] = rows
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = output_manifest.with_name("strike_only_validation.json")
    report_path.write_text(json.dumps({"manifest": str(output_manifest), "motions": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[strike-only] wrote {output_manifest}")
    print(f"[strike-only] wrote {len(rows)} motions and {report_path}")
    for row in rows:
        print(
            f"{row['episode_id']}: hit={row['hit_frame']} "
            f"frames={row['source_frames']}->{row['candidate_frames']} "
            f"zero_tail_max(joint/lin/ang)="
            f"{row['zero_tail_joint_vel_max_abs']:.3g}/"
            f"{row['zero_tail_body_lin_vel_max_abs']:.3g}/"
            f"{row['zero_tail_body_ang_vel_max_abs']:.3g}"
        )


if __name__ == "__main__":
    main()
