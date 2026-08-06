#!/usr/bin/env python3
"""Build a plant-consistent strike reference from continuous floating-base rollouts.

The result is a phase reference for the Stage-A leg stabilizer, not a new
ball/impact target.  It keeps the original strike target metadata for
monitoring while replacing the reference state trajectory with what the
current floating PD_STAND plant actually realized under zero residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _finite(name: str, value: np.ndarray, source: Path) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{source}: non-finite {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-bank", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prefix_manifest = json.loads(
        args.prefix_bank.joinpath("rsi_capture_manifest.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source_by_episode = {str(entry["episode_id"]): entry for entry in source_manifest["motions"]}
    frame_z_offset = float(prefix_manifest.get("manifest_frame_z_offset_m", 0.0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motions = []

    for prefix_entry in prefix_manifest["entries"]:
        episode_id = str(prefix_entry["episode_id"])
        source_entry = source_by_episode[episode_id]
        source_motion_path = Path(source_entry["motion_npz"]).expanduser()
        with np.load(args.prefix_bank / prefix_entry["state_file"], allow_pickle=False) as capture:
            with np.load(source_motion_path, allow_pickle=False) as source_motion:
                joint_pos = np.asarray(capture["joint_pos"], dtype=np.float32)
                joint_vel = np.asarray(capture["joint_vel"], dtype=np.float32)
                body_pos_w = np.asarray(capture["body_pos_w"], dtype=np.float32)
                body_quat_w = np.asarray(capture["body_quat_w"], dtype=np.float32)
                body_lin_vel_w = np.asarray(capture["body_lin_vel_w"], dtype=np.float32)
                body_ang_vel_w = np.asarray(capture["body_ang_vel_w"], dtype=np.float32)
                # Captures are world-space and include each vectorized env's
                # scene origin.  Derive that origin from the invariant manifest
                # strike target, then store the teacher motion in local space
                # because MotionCommand adds env_origins on load.
                capture_target = np.asarray(capture["racket_target_pos_w"][0], dtype=np.float32)
                source_target = np.asarray(source_entry["strike_target"]["racket_position_m"], dtype=np.float32)
                source_target[2] += frame_z_offset
                env_origin = capture_target - source_target
                body_pos_w = body_pos_w - env_origin.reshape(1, 1, 3)

                for name, value in (
                    ("joint_pos", joint_pos),
                    ("joint_vel", joint_vel),
                    ("body_pos_w", body_pos_w),
                    ("body_quat_w", body_quat_w),
                    ("body_lin_vel_w", body_lin_vel_w),
                    ("body_ang_vel_w", body_ang_vel_w),
                ):
                    _finite(name, value, args.prefix_bank / prefix_entry["state_file"])
                if joint_pos.shape[1] != 31 or body_pos_w.shape[1:] != (32, 3):
                    raise ValueError(f"{episode_id}: incompatible captured A3 state shapes")

                hit_index = int(np.argmin(np.abs(capture["time_to_strike_s"])))
                output_motion = args.output_dir / f"{episode_id}.npz"
                np.savez_compressed(
                    output_motion,
                    fps=np.asarray(int(source_entry.get("fps", 50)), dtype=np.int32),
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    body_pos_w=body_pos_w,
                    body_quat_w=body_quat_w,
                    body_lin_vel_w=body_lin_vel_w,
                    body_ang_vel_w=body_ang_vel_w,
                )

        motion_entry = dict(source_entry)
        motion_entry["motion_npz"] = output_motion.name
        motion_entry["library_motion_npz"] = output_motion.name
        motion_entry["fps"] = int(source_entry.get("fps", 50))
        motion_entry["joint_pos_shape"] = list(joint_pos.shape)
        motion_entry["body_pos_w_shape"] = list(body_pos_w.shape)
        motion_entry["hit_event"] = dict(source_entry["hit_event"])
        motion_entry["hit_event"]["motion_hit_frame"] = hit_index
        # The new teacher files already carry the source frame's +Z adapter.
        # Preserve the old monitor target in the same local frame instead of
        # relying on MotionLibraryLoader to apply that offset again.
        motion_entry["strike_target"] = dict(source_entry["strike_target"])
        motion_entry["strike_target"]["racket_position_m"] = list(
            source_entry["strike_target"]["racket_position_m"]
        )
        motion_entry["strike_target"]["racket_position_m"][2] += frame_z_offset
        motion_entry["teacher_reference"] = {
            "kind": "continuous_floating_zero_residual_realized_state",
            "source_prefix_bank": str(args.prefix_bank),
            "source_motion_manifest": str(args.source_manifest),
            "frame_z_offset_already_applied": frame_z_offset,
            "env_origin_removed": env_origin.tolist(),
            "direct_load_eligible": False,
        }
        motions.append(motion_entry)

    output_manifest = {
        "schema_version": 1,
        "stage": "strike_stabilizer_floating_teacher_reference_v1",
        "training_eligible": False,
        "reference_semantics": "continuous floating-base zero-residual realization",
        "original_strike_target_semantics": "preserved only for monitoring in Stage-A",
        "manifest_frame_z_offset_m": 0.0,
        "motions": motions,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"motions": len(motions), "manifest": str(manifest_path), "training_eligible": False},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
