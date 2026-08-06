"""Build a one-motion upright-torso backhand replay probe.

This is an evaluation-only deformation: waist yaw is preserved, while waist
roll and waist pitch are set to zero.  Body FK is regenerated in Isaac
articulation order so the replay reference is internally consistent enough for
the visual stability check.  It is not a training manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_upper_momentum_library import UrdfModel  # noqa: E402
from materialize_p4b_repaired_canonical_prior import (  # noqa: E402
    DEFAULT_URDF,
    _regenerate_body_arrays,
    _relative_body_velocity_from_joint_state,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--motion-index", type=int, default=16)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "docs/a3_articulation_metadata.json")
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    entries = source["motions"]
    if not 0 <= args.motion_index < len(entries):
        raise ValueError(f"motion index {args.motion_index} outside 0..{len(entries)-1}")
    source_entry = entries[args.motion_index]
    motion_path = Path(str(source_entry["motion_npz"])).expanduser()
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    joint_names = list(metadata["joint_names"])
    body_names = list(metadata["body_names"])
    joint_index = {name: i for i, name in enumerate(joint_names)}
    q = np.asarray(np.load(motion_path, allow_pickle=False)["joint_pos"], dtype=np.float64)
    if q.shape[1] != len(joint_names):
        raise ValueError(f"joint count mismatch: {q.shape[1]} != {len(joint_names)}")

    # Keep waist yaw (the allowed trunk rotation); remove forward/back and
    # lateral trunk lean.  The surrounding READY bridge handles entry into the
    # first upright frame without teleporting the physical robot.
    q[:, joint_index["waist_roll_joint"]] = 0.0
    q[:, joint_index["waist_pitch_joint"]] = 0.0
    fps = float(np.asarray(np.load(motion_path, allow_pickle=False)["fps"]).reshape(-1)[0])
    qd = np.gradient(q, 1.0 / fps, axis=0)

    source_data = {key: np.asarray(value).copy() for key, value in np.load(motion_path, allow_pickle=False).items()}
    root_pos = np.asarray(source_data["body_pos_w"][:, 0], dtype=np.float64)
    root_quat = np.asarray(source_data["body_quat_w"][:, 0], dtype=np.float64)
    model = UrdfModel(DEFAULT_URDF)
    body_pos, body_quat = _regenerate_body_arrays(
        model, joint_names, body_names, q, root_pos, root_quat, fps
    )
    body_lin, body_ang = _relative_body_velocity_from_joint_state(
        model, joint_names, body_names, q, qd, root_quat
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = f"upright_backhand_motion_{args.motion_index:02d}"
    output_motion = args.output_root / f"{stem}.npz"
    source_data.update(
        {
            "joint_pos": q.astype(np.float32),
            "joint_vel": qd.astype(np.float32),
            "body_pos_w": body_pos.astype(np.float32),
            "body_quat_w": body_quat.astype(np.float32),
            "body_lin_vel_w": body_lin.astype(np.float32),
            "body_ang_vel_w": body_ang.astype(np.float32),
            "physics_qualified": np.asarray([False]),
        }
    )
    np.savez_compressed(output_motion, **source_data)

    output = copy.deepcopy(source)
    output["dataset_status"] = "evaluation_only_upright_torso_probe_not_training_approved"
    output["upright_torso_probe"] = {
        "source_motion_index": int(args.motion_index),
        "waist_yaw": "preserved",
        "waist_roll": 0.0,
        "waist_pitch": 0.0,
        "purpose": "visual fall check only; ignore strike position, velocity, and precision",
    }
    entry = copy.deepcopy(source_entry)
    entry["episode_id"] = stem
    entry["motion_npz"] = str(output_motion.resolve())
    entry.pop("library_motion_npz", None)
    entry["upright_torso_probe"] = True
    output["motions"] = [entry]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output_manifest), "motion": str(output_motion), "frames": int(q.shape[0])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
