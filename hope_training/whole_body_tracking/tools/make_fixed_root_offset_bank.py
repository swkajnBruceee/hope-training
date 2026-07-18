#!/usr/bin/env python3
"""Create a deterministic reset bank for fixed-base stance-offset probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--y", type=float, required=True, help="world/base reset lateral offset in metres")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    motions = []
    for motion in data["motions"]:
        episode_id = str(motion["episode_id"])
        joint_shape = motion.get("joint_pos_shape", [0, 31])
        joint_dof = int(joint_shape[-1])
        motions.append(
            {
                "episode_id": episode_id,
                "root_pose_delta": [args.x, args.y, args.z, 0.0, 0.0, 0.0],
                "root_velocity_delta": [0.0] * 6,
                "joint_position_delta": [0.0] * joint_dof,
            }
        )

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "motion_manifest": str(args.manifest),
        "probe": "fixed_root_offset",
        "root_offset_m": {"x": args.x, "y": args.y, "z": args.z},
        "motions": motions,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
