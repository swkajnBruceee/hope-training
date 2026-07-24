"""Print the per-motion strike timing contract used by MotionCommand.

Usage:
    python scripts/audit_manifest_strike_timing.py \
        sample_motions/p2_strike_stabilizer_library_k17_v1/tracking_motion_manifest_backhand.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.expanduser().read_text())
    motions = manifest.get("motions", [])
    if not motions:
        raise SystemExit("manifest contains no motions")

    print("episode_id\tstroke\tframes\thit_frame\tphase\thit_time_s\tfps")
    for entry in motions:
        motion_path = Path(entry["motion_npz"]).expanduser()
        data = np.load(motion_path)
        frames = int(np.asarray(data["joint_pos"]).shape[0])
        fps = int(entry.get("fps", data.get("fps", 50)))
        hit_frame = int(entry.get("hit_event", {}).get("motion_hit_frame", round(0.46 * (frames - 1))))
        phase = hit_frame / max(frames - 1, 1)
        hit_time = hit_frame / max(fps, 1)
        print(
            f"{entry.get('episode_id', '')}\t{entry.get('stroke_type', '')}\t"
            f"{frames}\t{hit_frame}\t{phase:.6f}\t{hit_time:.4f}\t{fps}"
        )


if __name__ == "__main__":
    main()
