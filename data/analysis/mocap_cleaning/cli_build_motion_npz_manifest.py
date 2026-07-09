#!/usr/bin/env python3
"""Build a manifest/summary for motion NPZ outputs generated from csv_to_npz jobs."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    del _ROOT

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    jobs = json.loads(args.jobs_json.read_text(encoding="utf-8"))["jobs"]
    entries = []
    for job in jobs:
        p = Path(job["output_file"])
        z = np.load(p)
        entries.append(
            {
                "episode_id": str(job["output_name"]),
                "motion_npz": str(p),
                "fps": int(z["fps"][0]),
                "joint_pos_shape": list(z["joint_pos"].shape),
                "joint_vel_shape": list(z["joint_vel"].shape),
                "body_pos_w_shape": list(z["body_pos_w"].shape),
                "body_quat_w_shape": list(z["body_quat_w"].shape),
            }
        )

    manifest = {
        "stage": "optimized_motion_npz",
        "count": len(entries),
        "all_joint_pos_shape": sorted({tuple(x["joint_pos_shape"]) for x in entries}),
        "all_body_pos_w_shape": sorted({tuple(x["body_pos_w_shape"]) for x in entries}),
        "all_fps": sorted({x["fps"] for x in entries}),
        "entries": entries,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Optimized Motion NPZ Summary",
        "",
        f"- count: `{len(entries)}`",
        f"- fps: `{manifest['all_fps']}`",
        f"- joint_pos shape: `{manifest['all_joint_pos_shape']}`",
        f"- body_pos_w shape: `{manifest['all_body_pos_w_shape']}`",
        "",
        "## Output Directory",
        "",
        f"- `manifest`: `{args.output_manifest}`",
        "",
        "## Episodes",
        "",
    ]
    for entry in entries:
        lines.append(f"- `{entry['episode_id']}`")
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_manifest}")
    print(f"Wrote {args.output_summary}")


if __name__ == "__main__":
    main()
