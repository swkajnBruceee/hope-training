"""Create a diagnostic manifest with per-entry hit frames shifted by N frames."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--delta", type=int, required=True)
    args = parser.parse_args()
    data = json.loads(args.source.read_text(encoding="utf-8"))
    out = copy.deepcopy(data)
    for entry in out.get("motions", []):
        hit = dict(entry.get("hit_event", {}))
        old = int(hit["motion_hit_frame"])
        new = old + int(args.delta)
        hit["motion_hit_frame"] = new
        hit["hit_time_from_start_s"] = float(new / max(int(entry.get("fps", 50)), 1))
        hit["diagnostic_hit_frame_shift"] = int(args.delta)
        entry["hit_event"] = hit
    out["dataset_status"] = "diagnostic_hit_frame_shift_not_training_approved"
    out["diagnostic_hit_frame_shift"] = int(args.delta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output} (delta={args.delta})")


if __name__ == "__main__":
    main()
