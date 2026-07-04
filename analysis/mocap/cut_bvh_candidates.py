#!/usr/bin/env python3
"""Cut candidate swing windows from BVH files based on the dataset analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _split_bvh(path: Path) -> tuple[list[str], int, float, list[str]]:
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    motion_idx = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    frames_idx = motion_idx + 1
    frame_time_idx = motion_idx + 2
    frames = int(lines[frames_idx].split(":", 1)[1].strip())
    frame_time = float(lines[frame_time_idx].split(":", 1)[1].strip())
    data_start = frame_time_idx + 1
    return lines[:data_start], frames, frame_time, lines[data_start:]


def _write_clip(source: Path, dest: Path, start_s: float, end_s: float) -> dict[str, Any]:
    header, total_frames, frame_time, motion_lines = _split_bvh(source)
    fps = 1.0 / frame_time
    start_frame = max(0, int(round(start_s * fps)))
    end_frame = min(total_frames, int(round(end_s * fps)))
    if end_frame <= start_frame:
        raise ValueError(f"empty clip for {source}: {start_s}-{end_s}s")

    selected = motion_lines[start_frame:end_frame]
    header = list(header)
    # Header layout is ... MOTION, Frames, Frame Time. Replace only Frames.
    header[-2] = f"Frames:    {len(selected)}\n"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(header + selected))
    return {
        "source": str(source),
        "output": str(dest),
        "start_s": start_s,
        "end_s": end_s,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frames": len(selected),
        "fps": fps,
    }


def _csv_to_bvh_base(csv_rel: str) -> str:
    rel = Path(csv_rel)
    group = rel.parts[1]
    take = rel.stem
    return str(Path("Bvh") / group / take)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", type=Path, default=Path("analysis/mocap/DATA260703_analysis.json"))
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/DATA260703"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/mocap/clips"))
    parser.add_argument("--max-speed-mps", type=float, default=10.0)
    parser.add_argument("--pad-before", type=float, default=1.0)
    parser.add_argument("--pad-after", type=float, default=1.0)
    args = parser.parse_args()

    analysis = json.loads(args.analysis_json.read_text())
    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()

    clips = []
    skipped = []
    for item in analysis["csv"]:
        bvh_base = _csv_to_bvh_base(item["path"])
        for racket, speed_summary in item.get("racket_speed_summary", {}).items():
            peaks = speed_summary.get("top_speed_peaks", [])
            if not peaks:
                continue
            peak = peaks[0]
            speed = float(peak["speed_mps"])
            if speed > args.max_speed_mps:
                skipped.append({"csv": item["path"], "racket": racket, "reason": f"speed {speed:.2f} > max"})
                continue

            center = float(peak["time_s"])
            start_s = max(0.0, center - args.pad_before)
            end_s = center + args.pad_after
            take_stem = Path(item["path"]).stem.replace(" ", "_")
            label = f"{take_stem}_{racket}_{start_s:.2f}_{end_s:.2f}".replace(".", "p")

            for skeleton in ("001", "002"):
                source = dataset / f"{bvh_base}_Skeleton {skeleton}.bvh"
                if not source.exists():
                    skipped.append({"csv": item["path"], "racket": racket, "skeleton": skeleton, "reason": "missing BVH"})
                    continue
                dest = output_dir / f"{label}_Skeleton{skeleton}.bvh"
                clip = _write_clip(source, dest, start_s, end_s)
                clip.update({"csv": item["path"], "racket": racket, "skeleton": skeleton, "peak_speed_mps": speed})
                clips.append(clip)

    manifest = {
        "analysis_json": str(args.analysis_json),
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "max_speed_mps": args.max_speed_mps,
        "pad_before": args.pad_before,
        "pad_after": args.pad_after,
        "clips": clips,
        "skipped": skipped,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(clips)} clips")
    print(f"Skipped {len(skipped)} candidates")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
