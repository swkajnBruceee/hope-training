#!/usr/bin/env python3
"""Profile approved TTMD6 clips without converting them to A3.

This is a source-space diagnostic only. It estimates motion scale and paddle
speed under an explicit hypothesis, but never promotes a frame to hit_event.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


EDGES = [
    (0, 1),
    (0, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7),
    (0, 8), (8, 9), (9, 10),
    (0, 11), (11, 12), (12, 13),
]


def read_rows(path: Path, width: int) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            values = [float(value) for value in row]
            if not any(values):
                break
            if len(values) != width:
                raise ValueError(f"{path}: expected {width} values, got {len(values)}")
            rows.append(values)
    return rows


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def velocity(points: list[list[float]], frame: int, fps: float) -> float:
    if len(points) < 2:
        return 0.0
    if frame <= 0:
        delta = distance(points[1], points[0])
    elif frame >= len(points) - 1:
        delta = distance(points[-1], points[-2])
    else:
        delta = distance(points[frame + 1], points[frame - 1]) / 2.0
    return delta * fps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("approved_manifest", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=120.0)
    args = parser.parse_args()

    approved = json.loads(args.approved_manifest.read_text(encoding="utf-8"))
    profiles = []
    for record in approved["records"]:
        human_rows = read_rows(Path(record["human_path"]), 42)
        bat_rows = read_rows(Path(record["bat_path"]), 3)
        frame_count = min(len(human_rows), len(bat_rows))
        human = [[row[i : i + 3] for i in range(0, 42, 3)] for row in human_rows[:frame_count]]
        bat = [row for row in bat_rows[:frame_count]]

        bone_medians = []
        for a, b in EDGES:
            bone_medians.append(median(distance(frame[a], frame[b]) for frame in human))
        bat_speed = [velocity(bat, i, args.fps) for i in range(frame_count)]
        peak_speed_frame = max(range(frame_count), key=lambda index: bat_speed[index]) if bat_speed else 0
        z_values = [point[2] for frame in human for point in frame]
        axis_ranges = [
            max(point[axis] for frame in human for point in frame)
            - min(point[axis] for frame in human for point in frame)
            for axis in range(3)
        ]
        profiles.append(
            {
                "class_id": record["class_id"],
                "class_label": record["class_label"],
                "sample_id": record["sample_id"],
                "group_id": record["group_id"],
                "fps": args.fps,
                "stored_active_frames": frame_count,
                "source_length_declared": record["source_length_declared"],
                "human_bone_length_median_raw": bone_medians,
                "axis_range_raw": axis_ranges,
                "vertical_axis_hypothesis": 2,
                "unit_scale_hypothesis_m_per_raw_unit": 0.001,
                "estimated_height_m_under_scale_hypothesis": (max(z_values) - min(z_values)) * 0.001,
                "paddle_speed_raw_per_s_peak": max(bat_speed, default=0.0),
                "paddle_speed_m_per_s_peak_under_scale_hypothesis": max(bat_speed, default=0.0) * 0.001,
                "peak_speed_frame_candidate_only": peak_speed_frame,
                "hit_frame_status": "unassigned",
                "paddle_orientation_status": "missing_in_source",
            }
        )

    output = {
        "dataset": "TTMD6",
        "input": str(args.approved_manifest),
        "purpose": "source_space_calibration_profile",
        "training_artifacts_written": False,
        "coordinate_status": "hypothesis_only",
        "unit_status": "hypothesis_only",
        "impact_status": "not_inferred",
        "orientation_status": "not_available_in_source",
        "records": profiles,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    by_class: dict[str, list[dict]] = {}
    for profile in profiles:
        by_class.setdefault(profile["class_label"], []).append(profile)
    lines = [
        "# TTMD6 Pilot Source Profile",
        "",
        "This report is source-space only. It does not create A3 joint data,",
        "hit events, racket orientation, or training artifacts.",
        "",
        f"Approved clips profiled: {len(profiles)}",
        "",
        "| Class | Clips | Active frames median | Peak paddle speed under 0.001 m/unit (m/s) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, items in sorted(by_class.items()):
        lines.append(
            f"| {label} | {len(items)} | "
            f"{median(item['stored_active_frames'] for item in items):.1f} | "
            f"{median(item['paddle_speed_m_per_s_peak_under_scale_hypothesis'] for item in items):.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Third coordinate is retained as the current vertical-axis hypothesis.",
        "- 0.001 m per raw unit is retained as a scale hypothesis only.",
        "- Peak paddle-speed frames are diagnostic candidates, not hit frames.",
        "- Paddle orientation is absent and must be constructed by the TTMD6 adapter.",
        "- No record is training-eligible after this profiling step.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_json)
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
