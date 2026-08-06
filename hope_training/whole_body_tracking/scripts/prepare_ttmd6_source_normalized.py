#!/usr/bin/env python3
"""Normalize the approved TTMD6 pilot into a separate source-space package.

The transform is human-local and deliberately independent of A3. It preserves
raw coordinates and records every hypothesis so this output cannot be mistaken
for a retargeted or training-ready motion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_rows(path: Path, width: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            values = [float(value) for value in row]
            if not any(values):
                break
            if len(values) != width:
                raise ValueError(f"{path}: expected {width} values, got {len(values)}")
            rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def unit(value: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ValueError(f"degenerate {name} axis")
    return value / norm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("approved_manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--scale-hypothesis-m-per-unit", type=float, default=0.001)
    args = parser.parse_args()

    approved = json.loads(args.approved_manifest.read_text(encoding="utf-8"))
    output_root = args.output_root
    clip_root = output_root / "clips"
    clip_root.mkdir(parents=True, exist_ok=True)
    entries = []

    for record in approved["records"]:
        human_rows = read_rows(Path(record["human_path"]), 42)
        bat_rows = read_rows(Path(record["bat_path"]), 3)
        frame_count = min(len(human_rows), len(bat_rows))
        human_raw = human_rows[:frame_count].reshape(frame_count, 14, 3)
        paddle_raw = bat_rows[:frame_count]

        # TTMD6 source point hypothesis from the published 14-point order:
        # hips=0, head=1, left shoulder=2, right shoulder=5.
        origin = human_raw[:, 0, :]
        up = np.asarray([unit(human_raw[i, 1] - human_raw[i, 0], "up") for i in range(frame_count)])
        lateral = np.asarray([unit(human_raw[i, 5] - human_raw[i, 2], "lateral") for i in range(frame_count)])
        # Keep a right-handed frame: lateral x forward = up.
        forward = np.asarray([unit(np.cross(up[i], lateral[i]), "forward") for i in range(frame_count)])
        lateral = np.asarray([unit(np.cross(forward[i], up[i]), "lateral") for i in range(frame_count)])
        basis = np.stack([lateral, forward, up], axis=1)  # local columns in source coordinates

        human_local_raw = np.einsum("tij,tkj->tki", basis, human_raw - origin[:, None, :])
        paddle_local_raw = np.einsum("tij,tj->ti", basis, paddle_raw - origin)
        paddle_vel_raw = np.gradient(paddle_local_raw, 1.0 / args.fps, axis=0, edge_order=1)
        speed_raw = np.linalg.norm(paddle_vel_raw, axis=1)
        peak_frame = int(np.argmax(speed_raw)) if len(speed_raw) else 0
        lo = max(0, peak_frame - 5)
        hi = min(frame_count - 1, peak_frame + 5)

        stem = f"class{record['class_id']}_sample{record['sample_id']}"
        clip_path = clip_root / f"{stem}.npz"
        scale = float(args.scale_hypothesis_m_per_unit)
        np.savez_compressed(
            clip_path,
            human_raw=human_raw,
            paddle_raw=paddle_raw,
            pelvis_origin_raw=origin,
            basis_local_columns_raw=basis,
            human_local_raw=human_local_raw,
            paddle_local_raw=paddle_local_raw,
            human_local_m_hypothesis=human_local_raw * scale,
            paddle_local_m_hypothesis=paddle_local_raw * scale,
            paddle_velocity_local_mps_hypothesis=paddle_vel_raw * scale,
        )
        entries.append(
            {
                "source_id": stem,
                "class_id": record["class_id"],
                "class_label": record["class_label"],
                "sample_id": record["sample_id"],
                "group_id": record["group_id"],
                "source_human_path": record["human_path"],
                "source_bat_path": record["bat_path"],
                "normalized_npz": str(clip_path),
                "fps": args.fps,
                "active_frames": frame_count,
                "source_length_declared": record["source_length_declared"],
                "source_point_order_hypothesis": {
                    "hips": 0,
                    "head": 1,
                    "left_shoulder": 2,
                    "left_arm": 3,
                    "left_forearm": 4,
                    "right_shoulder": 5,
                    "right_arm": 6,
                    "right_forearm": 7,
                    "left_upper_leg": 8,
                    "left_leg": 9,
                    "left_foot": 10,
                    "right_upper_leg": 11,
                    "right_leg": 12,
                    "right_foot": 13,
                },
                "local_frame_definition": {
                    "origin": "hips point 0 per frame",
                    "up": "head(1) - hips(0)",
                    "lateral": "right_shoulder(5) - left_shoulder(2)",
                    "forward": "cross(up, lateral)",
                    "basis_columns": ["lateral", "forward", "up"],
                    "note": "source-local diagnostic frame; not an A3 frame",
                },
                "unit_status": "hypothesis_only",
                "unit_scale_hypothesis_m_per_raw_unit": scale,
                "axis_status": "hypothesis_only",
                "paddle_orientation_status": "missing_in_source",
                "hit_frame_status": "unassigned",
                "paddle_speed_peak_frame_candidate_only": peak_frame,
                "paddle_speed_candidate_window": [lo, hi],
                "paddle_speed_peak_mps_under_scale_hypothesis": float(speed_raw[peak_frame] * scale),
                "retarget_status": "not_started",
                "training_eligible": False,
            }
        )

    output = {
        "dataset": "TTMD6",
        "stage": "source_normalized_pilot_v0",
        "input_manifest": str(args.approved_manifest),
        "output_root": str(output_root),
        "record_count": len(entries),
        "training_artifacts_written": False,
        "a3_retarget_started": False,
        "unit_status": "hypothesis_only",
        "axis_status": "hypothesis_only",
        "hit_frame_status": "unassigned",
        "paddle_orientation_status": "missing_in_source",
        "records": entries,
    }
    manifest_path = output_root / "source_normalized_manifest.json"
    manifest_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"normalized {len(entries)} source clips -> {output_root}")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
