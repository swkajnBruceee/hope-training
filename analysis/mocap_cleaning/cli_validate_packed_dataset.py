#!/usr/bin/env python3
"""Validate a packed CleanSample NPZ dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = [
    "time",
    "time_rel",
    "valid_mask",
    "ball_pos",
    "ball_vel",
    "racket_pos",
    "racket_quat",
    "racket_vel",
    "racket_omega",
    "body_center",
    "body_right_axis",
    "hit_index",
    "hit_time",
    "hit_pos",
    "racket_pose_at_hit",
    "racket_vel_at_hit",
    "ball_in_vel",
    "ball_out_vel",
    "landing_pos",
    "success",
    "stroke_type",
    "quality_flags_json",
    "source_json",
    "dist",
    "ball_dv",
    "score",
]


FINITE_FIELDS = [
    "time",
    "time_rel",
    "ball_pos",
    "ball_vel",
    "racket_pos",
    "racket_quat",
    "racket_vel",
    "racket_omega",
    "body_center",
    "body_right_axis",
    "hit_pos",
    "racket_pose_at_hit",
    "racket_vel_at_hit",
    "ball_in_vel",
    "ball_out_vel",
    "dist",
    "ball_dv",
    "score",
]


def _bool_counter(values: list[bool]) -> dict[str, int]:
    counts = Counter(values)
    return {"true": counts[True], "false": counts[False]}


def validate_dataset(path: Path, expected_fps: float, expected_hit_index: int) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    missing = [field for field in REQUIRED_FIELDS if field not in data.files]
    if missing:
        raise ValueError(f"missing required fields: {missing}")

    n = int(data["ball_pos"].shape[0])
    frames = int(data["ball_pos"].shape[1])
    shape_errors = []
    for field in REQUIRED_FIELDS:
        if data[field].shape[0] != n:
            shape_errors.append(f"{field}: first dim {data[field].shape[0]} != {n}")
    if shape_errors:
        raise ValueError("shape errors: " + "; ".join(shape_errors))

    dt = np.nanmedian(np.diff(data["time"], axis=1), axis=1)
    fps = 1.0 / dt
    finite_flags = {
        field: bool(np.isfinite(data[field]).all())
        for field in FINITE_FIELDS
    }
    quat_norm = np.linalg.norm(data["racket_quat"], axis=2)
    body_axis_norm = np.linalg.norm(data["body_right_axis"], axis=2)
    quality_flags = [json.loads(str(x)) for x in data["quality_flags_json"]]
    source_json = [json.loads(str(x)) for x in data["source_json"]]
    return {
        "dataset_path": str(path),
        "samples": n,
        "frames": frames,
        "fps_min": float(np.nanmin(fps)),
        "fps_max": float(np.nanmax(fps)),
        "fps_expected": expected_fps,
        "fps_ok": bool(np.all(np.abs(fps - expected_fps) < 1e-6)),
        "hit_index_unique": [int(x) for x in np.unique(data["hit_index"])],
        "hit_index_ok": bool(np.all(data["hit_index"] == expected_hit_index)),
        "time_rel_at_hit_max_abs": float(np.nanmax(np.abs(data["time_rel"][np.arange(n), data["hit_index"].astype(int)]))),
        "finite_fields": finite_flags,
        "all_finite_required": bool(all(finite_flags.values())),
        "quat_norm_min": float(np.nanmin(quat_norm)),
        "quat_norm_max": float(np.nanmax(quat_norm)),
        "body_axis_norm_min": float(np.nanmin(body_axis_norm)),
        "body_axis_norm_max": float(np.nanmax(body_axis_norm)),
        "success_counts": {str(int(k)): int(v) for k, v in Counter(data["success"].tolist()).items()},
        "stroke_counts": {str(k): int(v) for k, v in Counter(str(x) for x in data["stroke_type"]).items()},
        "source_csv_counts": dict(Counter(item["source_csv"] for item in source_json)),
        "usable_for_training_flags": _bool_counter([bool(item.get("usable_for_training", False)) for item in quality_flags]),
        "racket_omega_reasonable": _bool_counter([bool(item.get("racket_omega_reasonable", False)) for item in quality_flags]),
        "ball_speed_reasonable": _bool_counter([bool(item.get("ball_speed_reasonable", False)) for item in quality_flags]),
        "racket_speed_reasonable": _bool_counter([bool(item.get("racket_speed_reasonable", False)) for item in quality_flags]),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Packed Dataset Validation",
        "",
        f"Dataset: `{report['dataset_path']}`",
        "",
        "| Check | Value |",
        "|---|---:|",
        f"| Samples | {report['samples']} |",
        f"| Frames | {report['frames']} |",
        f"| FPS OK | {report['fps_ok']} |",
        f"| Hit index OK | {report['hit_index_ok']} |",
        f"| All finite required | {report['all_finite_required']} |",
        f"| Max abs time_rel at hit | {report['time_rel_at_hit_max_abs']:.3e} |",
        f"| Quaternion norm min | {report['quat_norm_min']:.6f} |",
        f"| Quaternion norm max | {report['quat_norm_max']:.6f} |",
        "",
        "## Stroke Counts",
        "",
    ]
    for key, value in sorted(report["stroke_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Source CSV Counts", ""])
    for key, value in sorted(report["source_csv_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("analysis/mocap_cleaning_outputs/DATA260703_max/packed/DATA260703_rigidbody_max_train.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/mocap_cleaning_outputs/DATA260703_max/packed"))
    parser.add_argument("--expected-fps", type=float, default=200.0)
    parser.add_argument("--expected-hit-index", type=int, default=120)
    args = parser.parse_args()

    report = validate_dataset(args.dataset, args.expected_fps, args.expected_hit_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "validation_report.json"
    md_path = args.output_dir / "validation_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Validated {report['samples']} samples")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
