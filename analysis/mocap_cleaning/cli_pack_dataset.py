#!/usr/bin/env python3
"""Pack usable CleanSample NPZ files into one training dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_FIELDS = [
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
    "hit_pos",
    "racket_pose_at_hit",
    "racket_vel_at_hit",
    "ball_in_vel",
    "ball_out_vel",
    "landing_pos",
    "dist",
    "ball_dv",
    "score",
]

SCALAR_FIELDS = ["hit_index", "hit_time", "success"]
STRING_FIELDS = ["episode_id", "stroke_type", "quality_flags_json", "source_json"]


def _read_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {name: data[name] for name in data.files}


def _ensure_same_shape(samples: list[dict[str, np.ndarray]], fields: list[str]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for field in fields:
        field_shapes = {tuple(sample[field].shape) for sample in samples}
        if len(field_shapes) != 1:
            raise ValueError(f"inconsistent shapes for {field}: {sorted(field_shapes)}")
        shapes[field] = field_shapes.pop()
    return shapes


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Packed CleanSample Dataset",
        "",
        f"Dataset: `{report['dataset_id']}`",
        f"Dataset file: `{report['dataset_path']}`",
        f"Format: `{report['format']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Source samples | {report['source_sample_count']} |",
        f"| Packed samples | {report['packed_sample_count']} |",
        f"| Skipped samples | {report['skipped_sample_count']} |",
        f"| Frames per sample | {report['frames_per_sample']} |",
        f"| FPS | {report['fps']} |",
        "",
        "## Stroke Distribution",
        "",
    ]
    for key, value in sorted(report["stroke_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Source CSV Distribution", ""])
    for key, value in sorted(report["source_csv_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Fields", ""])
    for field, shape in report["field_shapes"].items():
        lines.append(f"- `{field}`: `{shape}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("analysis/mocap_cleaning_outputs/DATA260703_max/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/mocap_cleaning_outputs/DATA260703_max/packed"))
    parser.add_argument("--dataset-id", default="DATA260703_rigidbody_max_train")
    parser.add_argument("--format", choices=("npz", "hdf5"), default="npz")
    parser.add_argument("--include-unusable", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    entries = [
        item
        for item in manifest["samples"]
        if args.include_unusable or bool(item["usable_for_training"])
    ]
    if not entries:
        raise ValueError("no samples selected for packing")

    samples = [_read_npz(item["sample_path"]) for item in entries]
    _ensure_same_shape(samples, ARRAY_FIELDS)
    _ensure_same_shape(samples, SCALAR_FIELDS)
    _ensure_same_shape(samples, STRING_FIELDS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / f"{args.dataset_id}.{args.format if args.format == 'npz' else 'hdf5'}"

    packed_arrays: dict[str, np.ndarray] = {}
    for field in ARRAY_FIELDS:
        packed_arrays[field] = np.stack([sample[field] for sample in samples], axis=0)
    for field in SCALAR_FIELDS:
        packed_arrays[field] = np.asarray([sample[field] for sample in samples])
    for field in STRING_FIELDS:
        packed_arrays[field] = np.asarray([str(sample[field]) for sample in samples])
    packed_arrays["sample_path"] = np.asarray([item["sample_path"] for item in entries])

    dataset_attrs = {
        "dataset_id": args.dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "source_dataset_id": str(manifest.get("dataset_id", "")),
        "packed_sample_count": len(samples),
        "quat_order": "xyzw",
        "position_unit": "m",
        "coordinate_frame": "motive_global_m",
        "success_encoding": "-1 unknown, 0 false, 1 true",
    }
    if args.format == "npz":
        packed_arrays["dataset_attrs_json"] = np.asarray(json.dumps(dataset_attrs, ensure_ascii=False))
        np.savez_compressed(dataset_path, **packed_arrays)
    else:
        try:
            import h5py
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "hdf5 export requires a working h5py install; use --format npz in this environment"
            ) from exc
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(dataset_path, "w") as h5:
            for key, value in dataset_attrs.items():
                h5.attrs[key] = value
            for field, values in packed_arrays.items():
                if values.dtype.kind in ("U", "O"):
                    h5.create_dataset(field, data=values.astype(object), dtype=string_dtype)
                else:
                    h5.create_dataset(field, data=values, compression="gzip", compression_opts=4)

    stroke_counts = Counter(str(sample["stroke_type"]) for sample in samples)
    source_csv_counts = Counter(json.loads(str(sample["source_json"]))["source_csv"] for sample in samples)
    first = samples[0]
    dt = float(np.nanmedian(np.diff(first["time"])))
    field_shapes = {
        field: str((len(samples), *first[field].shape))
        for field in ARRAY_FIELDS + SCALAR_FIELDS + STRING_FIELDS
    }
    report = {
        "dataset_id": args.dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "dataset_path": str(dataset_path),
        "format": args.format,
        "source_sample_count": len(manifest["samples"]),
        "packed_sample_count": len(samples),
        "skipped_sample_count": len(manifest["samples"]) - len(samples),
        "include_unusable": args.include_unusable,
        "frames_per_sample": int(len(first["time"])),
        "fps": float(1.0 / dt),
        "stroke_counts": dict(stroke_counts),
        "source_csv_counts": dict(source_csv_counts),
        "field_shapes": field_shapes,
    }
    json_path = args.output_dir / "pack_report.json"
    md_path = args.output_dir / "pack_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {dataset_path}")
    print(f"Packed {len(samples)} samples")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
