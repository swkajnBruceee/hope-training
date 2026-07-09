#!/usr/bin/env python3
"""Apply per-CSV table transforms to a packed CleanSample dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


POSITION_FIELDS = [
    "ball_pos",
    "racket_pos",
    "body_center",
    "hit_pos",
    "landing_pos",
]

VECTOR_FIELDS = [
    "ball_vel",
    "racket_vel",
    "racket_omega",
    "body_right_axis",
    "racket_vel_at_hit",
    "ball_in_vel",
    "ball_out_vel",
]


def _load_transforms(report_path: Path) -> dict[str, dict[str, Any]]:
    report = json.loads(report_path.read_text())
    out = {}
    for item in report["files"]:
        csv_name = Path(item["csv"]).name
        if item.get("geometry_ok") and item.get("transform"):
            out[csv_name] = item["transform"]
    return out


def _as_str_array(value: np.ndarray) -> list[str]:
    return [str(x) for x in value.tolist()]


def _transform_positions(values: np.ndarray, origin: np.ndarray, rot: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        return np.einsum("ij,ntj->nti", rot, values - origin)
    if values.ndim == 2:
        return np.einsum("ij,nj->ni", rot, values - origin)
    raise ValueError(f"unsupported position shape: {values.shape}")


def _transform_vectors(values: np.ndarray, rot: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        return np.einsum("ij,ntj->nti", rot, values)
    if values.ndim == 2:
        return np.einsum("ij,nj->ni", rot, values)
    raise ValueError(f"unsupported vector shape: {values.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/analysis/mocap_cleaning_outputs/DATA260708/packed/DATA260708_train.npz"))
    parser.add_argument(
        "--table-report",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260708/table_transforms/table_transform_report.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/analysis/mocap_cleaning_outputs/DATA260708/packed/DATA260708_train_table.npz"))
    parser.add_argument("--coordinate-frame", default="table_m")
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    arrays = {name: data[name] for name in data.files}
    transforms = _load_transforms(args.table_report)
    source_json = [json.loads(s) for s in _as_str_array(arrays["source_json"])]
    quality_flags = [json.loads(s) for s in _as_str_array(arrays["quality_flags_json"])]

    keep_indices = []
    transform_json = []
    for idx, source in enumerate(source_json):
        csv_name = Path(source["source_csv"]).name
        transform = transforms.get(csv_name)
        if transform is None:
            continue
        keep_indices.append(idx)
        transform_json.append(json.dumps(transform, ensure_ascii=False))

    if not keep_indices:
        raise ValueError("no samples have a valid table transform")
    keep = np.asarray(keep_indices, dtype=int)

    out: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        if values.shape and values.shape[0] == len(source_json):
            out[name] = values[keep]
        else:
            out[name] = values

    # Preserve original primary fields before converting them in-place.
    for field in POSITION_FIELDS + VECTOR_FIELDS:
        if field in out:
            out[f"{field}_motive"] = out[field].copy()
    if "racket_pose_at_hit" in out:
        out["racket_pose_at_hit_motive"] = out["racket_pose_at_hit"].copy()

    for out_idx, transform_text in enumerate(transform_json):
        transform = json.loads(transform_text)
        origin = np.asarray(transform["origin_motive_m"], dtype=float)
        rot = np.asarray(transform["rotation_motive_to_table"], dtype=float)
        for field in POSITION_FIELDS:
            if field in out:
                out[field][out_idx : out_idx + 1] = _transform_positions(out[field][out_idx : out_idx + 1], origin, rot)
        for field in VECTOR_FIELDS:
            if field in out:
                out[field][out_idx : out_idx + 1] = _transform_vectors(out[field][out_idx : out_idx + 1], rot)
        if "racket_pose_at_hit" in out:
            pos = out["racket_pose_at_hit"][out_idx : out_idx + 1, :3]
            out["racket_pose_at_hit"][out_idx : out_idx + 1, :3] = _transform_positions(pos, origin, rot)

    updated_quality = []
    updated_source = []
    for out_idx, src_idx in enumerate(keep_indices):
        q = dict(quality_flags[src_idx])
        q["coordinate_transform_available"] = True
        q["coordinate_frame"] = args.coordinate_frame
        q["source_coordinate_frame"] = "motive_global_m"
        q["table_transform_available"] = True
        updated_quality.append(json.dumps(q, ensure_ascii=False))

        source = dict(source_json[src_idx])
        source["table_transform_json"] = transform_json[out_idx]
        source["coordinate_frame"] = args.coordinate_frame
        source["source_coordinate_frame"] = "motive_global_m"
        updated_source.append(json.dumps(source, ensure_ascii=False))

    out["quality_flags_json"] = np.asarray(updated_quality)
    out["source_json"] = np.asarray(updated_source)
    out["table_transform_json"] = np.asarray(transform_json)
    if "dataset_attrs_json" in out:
        attrs = json.loads(str(out["dataset_attrs_json"]))
        attrs["coordinate_frame"] = args.coordinate_frame
        attrs["source_coordinate_frame"] = "motive_global_m"
        attrs["table_transform_report"] = str(args.table_report)
        attrs["table_transformed_sample_count"] = int(len(keep))
        out["dataset_attrs_json"] = np.asarray(json.dumps(attrs, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    report = {
        "source_dataset": str(args.dataset),
        "output_dataset": str(args.output),
        "table_report": str(args.table_report),
        "source_samples": int(len(source_json)),
        "table_transformed_samples": int(len(keep)),
        "dropped_without_transform": int(len(source_json) - len(keep)),
    }
    report_path = args.output.with_name(args.output.stem + "_transform_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
