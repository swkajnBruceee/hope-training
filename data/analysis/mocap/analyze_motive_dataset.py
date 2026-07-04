#!/usr/bin/env python3
"""Summarize a Motive/OptiTrack dataset and estimate racket swing windows.

The script is intentionally lightweight: it parses BVH metadata and Motive CSV
headers, then samples rigid-body positions to estimate speed peaks. It does not
modify the source dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


RACKET_NAMES = ("TennisBats01", "TennisBats02")


def _read_bvh_summary(path: Path, root: Path) -> dict[str, Any]:
    frames = None
    frame_time = None
    roots = 0
    joints = 0
    with path.open("r", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("ROOT "):
                roots += 1
            elif stripped.startswith("JOINT "):
                joints += 1
            elif stripped.startswith("Frames:"):
                frames = int(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Frame Time:"):
                frame_time = float(stripped.split(":", 1)[1].strip())
                break

    fps = (1.0 / frame_time) if frame_time else None
    duration_s = (frames * frame_time) if frames is not None and frame_time is not None else None
    return {
        "path": str(path.relative_to(root)),
        "size_mb": path.stat().st_size / 1_000_000.0,
        "frames": frames,
        "fps": fps,
        "duration_s": duration_s,
        "roots": roots,
        "joints": joints,
    }


def _meta_value(meta: list[str], key: str) -> str | None:
    try:
        return meta[meta.index(key) + 1]
    except ValueError:
        return None


def _parse_motive_header(path: Path) -> tuple[list[list[str]], dict[str, Any]]:
    with path.open("r", errors="replace") as f:
        rows = [next(f).rstrip("\n\r").split(",") for _ in range(8)]

    meta = rows[0]
    return rows, {
        "take_name": _meta_value(meta, "Take Name"),
        "capture_fps": float(_meta_value(meta, "Capture Frame Rate") or "nan"),
        "export_fps": float(_meta_value(meta, "Export Frame Rate") or "nan"),
        "frames": int(_meta_value(meta, "Total Exported Frames") or "0"),
        "units": _meta_value(meta, "Length Units"),
        "coordinate_space": _meta_value(meta, "Coordinate Space"),
    }


def _rigid_body_position_columns(rows: list[list[str]]) -> dict[str, tuple[int, int, int]]:
    columns: dict[str, dict[str, int]] = {}
    for index, (typ, name, prop, axis) in enumerate(zip(rows[2], rows[3], rows[6], rows[7])):
        if typ != "Rigid Body" or prop != "Position" or not name:
            continue
        columns.setdefault(name, {})[axis] = index

    result = {}
    for name, comps in columns.items():
        if {"X", "Y", "Z"}.issubset(comps):
            result[name] = (comps["X"], comps["Y"], comps["Z"])
    return result


def _sample_rigid_body_speeds(
    path: Path,
    columns: dict[str, tuple[int, int, int]],
    fps: float,
    sample_step: int,
    top_k: int,
) -> dict[str, dict[str, Any]]:
    if not columns:
        return {}

    last: dict[str, tuple[int, tuple[float, float, float]]] = {}
    peaks: dict[str, list[tuple[float, int]]] = {name: [] for name in columns}
    valid_samples: dict[str, int] = {name: 0 for name in columns}

    with path.open("r", errors="replace") as f:
        for _ in range(8):
            next(f)
        for data_index, line in enumerate(f):
            if data_index % sample_step != 0:
                continue
            row = line.rstrip("\n\r").split(",")
            if len(row) < 2:
                continue
            try:
                frame = int(float(row[0]))
            except ValueError:
                frame = data_index

            for name, (cx, cy, cz) in columns.items():
                try:
                    pos_mm = (float(row[cx]), float(row[cy]), float(row[cz]))
                except (ValueError, IndexError):
                    continue

                valid_samples[name] += 1
                if name in last:
                    prev_frame, prev = last[name]
                    dt = (frame - prev_frame) / fps
                    if dt > 0:
                        dist_m = math.sqrt(sum((pos_mm[i] - prev[i]) ** 2 for i in range(3))) / 1000.0
                        speed = dist_m / dt
                        peaks[name].append((speed, frame))
                last[name] = (frame, pos_mm)

    summary = {}
    for name, values in peaks.items():
        values.sort(reverse=True, key=lambda item: item[0])
        summary[name] = {
            "valid_samples": valid_samples[name],
            "top_speed_peaks": [
                {
                    "speed_mps": speed,
                    "frame": frame,
                    "time_s": frame / fps if fps else None,
                    "suggested_window_s": [
                        max(0.0, frame / fps - 0.75) if fps else None,
                        frame / fps + 0.75 if fps else None,
                    ],
                }
                for speed, frame in values[:top_k]
            ],
        }
    return summary


def _read_csv_summary(path: Path, root: Path, sample_step: int, top_k: int) -> dict[str, Any]:
    rows, meta = _parse_motive_header(path)
    type_counts: dict[str, int] = {}
    for typ in rows[2]:
        if typ:
            type_counts[typ] = type_counts.get(typ, 0) + 1

    names_by_type: dict[str, list[str]] = {}
    for typ, name in zip(rows[2], rows[3]):
        if typ in ("Bone", "Rigid Body", "Marker") and name:
            names_by_type.setdefault(typ, [])
            if name not in names_by_type[typ]:
                names_by_type[typ].append(name)

    rigid_pos_columns = _rigid_body_position_columns(rows)
    speed_summary = _sample_rigid_body_speeds(
        path=path,
        columns={k: v for k, v in rigid_pos_columns.items() if k in RACKET_NAMES},
        fps=float(meta["export_fps"]),
        sample_step=sample_step,
        top_k=top_k,
    )

    return {
        "path": str(path.relative_to(root)),
        "size_mb": path.stat().st_size / 1_000_000.0,
        **meta,
        "type_counts": type_counts,
        "bone_skeletons": sorted({name.split(":", 1)[0] for name in names_by_type.get("Bone", []) if ":" in name}),
        "rigid_bodies": names_by_type.get("Rigid Body", []),
        "marker_count": len(names_by_type.get("Marker", [])),
        "racket_speed_summary": speed_summary,
    }


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Mocap Dataset Analysis",
        "",
        f"Dataset: `{report['dataset_root']}`",
        "",
        "## Overview",
        "",
        f"- BVH files: {len(report['bvh'])}",
        f"- CSV files: {len(report['csv'])}",
        f"- Total size: {report['total_size_gb']:.2f} GB",
        "",
        "## BVH Files",
        "",
        "| File | Frames | FPS | Duration (s) | Roots | Joints | Size (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["bvh"]:
        lines.append(
            f"| `{item['path']}` | {item['frames']} | {item['fps']:.1f} | "
            f"{item['duration_s']:.1f} | {item['roots']} | {item['joints']} | {item['size_mb']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## CSV Files",
            "",
            "| File | Frames | FPS | Duration (s) | Rigid Bodies | Markers | Size (MB) |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for item in report["csv"]:
        duration = item["frames"] / item["export_fps"] if item["export_fps"] else 0.0
        lines.append(
            f"| `{item['path']}` | {item['frames']} | {item['export_fps']:.1f} | {duration:.1f} | "
            f"{', '.join(item['rigid_bodies'])} | {item['marker_count']} | {item['size_mb']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Racket Speed Peaks",
            "",
            "The windows below are coarse 1.5 s candidates centered on sampled racket speed peaks.",
            "",
            "| File | Racket | Peak Speed (m/s) | Time (s) | Suggested Window (s) |",
            "|---|---|---:|---:|---|",
        ]
    )
    for item in report["csv"]:
        for racket, speed_data in item["racket_speed_summary"].items():
            peaks = speed_data["top_speed_peaks"]
            if not peaks:
                continue
            peak = peaks[0]
            win = peak["suggested_window_s"]
            lines.append(
                f"| `{item['path']}` | {racket} | {peak['speed_mps']:.2f} | "
                f"{peak['time_s']:.2f} | {win[0]:.2f}-{win[1]:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Practical Read",
            "",
            "- Use BVH as the main retargeting input; it is already split by skeleton.",
            "- Use CSV rigid bodies to locate swing/contact candidates and to validate racket motion offline.",
            "- Before training, cut one-person BVH clips around selected windows, then retarget to A3 joint space.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/DATA260703"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/mocap"))
    parser.add_argument("--sample-step", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bvh = [_read_bvh_summary(path, dataset) for path in sorted(dataset.rglob("*.bvh"))]
    csv = [_read_csv_summary(path, dataset, args.sample_step, args.top_k) for path in sorted(dataset.rglob("*.csv"))]
    total_size = sum(path.stat().st_size for path in dataset.rglob("*") if path.is_file())
    report = {
        "dataset_root": str(dataset),
        "total_size_gb": total_size / 1_000_000_000.0,
        "sample_step": args.sample_step,
        "bvh": bvh,
        "csv": csv,
    }

    json_path = output_dir / "DATA260703_analysis.json"
    md_path = output_dir / "DATA260703_analysis.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
