#!/usr/bin/env python3
"""Rank possible ball markers in DATA260703 Point CSV files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis.mocap_cleaning.config import load_config
from analysis.mocap_cleaning.motive_loader import (
    HEADER_ROWS,
    find_entity_columns,
    list_entities,
    parse_motive_metadata,
    read_motive_header,
)
from analysis.mocap_cleaning.units import position_scale_to_meters


@dataclass
class MarkerStats:
    name: str
    sampled_frames: int
    valid_frames: int
    valid_ratio: float
    height_range_m: float
    position_range_m: list[float]
    median_speed_mps: float
    p95_speed_mps: float
    max_speed_mps: float
    max_frame_gap_s: float
    near_racket_events: dict[str, int]
    score: float
    decision: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_marker_names(header: list[list[str]]) -> list[str]:
    markers = list_entities(header).get("Marker", [])
    out = []
    for name in markers:
        if name.startswith("Skeleton "):
            continue
        if name.startswith("TennisBats"):
            continue
        out.append(name)
    return out


def _read_pos(row: list[str], cols: tuple[int, int, int] | None, scale: float) -> np.ndarray | None:
    if cols is None:
        return None
    try:
        pos = np.asarray([float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]])], dtype=float) * scale
    except (IndexError, ValueError):
        return None
    if not np.isfinite(pos).all():
        return None
    return pos


def _score_marker(
    *,
    name: str,
    sampled_frames: int,
    positions: list[np.ndarray],
    times: list[float],
    near_racket_events: dict[str, int],
) -> MarkerStats:
    valid_frames = len(positions)
    valid_ratio = valid_frames / sampled_frames if sampled_frames else 0.0
    if valid_frames < 3:
        return MarkerStats(
            name=name,
            sampled_frames=sampled_frames,
            valid_frames=valid_frames,
            valid_ratio=valid_ratio,
            height_range_m=float("nan"),
            position_range_m=[float("nan")] * 3,
            median_speed_mps=float("nan"),
            p95_speed_mps=float("nan"),
            max_speed_mps=float("nan"),
            max_frame_gap_s=float("nan"),
            near_racket_events=near_racket_events,
            score=-1.0,
            decision="invalid",
            reasons=["too few valid frames"],
        )

    pos = np.stack(positions)
    t = np.asarray(times, dtype=float)
    dpos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dt = np.diff(t)
    valid_dt = dt > 0
    speeds = dpos[valid_dt] / dt[valid_dt]
    finite_speeds = speeds[np.isfinite(speeds)]
    pos_range = np.nanmax(pos, axis=0) - np.nanmin(pos, axis=0)
    height_range = float(pos_range[2])
    median_speed = float(np.nanmedian(finite_speeds)) if len(finite_speeds) else float("nan")
    p95_speed = float(np.nanpercentile(finite_speeds, 95)) if len(finite_speeds) else float("nan")
    max_speed = float(np.nanmax(finite_speeds)) if len(finite_speeds) else float("nan")
    max_frame_gap = float(np.nanmax(dt)) if len(dt) else float("nan")
    near_total = sum(near_racket_events.values())

    reasons: list[str] = []
    if valid_ratio < 0.02:
        reasons.append(f"very sparse marker ({valid_ratio:.3f})")
    if height_range < 0.10:
        reasons.append(f"height range too small ({height_range:.3f} m)")
    if not (0.5 <= p95_speed <= 80.0):
        reasons.append(f"p95 speed outside ball-like range ({p95_speed:.3f} m/s)")
    if max_speed > 120.0:
        reasons.append(f"large tracking jump ({max_speed:.3f} m/s)")
    if near_total == 0:
        reasons.append("never near a racket")

    score = (
        2.0 * min(1.0, height_range / 1.0)
        + 1.5 * min(1.0, p95_speed / 12.0)
        + 1.0 * min(1.0, near_total / 10.0)
        + 0.5 * min(1.0, valid_ratio / 0.20)
        - 1.0 * max(0.0, (max_speed - 80.0) / 80.0)
    )
    if not reasons:
        decision = "candidate"
        reasons.append("ball-like marker fragment")
    elif near_total > 0 and height_range >= 0.10 and 0.5 <= p95_speed <= 80.0:
        decision = "uncertain"
    else:
        decision = "invalid"

    return MarkerStats(
        name=name,
        sampled_frames=sampled_frames,
        valid_frames=valid_frames,
        valid_ratio=valid_ratio,
        height_range_m=height_range,
        position_range_m=[float(x) for x in pos_range],
        median_speed_mps=median_speed,
        p95_speed_mps=p95_speed,
        max_speed_mps=max_speed,
        max_frame_gap_s=max_frame_gap,
        near_racket_events=near_racket_events,
        score=float(score),
        decision=decision,
        reasons=reasons,
    )


def analyze_file(path: Path, config: dict[str, Any], sample_step: int, near_distance_m: float) -> dict[str, Any]:
    header = read_motive_header(path)
    metadata = parse_motive_metadata(header)
    scale = position_scale_to_meters(metadata["length_units"].lower())
    marker_names = _candidate_marker_names(header)
    marker_cols = {name: find_entity_columns(header, "Marker", name).pos for name in marker_names}
    marker_cols = {name: cols for name, cols in marker_cols.items() if cols is not None}
    racket_names = list(config["entities"]["rackets"].keys())
    racket_cols = {
        name: (find_entity_columns(header, "Rigid Body", name).pos if find_entity_columns(header, "Rigid Body", name) else None)
        for name in racket_names
    }

    sampled_frames = 0
    positions: dict[str, list[np.ndarray]] = defaultdict(list)
    times: dict[str, list[float]] = defaultdict(list)
    near: dict[str, dict[str, int]] = {name: {r: 0 for r in racket_names} for name in marker_cols}

    with path.open("r", errors="replace") as f:
        for _ in range(HEADER_ROWS):
            next(f)
        for row_idx, line in enumerate(f):
            if row_idx % sample_step:
                continue
            row = line.rstrip("\n\r").split(",")
            try:
                t = float(row[1])
            except (IndexError, ValueError):
                continue
            sampled_frames += 1
            racket_pos = {name: _read_pos(row, cols, scale) for name, cols in racket_cols.items()}
            for name, cols in marker_cols.items():
                pos = _read_pos(row, cols, scale)
                if pos is None:
                    continue
                positions[name].append(pos)
                times[name].append(t)
                for racket, rpos in racket_pos.items():
                    if rpos is not None and np.linalg.norm(pos - rpos) < near_distance_m:
                        near[name][racket] += 1

    ranked = [
        _score_marker(
            name=name,
            sampled_frames=sampled_frames,
            positions=positions[name],
            times=times[name],
            near_racket_events=near[name],
        )
        for name in marker_cols
    ]
    ranked.sort(key=lambda item: item.score, reverse=True)
    return {
        "csv": str(path),
        "metadata": metadata,
        "sample_step": sample_step,
        "near_distance_m": near_distance_m,
        "candidate_marker_count": len(marker_cols),
        "ranked": [item.to_dict() for item in ranked],
    }


def _write_markdown(report: dict[str, Any], path: Path, top_n: int) -> None:
    lines = ["# Point CSV Ball Marker Candidate Report", ""]
    for item in report["files"]:
        lines.extend(
            [
                f"## `{Path(item['csv']).name}`",
                "",
                f"Candidate markers: `{item['candidate_marker_count']}`",
                "",
                "| Rank | Marker | Decision | Score | Valid Ratio | Height Range | P95 Speed | Max Speed | Near Rackets | Reasons |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for rank, marker in enumerate(item["ranked"][:top_n], 1):
            near_total = sum(marker["near_racket_events"].values())
            lines.append(
                f"| {rank} | `{marker['name']}` | {marker['decision']} | {marker['score']:.2f} | "
                f"{marker['valid_ratio']:.3f} | {marker['height_range_m']:.3f} | "
                f"{marker['p95_speed_mps']:.2f} | {marker['max_speed_mps']:.2f} | "
                f"{near_total} | {'; '.join(marker['reasons'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/DATA260703.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_point"))
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--near-distance-m", type=float, default=0.20)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    point_dir = Path(config["dataset_root"]) / "Csv" / "Point"
    files = [analyze_file(path, config, args.sample_step, args.near_distance_m) for path in sorted(point_dir.glob("*.csv"))]
    report = {
        "config": str(args.config),
        "point_dir": str(point_dir),
        "files": files,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "point_ball_marker_candidates.json"
    md_path = args.output_dir / "point_ball_marker_candidates.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path, args.top_n)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
