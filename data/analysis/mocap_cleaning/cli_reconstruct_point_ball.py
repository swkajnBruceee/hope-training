#!/usr/bin/env python3
"""Experimental multi-marker ball-center reconstruction for Point CSV files."""

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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analysis.mocap_cleaning.config import load_config
from analysis.mocap_cleaning.derivative import compute_velocity
from analysis.mocap_cleaning.motive_loader import load_motive_csv
from analysis.mocap_cleaning.trajectory_cleaning import clean_position_trajectory
from analysis.mocap_cleaning.units import position_scale_to_meters


def _marker_names(candidate_report: dict[str, Any], csv_name: str, include_uncertain: bool, top_n: int) -> list[str]:
    for item in candidate_report["files"]:
        if Path(item["csv"]).name == csv_name:
            allowed = {"candidate", "uncertain"} if include_uncertain else {"candidate"}
            names = [m["name"] for m in item["ranked"] if m["decision"] in allowed]
            return names[:top_n]
    return []


def _largest_cluster(points: np.ndarray, radius_m: float) -> tuple[np.ndarray | None, int, float]:
    if len(points) == 0:
        return None, 0, float("nan")
    best_members = None
    best_count = 0
    best_spread = float("inf")
    for p in points:
        dist = np.linalg.norm(points - p, axis=1)
        members = points[dist <= radius_m]
        if len(members) == 0:
            continue
        center = np.nanmean(members, axis=0)
        spread = float(np.nanmax(np.linalg.norm(members - center, axis=1))) if len(members) > 1 else 0.0
        if len(members) > best_count or (len(members) == best_count and spread < best_spread):
            best_members = members
            best_count = len(members)
            best_spread = spread
    if best_members is None:
        return None, 0, float("nan")
    return np.nanmean(best_members, axis=0), best_count, best_spread


def reconstruct_ball_center(
    markers: dict[str, np.ndarray],
    radius_m: float,
    min_markers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = sorted(markers)
    if not names:
        return np.empty((0, 3)), np.empty(0, dtype=int), np.empty(0)
    n = len(next(iter(markers.values())))
    center = np.full((n, 3), np.nan)
    counts = np.zeros(n, dtype=int)
    spreads = np.full(n, np.nan)
    for i in range(n):
        pts = [markers[name][i] for name in names if np.isfinite(markers[name][i]).all()]
        if not pts:
            continue
        c, count, spread = _largest_cluster(np.stack(pts), radius_m)
        if c is not None and count >= min_markers:
            center[i] = c
            counts[i] = count
            spreads[i] = spread
    return center, counts, spreads


def _near_racket_counts(ball_pos: np.ndarray, rackets: dict[str, np.ndarray], distance_m: float) -> dict[str, int]:
    out = {}
    for name, pos in rackets.items():
        finite = np.isfinite(ball_pos).all(axis=1) & np.isfinite(pos).all(axis=1)
        if not np.any(finite):
            out[name] = 0
            continue
        dist = np.linalg.norm(ball_pos[finite] - pos[finite], axis=1)
        out[name] = int(np.sum(dist < distance_m))
    return out


def _candidate_hit_count(time: np.ndarray, ball_pos: np.ndarray, racket_pos: np.ndarray, cfg: dict[str, Any]) -> int:
    ball_vel = compute_velocity(ball_pos, time)
    racket_vel = compute_velocity(racket_pos, time)
    dist = np.linalg.norm(ball_pos - racket_pos, axis=1)
    racket_speed = np.linalg.norm(racket_vel, axis=1)
    ball_dv = np.zeros(len(time))
    if len(time) > 2:
        ball_dv[1:-1] = np.linalg.norm(ball_vel[2:] - ball_vel[:-2], axis=1)
    finite = np.isfinite(ball_pos).all(axis=1) & np.isfinite(racket_pos).all(axis=1)
    valid = (
        finite
        & (dist < float(cfg["hit_detection"]["max_distance_m"]))
        & (racket_speed > float(cfg["hit_detection"]["min_racket_speed_mps"]))
        & (ball_dv > float(cfg["hit_detection"]["min_ball_dv_mps"]))
    )
    return int(np.sum(valid))


def analyze_file(
    path: Path,
    marker_names: list[str],
    config: dict[str, Any],
    radius_m: float,
    min_markers: int,
) -> dict[str, Any]:
    racket_names = list(config["entities"]["rackets"].keys())
    trial = load_motive_csv(path, rigid_bodies=racket_names, markers=marker_names)
    scale = position_scale_to_meters(trial.position_unit)
    marker_pos = {name: pos * scale for name, pos in trial.markers.items()}
    racket_pos = {name: trial.rigid_bodies[name].pos * scale for name in racket_names if name in trial.rigid_bodies}
    raw_center, counts, spreads = reconstruct_ball_center(marker_pos, radius_m, min_markers)
    clean_center, cleaning = clean_position_trajectory(
        raw_center,
        trial.time,
        max_speed_mps=float(config["speed_thresholds"]["ball_mps"]),
        max_gap_s=float(config["gap_policy"]["interpolate_max_s"]),
        min_valid_ratio=0.50,
    )
    vel = compute_velocity(clean_center, trial.time)
    speed = np.linalg.norm(vel, axis=1)
    finite = np.isfinite(clean_center).all(axis=1)
    hit_counts = {name: _candidate_hit_count(trial.time, clean_center, pos, config) for name, pos in racket_pos.items()}
    return {
        "csv": str(path),
        "markers_used": sorted(marker_pos),
        "marker_count": len(marker_pos),
        "cluster_radius_m": radius_m,
        "min_markers": min_markers,
        "raw_valid_ratio": float(np.mean(np.isfinite(raw_center).all(axis=1))) if len(raw_center) else 0.0,
        "clean_valid_ratio": float(np.mean(finite)) if len(finite) else 0.0,
        "cluster_count_distribution": dict(Counter(int(x) for x in counts if x > 0)),
        "cluster_spread_p95_m": float(np.nanpercentile(spreads[np.isfinite(spreads)], 95)) if np.isfinite(spreads).any() else float("nan"),
        "speed_median_mps": float(np.nanmedian(speed)) if len(speed) else float("nan"),
        "speed_p95_mps": float(np.nanpercentile(speed, 95)) if len(speed) else float("nan"),
        "speed_max_mps": float(np.nanmax(speed)) if len(speed) else float("nan"),
        "near_racket_events": _near_racket_counts(clean_center, racket_pos, 0.20),
        "candidate_hit_frames": hit_counts,
        "cleaning": cleaning.to_dict(),
        "decision": "usable_for_candidate_generation"
        if cleaning.usable and sum(hit_counts.values()) > 0
        else "analysis_only",
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Point Multi-Marker Ball Reconstruction Report",
        "",
        "| CSV | Markers | Raw Valid | Clean Valid | Cluster p95 | Speed p95 | Speed max | Near Rackets | Hit Frames | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["files"]:
        lines.append(
            f"| `{Path(item['csv']).name}` | {item['marker_count']} | {item['raw_valid_ratio']:.3f} | "
            f"{item['clean_valid_ratio']:.3f} | {item['cluster_spread_p95_m']:.3f} | "
            f"{item['speed_p95_mps']:.2f} | {item['speed_max_mps']:.2f} | "
            f"{sum(item['near_racket_events'].values())} | {sum(item['candidate_hit_frames'].values())} | "
            f"{item['decision']} |"
        )
    lines.extend(["", "## Details", ""])
    for item in report["files"]:
        lines.append(f"### `{Path(item['csv']).name}`")
        lines.append(f"- Markers used: {', '.join(item['markers_used'])}")
        lines.append(f"- Cleaning: {'; '.join(item['cleaning']['reasons'])}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/DATA260703.yaml"))
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_point/point_ball_marker_candidates.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_point_reconstruct"))
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--cluster-radius-m", type=float, default=0.08)
    parser.add_argument("--min-markers", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    candidate_report = json.loads(args.candidate_report.read_text())
    point_dir = Path(config["dataset_root"]) / "Csv" / "Point"
    files = []
    for path in sorted(point_dir.glob("*.csv")):
        marker_names = _marker_names(candidate_report, path.name, args.include_uncertain, args.top_n)
        if len(marker_names) < args.min_markers:
            continue
        files.append(analyze_file(path, marker_names, config, args.cluster_radius_m, args.min_markers))

    report = {
        "config": str(args.config),
        "candidate_report": str(args.candidate_report),
        "include_uncertain": args.include_uncertain,
        "top_n": args.top_n,
        "cluster_radius_m": args.cluster_radius_m,
        "min_markers": args.min_markers,
        "files": files,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "point_ball_reconstruction_report.json"
    md_path = args.output_dir / "point_ball_reconstruction_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
