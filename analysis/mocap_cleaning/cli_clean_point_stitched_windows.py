#!/usr/bin/env python3
"""Clean hit windows from Point CSV files by stitching ball marker IDs."""

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
from pathlib import Path

import numpy as np

from analysis.mocap_cleaning.cli_clean_tennis_windows import _body_bone_names, _body_reference, _dist_stats, _window_slice
from analysis.mocap_cleaning.cli_generate_hit_candidates import (
    _candidate_scores,
    _label,
    _local_maxima,
    _select_non_overlapping,
    _skeleton_id,
)
from analysis.mocap_cleaning.config import load_config
from analysis.mocap_cleaning.derivative import compute_velocity
from analysis.mocap_cleaning.motive_loader import load_motive_csv
from analysis.mocap_cleaning.trajectory_cleaning import clean_position_trajectory
from analysis.mocap_cleaning.units import position_scale_to_meters


def _selected_markers(candidate_report: dict, csv_name: str, include_uncertain: bool) -> dict[str, float]:
    allowed = {"candidate", "uncertain"} if include_uncertain else {"candidate"}
    for item in candidate_report["files"]:
        if Path(item["csv"]).name == csv_name:
            return {
                marker["name"]: float(marker["score"])
                for marker in item["ranked"]
                if marker["decision"] in allowed
            }
    return {}


def _stitch_single_ball(markers: dict[str, np.ndarray], marker_scores: dict[str, float], time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = sorted(markers, key=lambda name: marker_scores.get(name, 0.0), reverse=True)
    ball = np.full((len(time), 3), np.nan)
    source_idx = np.full(len(time), -1, dtype=int)
    prev = None
    for i in range(len(time)):
        candidates = [(idx, name, markers[name][i]) for idx, name in enumerate(names) if np.isfinite(markers[name][i]).all()]
        if not candidates:
            prev = None
            continue
        if prev is None:
            chosen = candidates[0]
        else:
            dt = max(float(time[i] - time[i - 1]), 1e-6)
            # Prefer temporal continuity; fall back to marker score when distances are similar.
            ranked = sorted(
                candidates,
                key=lambda item: (
                    np.linalg.norm(item[2] - prev) / dt,
                    -marker_scores.get(item[1], 0.0),
                ),
            )
            chosen = ranked[0]
        source_idx[i] = chosen[0]
        ball[i] = chosen[2]
        prev = chosen[2]
    return ball, source_idx


def _bvh_clip_path(dataset_root: Path, csv_rel: str, skeleton_id: str) -> Path:
    stem = Path(csv_rel).stem
    return dataset_root / "Bvh" / "Point" / f"{stem}_Skeleton {skeleton_id}.bvh"


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# DATA260703 Point Stitched Cleaned Windows",
        "",
        f"Candidate report: `{report['candidate_report']}`",
        "",
        "| Clip | CSV | Racket | Raw Valid | Clean Usable | Clean Valid | Min Dist | Candidate Score |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for item in report["windows"]:
        min_dist = item["distance_after_cleaning"]["min_distance_m"]
        lines.append(
            f"| `{Path(item['selected_path']).name}` | `{item['source_csv']}` | {item['racket']} | "
            f"{item['stitched_raw_valid_ratio']:.3f} | {item['cleaning']['usable']} | "
            f"{item['cleaning']['cleaned_valid_ratio']:.3f} | "
            f"{'nan' if min_dist is None else f'{min_dist:.3f}'} | {item['candidate_score']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("analysis/mocap_cleaning/configs/DATA260703.yaml"))
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=Path("analysis/mocap_cleaning_outputs/DATA260703_point/point_ball_marker_candidates.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/mocap_cleaning_outputs/DATA260703_point_stitched"))
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--max-per-racket-per-csv", type=int, default=100000)
    parser.add_argument("--min-separation-s", type=float, default=1.2)
    parser.add_argument("--pad-before-s", type=float, default=1.0)
    parser.add_argument("--pad-after-s", type=float, default=1.0)
    parser.add_argument("--candidate-distance-m", type=float, default=0.20)
    parser.add_argument("--min-racket-speed-mps", type=float, default=1.0)
    parser.add_argument("--min-ball-dv-mps", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.config)
    candidate_report = json.loads(args.candidate_report.read_text())
    dataset_root = Path(config["dataset_root"])
    point_dir = dataset_root / "Csv" / "Point"
    racket_config = config["entities"]["rackets"]
    racket_names = list(racket_config.keys())
    bone_names = []
    for racket in racket_names:
        bone_names.extend(_body_bone_names(racket_config[racket]["expected_skeleton"]))

    output_dir = args.output_dir
    window_dir = output_dir / "cleaned_windows"
    window_dir.mkdir(parents=True, exist_ok=True)
    max_speed = float(config["speed_thresholds"]["ball_mps"])
    max_gap_s = float(config["gap_policy"]["interpolate_max_s"])

    windows = []
    stitch_reports = []
    for csv_path in sorted(point_dir.glob("*.csv")):
        marker_scores = _selected_markers(candidate_report, csv_path.name, args.include_uncertain)
        if not marker_scores:
            continue
        csv_rel = str(csv_path.relative_to(dataset_root))
        trial = load_motive_csv(csv_path, rigid_bodies=racket_names, bones=bone_names, markers=marker_scores.keys())
        scale = position_scale_to_meters(trial.position_unit)
        markers = {name: trial.markers[name] * scale for name in marker_scores if name in trial.markers}
        if not markers:
            continue
        ball_raw, source_idx = _stitch_single_ball(markers, marker_scores, trial.time)
        ball_clean, cleaning_full = clean_position_trajectory(
            ball_raw,
            trial.time,
            max_speed_mps=max_speed,
            max_gap_s=max_gap_s,
            min_valid_ratio=0.50,
        )
        raw_valid_ratio = float(np.mean(np.isfinite(ball_raw).all(axis=1)))
        stitch_reports.append(
            {
                "csv": csv_rel,
                "markers": sorted(markers),
                "raw_valid_ratio": raw_valid_ratio,
                "full_cleaning": cleaning_full.to_dict(),
            }
        )

        for racket in racket_names:
            if racket not in trial.rigid_bodies:
                continue
            racket_pos_full = trial.rigid_bodies[racket].pos * scale
            scores = _candidate_scores(trial.time, ball_clean, racket_pos_full, config["hit_detection"]["weights"])
            finite = np.isfinite(ball_clean).all(axis=1) & np.isfinite(racket_pos_full).all(axis=1)
            valid = (
                finite
                & (scores["dist"] < args.candidate_distance_m)
                & (scores["racket_speed"] > args.min_racket_speed_mps)
                & (scores["ball_dv"] > args.min_ball_dv_mps)
            )
            fps = float(trial.fps)
            peaks = _local_maxima(scores["score"], valid, radius=max(1, int(round(0.08 * fps))))
            chosen = _select_non_overlapping(
                peaks,
                scores["score"],
                min_sep_frames=max(1, int(round(args.min_separation_s * fps))),
                limit=args.max_per_racket_per_csv,
            )
            skeleton_name = racket_config[racket]["expected_skeleton"]
            sid = _skeleton_id(skeleton_name)
            bvh_source = _bvh_clip_path(dataset_root, csv_rel, sid)
            for idx in chosen:
                center = float(trial.time[idx])
                start_s = max(float(trial.time[0]), center - args.pad_before_s)
                end_s = min(float(trial.time[-1]), center + args.pad_after_s)
                mask = _window_slice(trial.time, start_s, end_s)
                time = trial.time[mask]
                time_rel = time - start_s
                ball_raw_win = ball_raw[mask]
                ball_clean_win, cleaning_win = clean_position_trajectory(
                    ball_raw_win,
                    time,
                    max_speed_mps=max_speed,
                    max_gap_s=max_gap_s,
                    min_valid_ratio=0.95,
                )
                racket_pos = racket_pos_full[mask]
                racket_quat = trial.rigid_bodies[racket].quat_xyzw
                if racket_quat is None:
                    racket_quat = np.full((len(trial.time), 4), np.nan)
                racket_quat = racket_quat[mask]
                body_center, body_right_axis = _body_reference(trial, skeleton_name, mask, scale)
                ball_vel = compute_velocity(ball_clean_win, time)
                start_frame = int(round(start_s * fps))
                end_frame = int(round(end_s * fps))
                episode_id = _label(trial.take_name, racket, start_s, end_s, sid)
                selected_path = args.output_dir / f"{episode_id}.bvh"
                npz_path = window_dir / f"{episode_id}_PointStitched_cleaned.npz"
                np.savez(
                    npz_path,
                    episode_id=np.asarray(episode_id),
                    time=time,
                    time_rel=time_rel,
                    ball_pos_raw=ball_raw_win,
                    ball_pos_clean=ball_clean_win,
                    ball_vel=ball_vel,
                    racket_pos=racket_pos,
                    racket_quat=racket_quat,
                    body_center=body_center,
                    body_right_axis=body_right_axis,
                    skeleton=np.asarray(skeleton_name),
                    source_csv=np.asarray(csv_rel),
                    source_bvh=np.asarray(str(selected_path)),
                    candidate=np.asarray("PointStitched"),
                    racket=np.asarray(racket),
                )
                windows.append(
                    {
                        "episode_id": episode_id,
                        "selected_path": str(selected_path),
                        "source_csv": csv_rel,
                        "source": str(bvh_source),
                        "output": str(selected_path),
                        "candidate": "PointStitched",
                        "racket": racket,
                        "skeleton": sid,
                        "start_s": start_s,
                        "end_s": end_s,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "frames": max(0, end_frame - start_frame),
                        "fps": fps,
                        "npz_path": str(npz_path),
                        "samples": int(len(time)),
                        "candidate_score": float(scores["score"][idx]),
                        "candidate_dist_m": float(scores["dist"][idx]),
                        "candidate_ball_dv_mps": float(scores["ball_dv"][idx]),
                        "candidate_racket_speed_mps": float(scores["racket_speed"][idx]),
                        "stitched_raw_valid_ratio": raw_valid_ratio,
                        "cleaning": cleaning_win.to_dict(),
                        "distance_after_cleaning": _dist_stats(ball_clean_win, racket_pos),
                    }
                )

    report = {
        "config": str(args.config),
        "candidate_report": str(args.candidate_report),
        "include_uncertain": args.include_uncertain,
        "max_speed_mps": max_speed,
        "max_gap_s": max_gap_s,
        "stitch_reports": stitch_reports,
        "windows": windows,
    }
    json_path = output_dir / "cleaned_tennis_windows_report.json"
    md_path = output_dir / "cleaned_tennis_windows_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {len(windows)} cleaned window files to {window_dir}")


if __name__ == "__main__":
    main()
