#!/usr/bin/env python3
"""Generate hit candidates for a Point CSV using one validated marker as ball."""

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

from analysis.mocap_cleaning.cli_generate_hit_candidates import (
    _candidate_scores,
    _label,
    _local_maxima,
    _select_non_overlapping,
    _skeleton_id,
)
from analysis.mocap_cleaning.config import load_config
from analysis.mocap_cleaning.motive_loader import load_motive_csv
from analysis.mocap_cleaning.trajectory_cleaning import clean_position_trajectory
from analysis.mocap_cleaning.units import position_scale_to_meters


def _bvh_clip_path(dataset_root: Path, csv_rel: str, skeleton_id: str) -> Path:
    stem = Path(csv_rel).stem
    return dataset_root / "Bvh" / "Point" / f"{stem}_Skeleton {skeleton_id}.bvh"


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# DATA260703 Point Marker Hit Candidate Manifest",
        "",
        f"Ball marker: `{report['ball_marker']}`",
        f"Selected candidates: `{report['selected_count']}`",
        "",
        "| CSV | Racket | Skeleton | Center (s) | Window (s) | Dist | Racket Speed | Ball dV | Score |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in report["selected"]:
        lines.append(
            f"| `{item['csv']}` | {item['racket']} | {item['skeleton']} | {item['center_s']:.3f} | "
            f"{item['start_s']:.3f}-{item['end_s']:.3f} | {item['candidate_dist_m']:.3f} | "
            f"{item['candidate_racket_speed_mps']:.2f} | {item['candidate_ball_dv_mps']:.2f} | "
            f"{item['candidate_score']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/DATA260703.yaml"))
    parser.add_argument("--csv-rel", default="Csv/Point/Table Tennis_01_004.csv")
    parser.add_argument("--ball-marker", default="FKA-Markerset 001_Marker 001")
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/mocap/point_marker_candidates_004"))
    parser.add_argument("--max-per-racket", type=int, default=100000)
    parser.add_argument("--min-separation-s", type=float, default=1.2)
    parser.add_argument("--pad-before-s", type=float, default=1.0)
    parser.add_argument("--pad-after-s", type=float, default=1.0)
    parser.add_argument("--candidate-distance-m", type=float, default=0.20)
    parser.add_argument("--min-racket-speed-mps", type=float, default=1.0)
    parser.add_argument("--min-ball-dv-mps", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])
    csv_path = dataset_root / args.csv_rel
    racket_names = list(config["entities"]["rackets"].keys())
    trial = load_motive_csv(csv_path, rigid_bodies=racket_names, markers=[args.ball_marker])
    scale = position_scale_to_meters(trial.position_unit)
    if args.ball_marker not in trial.markers:
        raise ValueError(f"marker not found: {args.ball_marker}")

    ball_raw = trial.markers[args.ball_marker] * scale
    ball_pos, cleaning = clean_position_trajectory(
        ball_raw,
        trial.time,
        max_speed_mps=float(config["speed_thresholds"]["ball_mps"]),
        max_gap_s=float(config["gap_policy"]["interpolate_max_s"]),
        min_valid_ratio=0.50,
    )

    selected = []
    diagnostics = []
    for racket in racket_names:
        if racket not in trial.rigid_bodies:
            continue
        racket_pos = trial.rigid_bodies[racket].pos * scale
        scores = _candidate_scores(trial.time, ball_pos, racket_pos, config["hit_detection"]["weights"])
        finite = np.isfinite(ball_pos).all(axis=1) & np.isfinite(racket_pos).all(axis=1)
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
            limit=args.max_per_racket,
        )
        skeleton_name = config["entities"]["rackets"][racket]["expected_skeleton"]
        sid = _skeleton_id(skeleton_name)
        bvh_source = _bvh_clip_path(dataset_root, args.csv_rel, sid)
        for idx in chosen:
            center = float(trial.time[idx])
            start_s = max(float(trial.time[0]), center - args.pad_before_s)
            end_s = min(float(trial.time[-1]), center + args.pad_after_s)
            start_frame = int(round(start_s * fps))
            end_frame = int(round(end_s * fps))
            episode_id = _label(trial.take_name, racket, start_s, end_s, sid)
            selected_path = args.output_dir / f"{episode_id}.bvh"
            selected.append(
                {
                    "source": str(bvh_source),
                    "output": str(selected_path),
                    "start_s": start_s,
                    "end_s": end_s,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frames": max(0, end_frame - start_frame),
                    "fps": fps,
                    "csv": args.csv_rel,
                    "racket": racket,
                    "skeleton": sid,
                    "peak_speed_mps": float(scores["racket_speed"][idx]),
                    "selected_path": str(selected_path),
                    "center_s": center,
                    "candidate_index": int(idx),
                    "candidate_dist_m": float(scores["dist"][idx]),
                    "candidate_racket_speed_mps": float(scores["racket_speed"][idx]),
                    "candidate_ball_dv_mps": float(scores["ball_dv"][idx]),
                    "candidate_score": float(scores["score"][idx]),
                }
            )
        diagnostics.append(
            {
                "csv": args.csv_rel,
                "racket": racket,
                "cleaning": cleaning.to_dict(),
                "valid_candidate_frames": int(np.sum(valid)),
                "local_peaks": len(peaks),
                "selected": len(chosen),
            }
        )

    report = {
        "rule": "Point marker ball-racket proximity + ball velocity change + racket speed",
        "config": str(args.config),
        "csv": args.csv_rel,
        "ball_marker": args.ball_marker,
        "selected_count": len(selected),
        "selected": selected,
        "diagnostics": diagnostics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "manifest.json"
    md_path = args.output_dir / "manifest.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {len(selected)} candidates")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
