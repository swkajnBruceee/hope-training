#!/usr/bin/env python3
"""Validate whether DATA260703 rigid-body candidates can be used as ball tracks."""

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

from analysis.mocap_cleaning.ball_source import analyze_ball_candidate
from analysis.mocap_cleaning.config import load_config
from analysis.mocap_cleaning.derivative import compute_velocity
from analysis.mocap_cleaning.motive_loader import load_motive_csv
from analysis.mocap_cleaning.units import position_scale_to_meters


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# DATA260703 Ball Source Validation",
        "",
        f"Dataset: `{report['dataset_root']}`",
        "",
        "Decision legend: `valid` means usable as `ball_pos`; `invalid` means do not use as ball; `uncertain` means inspect plots/raw data before use.",
        "",
        "| CSV | Candidate | Decision | Valid Ratio | Height Range (m) | Median Speed | Robust P95 Speed | Max Speed | Jump Ratio | Near Racket Events |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["files"]:
        for candidate in item["candidates"]:
            near = ", ".join(f"{k}:{v}" for k, v in candidate["near_racket_events"].items())
            lines.append(
                f"| `{item['csv']}` | {candidate['name']} | **{candidate['decision']}** | "
                f"{candidate['valid_ratio']:.3f} | {candidate['height_range_m']:.3f} | "
                f"{candidate['median_speed_mps']:.3f} | {candidate['robust_p95_speed_mps']:.3f} | "
                f"{candidate['max_speed_mps']:.3f} | {candidate['speed_outlier_ratio']:.3%} | {near} |"
            )

    if report.get("selected_window_metrics"):
        lines.extend(
            [
                "",
                "## Selected Clip Window Metrics",
                "",
                "| Clip | Candidate | Racket | Window (s) | Min Dist (m) | Near<0.20m | Median Speed | P95 Speed | Max Speed | Z Range (m) |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in report["selected_window_metrics"]:
            lines.append(
                f"| `{Path(item['selected_path']).name}` | {item['candidate']} | {item['racket']} | "
                f"{item['start_s']:.2f}-{item['end_s']:.2f} | {item['min_distance_to_racket_m']:.3f} | "
                f"{item['near_racket_frames']}/{item['samples']} | {item['median_speed_mps']:.3f} | "
                f"{item['p95_speed_mps']:.3f} | {item['max_speed_mps']:.3f} | {item['z_range_m']:.3f} |"
            )

    lines.extend(["", "## Reasons", ""])
    for item in report["files"]:
        lines.append(f"### `{item['csv']}`")
        for candidate in item["candidates"]:
            lines.append(f"- `{candidate['name']}`: {candidate['decision']} - {'; '.join(candidate['reasons'])}")
        lines.append("")

    path.write_text("\n".join(lines))


def _window_metrics(trial, candidate: str, racket: str, start_s: float, end_s: float, scale: float) -> dict:
    candidate_pos = trial.rigid_bodies[candidate].pos * scale
    racket_pos = trial.rigid_bodies[racket].pos * scale
    mask = (trial.time >= start_s) & (trial.time <= end_s)
    pos = candidate_pos[mask]
    racket_window = racket_pos[mask]
    time = trial.time[mask]
    if len(time) < 2:
        return {
            "samples": int(len(time)),
            "min_distance_to_racket_m": float("nan"),
            "near_racket_frames": 0,
            "median_speed_mps": float("nan"),
            "p95_speed_mps": float("nan"),
            "max_speed_mps": float("nan"),
            "z_range_m": float("nan"),
        }
    dist = np.linalg.norm(pos - racket_window, axis=1)
    speed = np.linalg.norm(compute_velocity(pos, time), axis=1)
    return {
        "samples": int(len(time)),
        "min_distance_to_racket_m": float(np.nanmin(dist)),
        "near_racket_frames": int(np.sum(dist < 0.20)),
        "median_speed_mps": float(np.nanmedian(speed)),
        "p95_speed_mps": float(np.nanpercentile(speed, 95)),
        "max_speed_mps": float(np.nanmax(speed)),
        "z_range_m": float(np.nanmax(pos[:, 2]) - np.nanmin(pos[:, 2])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("analysis/mocap_cleaning/configs/DATA260703.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/mocap_cleaning_outputs/DATA260703"))
    parser.add_argument("--glob", default="Csv/Rige Body/*.csv")
    parser.add_argument("--selected-manifest", type=Path, default=Path("analysis/mocap/selected_clips/manifest.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])
    candidates = list(config["entities"]["ball_candidates"])
    rackets = list(config["entities"]["rackets"].keys())

    files = []
    loaded_trials = {}
    for csv_path in sorted(dataset_root.glob(args.glob)):
        trial = load_motive_csv(csv_path, rigid_bodies=[*rackets, *candidates])
        loaded_trials[str(csv_path.relative_to(dataset_root))] = trial
        scale = position_scale_to_meters(trial.position_unit)
        racket_positions_m = {
            name: pose.pos * scale
            for name, pose in trial.rigid_bodies.items()
            if name in rackets
        }
        candidate_reports = []
        for name in candidates:
            pose = trial.rigid_bodies.get(name)
            if pose is None:
                candidate_reports.append(
                    {
                        "name": name,
                        "decision": "invalid",
                        "reasons": ["candidate rigid body not found"],
                        "valid_ratio": 0.0,
                        "height_range_m": 0.0,
                        "median_speed_mps": 0.0,
                        "p95_speed_mps": 0.0,
                        "max_speed_mps": 0.0,
                        "static_ratio": 1.0,
                        "near_racket_events": {},
                    }
                )
                continue
            candidate_reports.append(
                analyze_ball_candidate(
                    name=name,
                    pos_m=pose.pos * scale,
                    time=trial.time,
                    racket_positions_m=racket_positions_m,
                ).to_dict()
            )
        files.append(
            {
                "csv": str(csv_path.relative_to(dataset_root)),
                "take_name": trial.take_name,
                "fps": trial.fps,
                "frames": int(len(trial.time)),
                "candidates": candidate_reports,
            }
        )

    selected_window_metrics = []
    if args.selected_manifest.exists():
        selected_manifest = json.loads(args.selected_manifest.read_text())
        for clip in selected_manifest.get("selected", []):
            csv_rel = clip["csv"]
            trial = loaded_trials.get(csv_rel)
            if trial is None:
                continue
            if clip["racket"] not in trial.rigid_bodies:
                continue
            scale = position_scale_to_meters(trial.position_unit)
            for candidate in candidates:
                if candidate not in trial.rigid_bodies:
                    continue
                metrics = _window_metrics(
                    trial=trial,
                    candidate=candidate,
                    racket=clip["racket"],
                    start_s=float(clip["start_s"]),
                    end_s=float(clip["end_s"]),
                    scale=scale,
                )
                selected_window_metrics.append(
                    {
                        "csv": csv_rel,
                        "selected_path": clip["selected_path"],
                        "candidate": candidate,
                        "racket": clip["racket"],
                        "start_s": float(clip["start_s"]),
                        "end_s": float(clip["end_s"]),
                        **metrics,
                    }
                )

    report = {
        "config": str(args.config),
        "dataset_root": str(dataset_root),
        "candidate_names": candidates,
        "files": files,
        "selected_window_metrics": selected_window_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ball_source_report.json"
    md_path = args.output_dir / "ball_source_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
