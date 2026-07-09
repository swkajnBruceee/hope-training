#!/usr/bin/env python3
"""Generate hit candidate windows from reconstructed ball trajectories."""

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
from typing import Any

import numpy as np

from analysis.mocap_cleaning.config import load_config
from analysis.mocap_cleaning.derivative import compute_velocity
from analysis.mocap_cleaning.motive_loader import load_motive_csv
from analysis.mocap_cleaning.units import position_scale_to_meters


def _local_maxima(score: np.ndarray, valid: np.ndarray, radius: int) -> list[int]:
    idxs = np.flatnonzero(valid & np.isfinite(score))
    peaks = []
    for idx in idxs:
        start = max(0, idx - radius)
        end = min(len(score), idx + radius + 1)
        if score[idx] >= np.nanmax(score[start:end]):
            peaks.append(int(idx))
    return peaks


def _select_non_overlapping(peaks: list[int], score: np.ndarray, min_sep_frames: int, limit: int) -> list[int]:
    ranked = sorted(peaks, key=lambda i: float(score[i]), reverse=True)
    selected: list[int] = []
    for idx in ranked:
        if all(abs(idx - existing) >= min_sep_frames for existing in selected):
            selected.append(idx)
        if len(selected) >= limit:
            break
    return sorted(selected)


def _candidate_scores(time: np.ndarray, ball_pos: np.ndarray, racket_pos: np.ndarray, weights: dict[str, float]) -> dict[str, np.ndarray]:
    ball_vel = compute_velocity(ball_pos, time)
    racket_vel = compute_velocity(racket_pos, time)
    dist = np.linalg.norm(ball_pos - racket_pos, axis=1)
    racket_speed = np.linalg.norm(racket_vel, axis=1)
    ball_dv = np.zeros(len(time), dtype=float)
    if len(time) > 2:
        ball_dv[1:-1] = np.linalg.norm(ball_vel[2:] - ball_vel[:-2], axis=1)

    def norm(v: np.ndarray) -> np.ndarray:
        finite = v[np.isfinite(v)]
        if len(finite) == 0 or np.nanmax(finite) <= 1e-8:
            return np.zeros_like(v)
        return v / float(np.nanmax(finite))

    dist_score = np.exp(-dist / 0.08)
    score = (
        float(weights.get("distance", 0.5)) * dist_score
        + float(weights.get("ball_dv", 0.3)) * norm(ball_dv)
        + float(weights.get("racket_speed", 0.2)) * norm(racket_speed)
    )
    return {
        "dist": dist,
        "racket_speed": racket_speed,
        "ball_dv": ball_dv,
        "score": score,
    }


def _bvh_path(dataset_root: Path, config: dict[str, Any], take: str, racket: str) -> Path:
    player = str(config["entities"]["rackets"][racket].get("player", racket))
    bvh_dir = dataset_root / config["source_layout"].get("bvh_dir", "BVH")
    pattern = str(config["source_layout"].get("bvh_name_pattern", "{take}_{player}.bvh"))
    return bvh_dir / pattern.format(take=take, player=player, racket=racket)


def _label(take: str, racket: str, start_s: float, end_s: float) -> str:
    return f"{take}_{racket}_{start_s:.2f}_{end_s:.2f}".replace(".", "p").replace(" ", "_")


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Hit Candidate Manifest From Reconstructed Ball",
        "",
        f"Selected candidates: `{report['selected_count']}`",
        "",
        "| CSV | Racket | Center | Window | Dist | Racket Speed | Ball dV | Score |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in report["selected"]:
        lines.append(
            f"| `{Path(item['csv']).name}` | {item['racket']} | {item['center_s']:.3f} | "
            f"{item['start_s']:.3f}-{item['end_s']:.3f} | {item['candidate_dist_m']:.3f} | "
            f"{item['candidate_racket_speed_mps']:.2f} | {item['candidate_ball_dv_mps']:.2f} | "
            f"{item['candidate_score']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/DATA260708.yaml"))
    parser.add_argument(
        "--ball-report",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260708/ball_reconstruction/unlabeled_ball_reconstruction_report.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/mocap/DATA260708_hit_candidates"))
    parser.add_argument("--max-per-racket-per-csv", type=int, default=8)
    parser.add_argument("--min-separation-s", type=float, default=1.2)
    parser.add_argument("--pad-before-s", type=float, default=1.0)
    parser.add_argument("--pad-after-s", type=float, default=1.0)
    parser.add_argument("--candidate-distance-m", type=float, default=0.20)
    parser.add_argument("--min-racket-speed-mps", type=float, default=1.0)
    parser.add_argument("--min-ball-dv-mps", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    racket_names = list(config["entities"]["rackets"].keys())
    ball_report = json.loads(args.ball_report.read_text())

    selected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for item in ball_report["files"]:
        if item["decision"] != "usable_for_hit_candidates":
            diagnostics.append({"csv": item["csv"], "reason": "ball reconstruction not usable"})
            continue
        csv_path = Path(item["csv"])
        ball_npz = np.load(item["npz_path"], allow_pickle=False)
        time = ball_npz["time"]
        ball_pos = ball_npz["ball_pos_clean"]
        trial = load_motive_csv(csv_path, rigid_bodies=racket_names)
        scale = position_scale_to_meters(trial.position_unit)
        csv_rel = str(csv_path.relative_to(dataset_root))
        for racket in racket_names:
            if racket not in trial.rigid_bodies:
                diagnostics.append({"csv": csv_rel, "racket": racket, "reason": "missing racket rigid body"})
                continue
            racket_pos = trial.rigid_bodies[racket].pos * scale
            scores = _candidate_scores(time, ball_pos, racket_pos, config["hit_detection"]["weights"])
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
                limit=args.max_per_racket_per_csv,
            )
            bvh_source = _bvh_path(dataset_root, config, trial.take_name, racket)
            for idx in chosen:
                center = float(time[idx])
                start_s = max(float(time[0]), center - args.pad_before_s)
                end_s = min(float(time[-1]), center + args.pad_after_s)
                episode_id = _label(trial.take_name, racket, start_s, end_s)
                selected_path = output_dir / f"{episode_id}.bvh"
                selected.append(
                    {
                        "source": str(bvh_source),
                        "output": str(selected_path),
                        "start_s": start_s,
                        "end_s": end_s,
                        "start_frame": int(round(start_s * fps)),
                        "end_frame": int(round(end_s * fps)),
                        "frames": max(0, int(round((end_s - start_s) * fps))),
                        "fps": fps,
                        "csv": csv_rel,
                        "racket": racket,
                        "skeleton": str(config["entities"]["rackets"][racket]["expected_skeleton"]),
                        "peak_speed_mps": float(scores["racket_speed"][idx]),
                        "selected_path": str(selected_path),
                        "center_s": center,
                        "candidate_index": int(idx),
                        "candidate_dist_m": float(scores["dist"][idx]),
                        "candidate_racket_speed_mps": float(scores["racket_speed"][idx]),
                        "candidate_ball_dv_mps": float(scores["ball_dv"][idx]),
                        "candidate_score": float(scores["score"][idx]),
                        "ball_reconstruction_npz": str(item["npz_path"]),
                        "ball_reconstruction_decision": item["decision"],
                    }
                )
            diagnostics.append(
                {
                    "csv": csv_rel,
                    "racket": racket,
                    "valid_candidate_frames": int(np.sum(valid)),
                    "local_peaks": len(peaks),
                    "selected": len(chosen),
                    "bvh_source_exists": bvh_source.exists(),
                }
            )

    report = {
        "rule": "reconstructed ball-racket proximity + ball velocity change + racket speed",
        "config": str(args.config),
        "ball_report": str(args.ball_report),
        "selected_count": len(selected),
        "selected": selected,
        "diagnostics": diagnostics,
    }
    json_path = output_dir / "manifest.json"
    md_path = output_dir / "manifest.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {len(selected)} candidates")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
