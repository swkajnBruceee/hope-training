#!/usr/bin/env python3
"""Clean candidate windows using reconstructed ball trajectories."""

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
from analysis.mocap_cleaning.trajectory_cleaning import clean_position_trajectory
from analysis.mocap_cleaning.units import position_scale_to_meters


def _window_slice(time: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    return (time >= start_s) & (time <= end_s)


def _body_bone_names(skeleton_name: str) -> list[str]:
    return [
        f"{skeleton_name}:{skeleton_name}",
        f"{skeleton_name}:LShoulder",
        f"{skeleton_name}:RShoulder",
    ]


def _body_reference(trial, skeleton_name: str, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    hip_name, left_shoulder_name, right_shoulder_name = _body_bone_names(skeleton_name)
    n = int(np.sum(mask))
    if not all(name in trial.bones for name in (hip_name, left_shoulder_name, right_shoulder_name)):
        return np.full((n, 3), np.nan), np.full((n, 3), np.nan)

    hip = trial.bones[hip_name].pos[mask] * scale
    left_shoulder = trial.bones[left_shoulder_name].pos[mask] * scale
    right_shoulder = trial.bones[right_shoulder_name].pos[mask] * scale
    shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
    body_center = 0.5 * (hip + shoulder_mid)
    body_right_axis = right_shoulder - left_shoulder
    norm = np.linalg.norm(body_right_axis, axis=1, keepdims=True)
    valid = np.isfinite(body_right_axis).all(axis=1, keepdims=True) & (norm > 1e-8)
    body_right_axis = np.divide(body_right_axis, norm, out=np.full_like(body_right_axis, np.nan), where=valid)
    return body_center, body_right_axis


def _dist_stats(ball_pos: np.ndarray, racket_pos: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(ball_pos).all(axis=1) & np.isfinite(racket_pos).all(axis=1)
    if not np.any(finite):
        return {"min_distance_m": None, "near_020_frames": 0}
    dist = np.linalg.norm(ball_pos[finite] - racket_pos[finite], axis=1)
    return {"min_distance_m": float(np.nanmin(dist)), "near_020_frames": int(np.sum(dist < 0.20))}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cleaned Reconstructed-Ball Windows",
        "",
        f"Selected manifest: `{report['selected_manifest']}`",
        "",
        "| Clip | Racket | Clean P95 Speed | Outliers | Filled Gaps | Long Gaps | Valid Ratio | Min Dist | Usable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["windows"]:
        clean = item["cleaning"]
        dist = item["distance_after_cleaning"]
        min_dist = dist["min_distance_m"]
        min_dist_text = "nan" if min_dist is None else f"{min_dist:.3f}"
        lines.append(
            f"| `{Path(item['selected_path']).name}` | {item['racket']} | "
            f"{clean['cleaned_p95_speed_mps']:.2f} | {clean['outlier_frames']} | "
            f"{clean['short_gaps_filled']} | {clean['long_gaps']} | {clean['cleaned_valid_ratio']:.3f} | "
            f"{min_dist_text} | {clean['usable']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/DATA260708.yaml"))
    parser.add_argument("--selected-manifest", type=Path, default=Path("data/analysis/mocap/DATA260708_hit_candidates/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/mocap_cleaning_outputs/DATA260708"))
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])
    racket_config = config["entities"]["rackets"]
    configured_rackets = list(racket_config.keys())
    configured_bones = []
    for racket_name in configured_rackets:
        configured_bones.extend(_body_bone_names(str(racket_config[racket_name]["expected_skeleton"])))

    manifest = json.loads(args.selected_manifest.read_text())
    output_dir = args.output_dir
    window_dir = output_dir / "cleaned_windows"
    window_dir.mkdir(parents=True, exist_ok=True)
    loaded_trials = {}
    loaded_ball = {}
    windows = []

    for clip in manifest["selected"]:
        csv_rel = clip["csv"]
        csv_path = dataset_root / csv_rel
        if csv_rel not in loaded_trials:
            loaded_trials[csv_rel] = load_motive_csv(csv_path, rigid_bodies=configured_rackets, bones=configured_bones)
        ball_path = str(clip["ball_reconstruction_npz"])
        if ball_path not in loaded_ball:
            loaded_ball[ball_path] = np.load(ball_path, allow_pickle=False)
        trial = loaded_trials[csv_rel]
        ball_data = loaded_ball[ball_path]
        if clip["racket"] not in trial.rigid_bodies:
            continue

        scale = position_scale_to_meters(trial.position_unit)
        mask = _window_slice(trial.time, float(clip["start_s"]), float(clip["end_s"]))
        time = trial.time[mask]
        time_rel = time - float(clip["start_s"])
        ball_raw = ball_data["ball_pos_clean"][mask]
        racket_pos = trial.rigid_bodies[clip["racket"]].pos[mask] * scale
        racket_quat = trial.rigid_bodies[clip["racket"]].quat_xyzw
        if racket_quat is None:
            racket_quat = np.full((len(trial.time), 4), np.nan)
        racket_quat = racket_quat[mask]
        skeleton_name = str(racket_config[clip["racket"]]["expected_skeleton"])
        body_center, body_right_axis = _body_reference(trial, skeleton_name, mask, scale)

        ball_clean, cleaning = clean_position_trajectory(
            ball_raw,
            time,
            max_speed_mps=float(config["speed_thresholds"]["ball_mps"]),
            max_gap_s=float(config["gap_policy"]["interpolate_max_s"]),
            min_valid_ratio=0.95,
        )
        ball_vel = compute_velocity(ball_clean, time)
        dist_stats = _dist_stats(ball_clean, racket_pos)
        episode_id = Path(clip["selected_path"]).stem
        npz_path = window_dir / f"{episode_id}_unlabeled_ball_cleaned.npz"
        np.savez(
            npz_path,
            episode_id=np.asarray(episode_id),
            time=time,
            time_rel=time_rel,
            ball_pos_raw=ball_raw,
            ball_pos_clean=ball_clean,
            ball_vel=ball_vel,
            racket_pos=racket_pos,
            racket_quat=racket_quat,
            body_center=body_center,
            body_right_axis=body_right_axis,
            skeleton=np.asarray(skeleton_name),
            source_csv=np.asarray(csv_rel),
            source_bvh=np.asarray(clip["source"]),
            candidate=np.asarray("UnlabeledBall"),
            racket=np.asarray(clip["racket"]),
            ball_reconstruction_npz=np.asarray(ball_path),
        )
        windows.append(
            {
                "episode_id": episode_id,
                "selected_path": clip["selected_path"],
                "source_csv": csv_rel,
                "candidate": "UnlabeledBall",
                "racket": clip["racket"],
                "start_s": float(clip["start_s"]),
                "end_s": float(clip["end_s"]),
                "npz_path": str(npz_path),
                "samples": int(len(time)),
                "cleaning": cleaning.to_dict(),
                "distance_after_cleaning": dist_stats,
                "ball_reconstruction_npz": ball_path,
            }
        )

    report = {
        "config": str(args.config),
        "selected_manifest": str(args.selected_manifest),
        "candidate": "UnlabeledBall",
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
