#!/usr/bin/env python3
"""Rank selected mocap clips for retargeting readiness."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _parse_header(path: Path) -> list[list[str]]:
    with path.open("r", errors="replace") as f:
        return [next(f).rstrip("\n\r").split(",") for _ in range(8)]


def _pos_cols(rows: list[list[str]], typ: str, name: str) -> tuple[int, int, int] | None:
    axes: dict[str, int] = {}
    for idx, (row_typ, row_name, prop, axis) in enumerate(zip(rows[2], rows[3], rows[6], rows[7])):
        if row_typ == typ and row_name == name and prop == "Position":
            axes[axis] = idx
    if {"X", "Y", "Z"} <= set(axes):
        return axes["X"], axes["Y"], axes["Z"]
    return None


def _read_pos(row: list[str], cols: tuple[int, int, int] | None) -> tuple[float, float, float] | None:
    if cols is None:
        return None
    try:
        return float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]])
    except (IndexError, ValueError):
        return None


def _dist_m(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))) / 1000.0


def _speed_mps(
    prev: tuple[int, tuple[float, float, float]] | None,
    current_frame: int,
    current_pos: tuple[float, float, float],
    fps: float,
) -> float | None:
    if prev is None:
        return None
    prev_frame, prev_pos = prev
    dt = (current_frame - prev_frame) / fps
    if dt <= 0:
        return None
    return _dist_m(prev_pos, current_pos) / dt


def _clip_metrics(dataset: Path, clip: dict[str, Any], sample_step: int) -> dict[str, Any]:
    csv_path = dataset / clip["csv"]
    rows = _parse_header(csv_path)
    skeleton = f"Skeleton {clip['skeleton']}"
    hand_name = f"{skeleton}:RHand"
    hip_name = f"{skeleton}:Hip"

    racket_cols = _pos_cols(rows, "Rigid Body", clip["racket"])
    hand_cols = _pos_cols(rows, "Bone", hand_name)
    hip_cols = _pos_cols(rows, "Bone", hip_name)

    hand_racket_dist = []
    hand_speeds = []
    racket_speeds = []
    hip_positions = []
    missing = 0
    samples = 0
    prev_hand = None
    prev_racket = None

    with csv_path.open("r", errors="replace") as f:
        for _ in range(8):
            next(f)
        for idx, line in enumerate(f):
            if idx % sample_step:
                continue
            row = line.rstrip("\n\r").split(",")
            try:
                frame = int(float(row[0]))
                time_s = float(row[1])
            except (IndexError, ValueError):
                continue
            if time_s < clip["start_s"]:
                continue
            if time_s > clip["end_s"]:
                break

            samples += 1
            hand = _read_pos(row, hand_cols)
            racket = _read_pos(row, racket_cols)
            hip = _read_pos(row, hip_cols)
            if hand is None or racket is None or hip is None:
                missing += 1
                continue

            hand_racket_dist.append(_dist_m(hand, racket))
            hip_positions.append(hip)
            hand_speed = _speed_mps(prev_hand, frame, hand, 360.0)
            racket_speed = _speed_mps(prev_racket, frame, racket, 360.0)
            if hand_speed is not None:
                hand_speeds.append(hand_speed)
            if racket_speed is not None:
                racket_speeds.append(racket_speed)
            prev_hand = (frame, hand)
            prev_racket = (frame, racket)

    hip_displacement = 0.0
    if len(hip_positions) >= 2:
        hip_displacement = _dist_m(hip_positions[0], hip_positions[-1])

    missing_ratio = missing / samples if samples else 1.0
    median_hand_racket = median(hand_racket_dist) if hand_racket_dist else float("inf")
    peak_hand_speed = max(hand_speeds) if hand_speeds else 0.0
    peak_racket_speed = max(racket_speeds) if racket_speeds else 0.0

    # Conservative score: prefer complete, right-hand-close, clearly dynamic clips.
    score = (
        peak_racket_speed
        + 0.5 * peak_hand_speed
        - 2.0 * missing_ratio
        - 1.5 * max(0.0, median_hand_racket - 0.25)
        - 0.4 * max(0.0, hip_displacement - 1.0)
    )
    return {
        "clip": clip,
        "samples": samples,
        "missing_ratio": missing_ratio,
        "median_hand_racket_distance_m": median_hand_racket,
        "min_hand_racket_distance_m": min(hand_racket_dist) if hand_racket_dist else None,
        "peak_hand_speed_mps": peak_hand_speed,
        "peak_racket_speed_mps": peak_racket_speed,
        "hip_displacement_m": hip_displacement,
        "score": score,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Selected Clip Ranking",
        "",
        "Ranking uses racket speed, right-hand speed, hand-racket distance, hip displacement, and missing data.",
        "",
        "| Rank | Clip | Racket | Skeleton | Score | Racket Peak | Hand Peak | Hand-Racket Median | Hip Disp. |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, item in enumerate(report["ranked"], 1):
        clip = item["clip"]
        lines.append(
            f"| {idx} | `{Path(clip['selected_path']).name}` | {clip['racket']} | {clip['skeleton']} | "
            f"{item['score']:.2f} | {item['peak_racket_speed_mps']:.2f} | {item['peak_hand_speed_mps']:.2f} | "
            f"{item['median_hand_racket_distance_m']:.3f} | {item['hip_displacement_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Recommended First Retargeting Tests",
            "",
        ]
    )
    for item in report["ranked"][:5]:
        clip = item["clip"]
        lines.append(
            f"- `{Path(clip['selected_path']).name}`: {clip['csv']} {clip['start_s']:.2f}-{clip['end_s']:.2f}s, "
            f"{clip['racket']} -> Skeleton {clip['skeleton']}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-manifest", type=Path, default=Path("data/analysis/mocap/selected_clips/manifest.json"))
    parser.add_argument("--dataset", type=Path, default=Path("data/DATA260703"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/mocap"))
    parser.add_argument("--sample-step", type=int, default=5)
    args = parser.parse_args()

    manifest = json.loads(args.selected_manifest.read_text())
    ranked = [_clip_metrics(args.dataset, clip, args.sample_step) for clip in manifest["selected"]]
    ranked.sort(key=lambda item: item["score"], reverse=True)

    report = {
        "selected_manifest": str(args.selected_manifest),
        "dataset": str(args.dataset),
        "sample_step": args.sample_step,
        "ranked": ranked,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "DATA260703_selected_clip_ranking.json"
    md_path = args.output_dir / "DATA260703_selected_clip_ranking.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
