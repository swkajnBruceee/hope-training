#!/usr/bin/env python3
"""Match tracked racket rigid bodies to skeleton hands in Motive CSV files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


RACKETS = ("TennisBats01", "TennisBats02")
HANDS = (
    "Skeleton 001:LHand",
    "Skeleton 001:RHand",
    "Skeleton 002:LHand",
    "Skeleton 002:RHand",
)


def _parse_header(path: Path) -> list[list[str]]:
    with path.open("r", errors="replace") as f:
        return [next(f).rstrip("\n\r").split(",") for _ in range(8)]


def _position_columns(rows: list[list[str]], typ: str, names: tuple[str, ...]) -> dict[str, tuple[int, int, int]]:
    out: dict[str, dict[str, int]] = {}
    for idx, (row_typ, name, prop, axis) in enumerate(zip(rows[2], rows[3], rows[6], rows[7])):
        if row_typ == typ and name in names and prop == "Position":
            out.setdefault(name, {})[axis] = idx
    return {name: (axes["X"], axes["Y"], axes["Z"]) for name, axes in out.items() if {"X", "Y", "Z"} <= set(axes)}


def _read_pos(row: list[str], cols: tuple[int, int, int]) -> tuple[float, float, float] | None:
    try:
        return float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]])
    except (IndexError, ValueError):
        return None


def _dist_m(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))) / 1000.0


def _window_hand_distances(
    csv_path: Path,
    start_s: float,
    end_s: float,
    sample_step: int,
) -> dict[str, Any]:
    rows = _parse_header(csv_path)
    racket_cols = _position_columns(rows, "Rigid Body", RACKETS)
    hand_cols = _position_columns(rows, "Bone", HANDS)
    distances: dict[str, dict[str, list[float]]] = {
        racket: {hand: [] for hand in hand_cols}
        for racket in racket_cols
    }

    with csv_path.open("r", errors="replace") as f:
        for _ in range(8):
            next(f)
        for idx, line in enumerate(f):
            if idx % sample_step:
                continue
            row = line.rstrip("\n\r").split(",")
            try:
                t = float(row[1])
            except (IndexError, ValueError):
                continue
            if t < start_s:
                continue
            if t > end_s:
                break

            hand_pos = {name: _read_pos(row, cols) for name, cols in hand_cols.items()}
            for racket, cols in racket_cols.items():
                racket_pos = _read_pos(row, cols)
                if racket_pos is None:
                    continue
                for hand, pos in hand_pos.items():
                    if pos is not None:
                        distances[racket][hand].append(_dist_m(racket_pos, pos))

    summary = {}
    for racket, hand_values in distances.items():
        hand_summary = []
        for hand, values in hand_values.items():
            if not values:
                continue
            hand_summary.append(
                {
                    "hand": hand,
                    "median_distance_m": median(values),
                    "min_distance_m": min(values),
                    "samples": len(values),
                }
            )
        hand_summary.sort(key=lambda item: item["median_distance_m"])
        summary[racket] = hand_summary
    return summary


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Racket To Skeleton Matching",
        "",
        "Distances are computed from each racket rigid body to skeleton hand bones inside the candidate window.",
        "Lower median distance is the better match.",
        "",
        "| CSV | Racket | Best Hand | Median Dist (m) | Min Dist (m) | Window (s) |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in report["matches"]:
        for racket, hands in item["distances"].items():
            if not hands:
                continue
            best = hands[0]
            lines.append(
                f"| `{item['csv']}` | {racket} | {best['hand']} | "
                f"{best['median_distance_m']:.3f} | {best['min_distance_m']:.3f} | "
                f"{item['start_s']:.2f}-{item['end_s']:.2f} |"
            )
    lines.append("")
    lines.append("## Full Ranking")
    lines.append("")
    for item in report["matches"]:
        lines.append(f"### `{item['csv']}` `{item['start_s']:.2f}-{item['end_s']:.2f}s`")
        for racket, hands in item["distances"].items():
            ranking = ", ".join(f"{h['hand']}={h['median_distance_m']:.3f}m" for h in hands)
            lines.append(f"- `{racket}`: {ranking}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", type=Path, default=Path("analysis/mocap/DATA260703_analysis.json"))
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/DATA260703"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/mocap"))
    parser.add_argument("--half-window", type=float, default=0.5)
    parser.add_argument("--sample-step", type=int, default=5)
    args = parser.parse_args()

    analysis = json.loads(args.analysis_json.read_text())
    matches = []
    for item in analysis["csv"]:
        csv_path = args.dataset / item["path"]
        for racket, speed_summary in item.get("racket_speed_summary", {}).items():
            peaks = speed_summary.get("top_speed_peaks", [])
            if not peaks:
                continue
            peak = peaks[0]
            if float(peak["speed_mps"]) > 10.0:
                continue
            center = float(peak["time_s"])
            start_s = max(0.0, center - args.half_window)
            end_s = center + args.half_window
            distances = _window_hand_distances(csv_path, start_s, end_s, args.sample_step)
            matches.append(
                {
                    "csv": item["path"],
                    "candidate_racket": racket,
                    "peak_speed_mps": peak["speed_mps"],
                    "start_s": start_s,
                    "end_s": end_s,
                    "distances": {racket: distances.get(racket, [])},
                }
            )

    report = {
        "analysis_json": str(args.analysis_json),
        "dataset": str(args.dataset),
        "half_window": args.half_window,
        "sample_step": args.sample_step,
        "matches": matches,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "DATA260703_racket_skeleton_matching.json"
    md_path = args.output_dir / "DATA260703_racket_skeleton_matching.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
