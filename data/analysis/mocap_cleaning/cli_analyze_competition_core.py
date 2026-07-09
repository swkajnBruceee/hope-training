#!/usr/bin/env python3
"""Generate a first-pass analysis report for the competition core dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _json_list(values: np.ndarray) -> list[dict[str, Any]]:
    return [json.loads(str(v)) for v in values]


def _rate(success: np.ndarray) -> float:
    if len(success) == 0:
        return float("nan")
    return float(np.mean(success == 1))


def _summary(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "p10": float(np.percentile(vals, 10)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
    }


def _group_rows(group: np.ndarray, success: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(str(x) for x in group)):
        mask = group == key
        counts = Counter(success[mask].astype(int).tolist())
        total = int(mask.sum())
        rows.append(
            {
                "group": key,
                "total": total,
                "success": int(counts[1]),
                "failure": int(counts[0]),
                "success_rate": _rate(success[mask]),
            }
        )
    return rows


def _bin_success(values: np.ndarray, success: np.ndarray, bins: list[float]) -> list[dict[str, Any]]:
    rows = []
    vals = np.asarray(values, dtype=float)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (vals >= lo) & (vals < hi)
        rows.append(
            {
                "range": f"[{lo:.3f}, {hi:.3f})",
                "total": int(mask.sum()),
                "success": int(np.sum(success[mask] == 1)),
                "failure": int(np.sum(success[mask] == 0)),
                "success_rate": _rate(success[mask]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# DATA260708 Competition Core First Analysis",
        "",
        f"Dataset: `{report['dataset']}`",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Samples | {report['samples']} |",
        f"| Success | {report['success']} |",
        f"| Failure | {report['failure']} |",
        f"| Success rate | {report['success_rate']:.3f} |",
        "",
        "## Success By Racket",
        "",
        "| racket | total | success | failure | success rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["success_by_racket"]:
        lines.append(f"| {row['group']} | {row['total']} | {row['success']} | {row['failure']} | {row['success_rate']:.3f} |")
    lines.extend(["", "## Success By Stroke Type", "", "| stroke | total | success | failure | success rate |", "|---|---:|---:|---:|---:|"])
    for row in report["success_by_stroke"]:
        lines.append(f"| {row['group']} | {row['total']} | {row['success']} | {row['failure']} | {row['success_rate']:.3f} |")
    lines.extend(["", "## Position And Speed Summary", ""])
    for name, stats in report["summaries"].items():
        lines.append(f"- `{name}`: p10={stats['p10']:.4f}, p50={stats['p50']:.4f}, p90={stats['p90']:.4f}")
    lines.extend(["", "## Landing Zones", ""])
    lines.append("| x bin | y bin | total | success | failure |")
    lines.append("|---|---|---:|---:|---:|")
    for row in report["landing_zone_counts"][:60]:
        lines.append(f"| {row['x_bin']} | {row['y_bin']} | {row['total']} | {row['success']} | {row['failure']} |")
    lines.extend(["", "## Output Tables", ""])
    for name, p in report["tables"].items():
        lines.append(f"- `{name}`: `{p}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    n = int(data["ball_pos"].shape[0])
    hit_index = data["hit_index"].astype(int)
    success = data["success"].astype(int)
    sources = _json_list(data["source_json"])
    rackets = np.asarray([str(s.get("racket", "unknown")) for s in sources])
    stroke = data["stroke_type_rule_v2"] if "stroke_type_rule_v2" in data.files else data["stroke_type"]
    stroke = np.asarray([str(x) for x in stroke])

    hit_ball = data["ball_pos"][np.arange(n), hit_index]
    hit_racket = data["racket_pos"][np.arange(n), hit_index]
    hit_dist = np.linalg.norm(hit_ball - hit_racket, axis=1)
    landing = data["landing_pos"]
    ball_in_speed = np.linalg.norm(data["ball_in_vel"], axis=1)
    ball_out_speed = np.linalg.norm(data["ball_out_vel"], axis=1)
    racket_speed_hit = np.linalg.norm(data["racket_vel_at_hit"], axis=1)

    speed_bins = [0.0, 1.5, 2.5, 3.5, 5.0, 8.0, 20.0]
    racket_speed_rows = _bin_success(racket_speed_hit, success, speed_bins)
    out_speed_rows = _bin_success(ball_out_speed, success, speed_bins)
    hit_dist_rows = _bin_success(hit_dist, success, [0.0, 0.03, 0.05, 0.08, 0.12, 0.16])

    x_edges = np.linspace(0.0, 2.74, 7)
    y_edges = np.linspace(-1.525, 0.0, 5)
    zone_rows = []
    for xi in range(len(x_edges) - 1):
        for yi in range(len(y_edges) - 1):
            mask = (
                (landing[:, 0] >= x_edges[xi])
                & (landing[:, 0] < x_edges[xi + 1])
                & (landing[:, 1] >= y_edges[yi])
                & (landing[:, 1] < y_edges[yi + 1])
            )
            if not np.any(mask):
                continue
            zone_rows.append(
                {
                    "x_bin": f"{x_edges[xi]:.2f}-{x_edges[xi + 1]:.2f}",
                    "y_bin": f"{y_edges[yi]:.2f}-{y_edges[yi + 1]:.2f}",
                    "total": int(mask.sum()),
                    "success": int(np.sum(success[mask] == 1)),
                    "failure": int(np.sum(success[mask] == 0)),
                }
            )
    zone_rows.sort(key=lambda r: r["total"], reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "success_by_racket": str(args.output_dir / "success_by_racket.csv"),
        "success_by_stroke": str(args.output_dir / "success_by_stroke.csv"),
        "success_by_racket_speed": str(args.output_dir / "success_by_racket_speed.csv"),
        "success_by_ball_out_speed": str(args.output_dir / "success_by_ball_out_speed.csv"),
        "success_by_hit_distance": str(args.output_dir / "success_by_hit_distance.csv"),
        "landing_zones": str(args.output_dir / "landing_zones.csv"),
    }
    success_by_racket = _group_rows(rackets, success)
    success_by_stroke = _group_rows(stroke, success)
    _write_csv(Path(tables["success_by_racket"]), success_by_racket)
    _write_csv(Path(tables["success_by_stroke"]), success_by_stroke)
    _write_csv(Path(tables["success_by_racket_speed"]), racket_speed_rows)
    _write_csv(Path(tables["success_by_ball_out_speed"]), out_speed_rows)
    _write_csv(Path(tables["success_by_hit_distance"]), hit_dist_rows)
    _write_csv(Path(tables["landing_zones"]), zone_rows)

    report = {
        "dataset": str(args.dataset),
        "samples": n,
        "success": int(np.sum(success == 1)),
        "failure": int(np.sum(success == 0)),
        "success_rate": _rate(success),
        "success_by_racket": success_by_racket,
        "success_by_stroke": success_by_stroke,
        "summaries": {
            "hit_x_m": _summary(hit_ball[:, 0]),
            "hit_y_m": _summary(hit_ball[:, 1]),
            "hit_z_m": _summary(hit_ball[:, 2]),
            "landing_x_m": _summary(landing[:, 0]),
            "landing_y_m": _summary(landing[:, 1]),
            "hit_distance_ball_to_racket_center_m": _summary(hit_dist),
            "racket_speed_at_hit_mps": _summary(racket_speed_hit),
            "ball_in_speed_mps": _summary(ball_in_speed),
            "ball_out_speed_mps": _summary(ball_out_speed),
        },
        "landing_zone_counts": zone_rows,
        "tables": tables,
    }
    report_json = args.output_dir / "competition_core_analysis_report.json"
    report_md = args.output_dir / "competition_core_analysis_report.md"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, report_md)
    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
