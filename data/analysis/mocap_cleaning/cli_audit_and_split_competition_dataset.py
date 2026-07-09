#!/usr/bin/env python3
"""Audit and split the final competition-table dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TABLE_LENGTH_M = 2.74
TABLE_WIDTH_M = 1.525
NET_X_M = TABLE_LENGTH_M / 2.0
Y_MIN_M = -TABLE_WIDTH_M
Y_MAX_M = 0.0


def _json_list(values: np.ndarray) -> list[dict[str, Any]]:
    return [json.loads(str(v)) for v in values]


def _counter(values: list[Any] | np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(values).items()}


def _percentiles(values: np.ndarray, qs: tuple[int, ...] = (1, 10, 50, 90, 99)) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {f"p{q:02d}": float("nan") for q in qs}
    return {f"p{q:02d}": float(np.percentile(vals, q)) for q in qs}


def _transform_report_by_csv(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    report = json.loads(path.read_text())
    out = {}
    for item in report.get("files", []):
        out[Path(item["csv"]).name] = item
    return out


def _subset_arrays(data: np.lib.npyio.NpzFile, indices: np.ndarray, subset_name: str) -> dict[str, np.ndarray]:
    n = int(data["ball_pos"].shape[0])
    out: dict[str, np.ndarray] = {}
    for key in data.files:
        value = data[key]
        if value.shape and value.shape[0] == n:
            out[key] = value[indices]
        else:
            out[key] = value
    attrs = json.loads(str(out["dataset_attrs_json"]))
    attrs["subset_name"] = subset_name
    attrs["subset_sample_count"] = int(len(indices))
    attrs["subset_source_indices"] = "stored in source_index"
    out["dataset_attrs_json"] = np.asarray(json.dumps(attrs, ensure_ascii=False))
    out["source_index"] = indices.astype(np.int64)
    return out


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


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
        "# DATA260708 Final Competition Dataset Audit",
        "",
        f"Input: `{report['input_dataset']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Samples | {report['samples']} |",
        f"| Frames | {report['frames']} |",
        f"| Success | {report['success_counts'].get('1', 0)} |",
        f"| Failure | {report['success_counts'].get('0', 0)} |",
        f"| Unknown | {report['success_counts'].get('-1', 0)} |",
        f"| Analysis core | {report['subsets']['analysis_core']['count']} |",
        f"| Training motion | {report['subsets']['training_motion']['count']} |",
        f"| QC review | {report['subsets']['qc_review']['count']} |",
        "",
        "## Subset Rules",
        "",
        "- `analysis_core`: reliable success/failure label, finite hit-frame racket/ball data, hit distance <= 0.15 m, and normal landing height.",
        "- `training_motion`: usable motion sample not in `analysis_core` or `qc_review`; may have unknown success label.",
        "- `qc_review`: unknown success, abnormal landing height/ball trajectory, or suspicious hit distance. Valid failures are kept in `analysis_core`.",
        "",
        "## Racket Reference Point",
        "",
        report["racket_reference_note"],
        "",
        "## Output Files",
        "",
        "| Subset | Path |",
        "|---|---|",
    ]
    for name, item in report["subsets"].items():
        lines.append(f"| `{name}` | `{item['path']}` |")
    lines.extend(["", "## Success By Racket", ""])
    lines.append("| racket | success | failure | unknown | total |")
    lines.append("|---|---:|---:|---:|---:|")
    for racket, row in sorted(report["success_by_racket"].items()):
        lines.append(f"| {racket} | {row.get('1', 0)} | {row.get('0', 0)} | {row.get('-1', 0)} | {sum(row.values())} |")
    lines.extend(["", "## Key Percentiles", ""])
    for name, stats in report["percentiles"].items():
        values = ", ".join(f"{k}={v:.4f}" for k, v in stats.items())
        lines.append(f"- `{name}`: {values}")
    lines.extend(["", "## QC Review", ""])
    lines.append(f"QC rows: {report['qc_review_count']}")
    lines.append(f"QC CSV: `{report['qc_review_csv']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ball-report",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260708/ball_reconstruction/unlabeled_ball_reconstruction_report.json"),
    )
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    n = int(data["ball_pos"].shape[0])
    hit_index = data["hit_index"].astype(int)
    success = data["success"].astype(int)
    sources = _json_list(data["source_json"])
    quality = _json_list(data["quality_flags_json"])
    attrs = json.loads(str(data["dataset_attrs_json"]))

    hit_ball = data["ball_pos"][np.arange(n), hit_index]
    hit_racket = data["racket_pos"][np.arange(n), hit_index]
    hit_dist = np.linalg.norm(hit_ball - hit_racket, axis=1)
    landing_index = data["landing_index"] if "landing_index" in data.files else np.full(n, -1)
    landing_pos = data["landing_pos"]
    landing_detected = landing_index >= 0
    landing_in_table = (
        landing_detected
        & (landing_pos[:, 0] >= 0.0)
        & (landing_pos[:, 0] <= TABLE_LENGTH_M)
        & (landing_pos[:, 1] >= Y_MIN_M)
        & (landing_pos[:, 1] <= Y_MAX_M)
        & (np.abs(landing_pos[:, 2]) <= 0.08)
    )
    finite_hit = np.isfinite(hit_ball).all(axis=1) & np.isfinite(hit_racket).all(axis=1)
    suspicious_hit_dist = hit_dist > 0.15
    abnormal_landing = landing_detected & (np.abs(landing_pos[:, 2]) > 0.08)
    unknown_success = success == -1

    qc_mask = unknown_success | suspicious_hit_dist | abnormal_landing
    analysis_core_mask = (~qc_mask) & np.isin(success, [0, 1]) & finite_hit
    training_motion_mask = (~analysis_core_mask) & (~qc_mask)

    ball_report_by_csv: dict[str, dict[str, Any]] = {}
    if args.ball_report.exists():
        ball_report = json.loads(args.ball_report.read_text())
        for item in ball_report.get("files", []):
            ball_report_by_csv[Path(item["csv"]).name] = item

    qc_rows = []
    for i in np.where(qc_mask)[0]:
        source = sources[int(i)]
        csv_name = Path(source.get("source_csv", "")).name
        ball_item = ball_report_by_csv.get(csv_name, {})
        reasons = []
        if unknown_success[int(i)]:
            reasons.append("unknown_success")
        if suspicious_hit_dist[int(i)]:
            reasons.append("hit_distance_gt_0p15m")
        if abnormal_landing[int(i)]:
            reasons.append("abnormal_landing")
        qc_rows.append(
            {
                "source_index": int(i),
                "episode_id": str(data["episode_id"][i]),
                "source_csv": source.get("source_csv", ""),
                "racket": source.get("racket", ""),
                "success": int(success[i]),
                "qc_reasons": ";".join(reasons),
                "hit_dist_m": float(hit_dist[i]),
                "landing_index": int(landing_index[i]),
                "landing_x": float(landing_pos[i, 0]) if np.isfinite(landing_pos[i, 0]) else "",
                "landing_y": float(landing_pos[i, 1]) if np.isfinite(landing_pos[i, 1]) else "",
                "landing_z": float(landing_pos[i, 2]) if np.isfinite(landing_pos[i, 2]) else "",
                "ball_non_rule_links_for_csv": int(ball_item.get("non_rule_link_count", 0)),
                "ball_decision_for_csv": ball_item.get("decision", ""),
            }
        )

    subsets = {
        "analysis_core": np.where(analysis_core_mask)[0],
        "training_motion": np.where(training_motion_mask)[0],
        "qc_review": np.where(qc_mask)[0],
    }
    subset_reports = {}
    for name, indices in subsets.items():
        out_path = args.output_dir / f"{Path(args.dataset).stem}_{name}.npz"
        _write_npz(out_path, _subset_arrays(data, indices.astype(np.int64), name))
        subset_reports[name] = {"count": int(len(indices)), "path": str(out_path)}

    success_by_racket: dict[str, dict[str, int]] = {}
    for racket in sorted(set(str(s.get("racket", "unknown")) for s in sources)):
        mask = np.asarray([str(s.get("racket", "unknown")) == racket for s in sources])
        success_by_racket[racket] = _counter(success[mask])

    source_csv_success: dict[str, dict[str, int]] = {}
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, source in enumerate(sources):
        grouped[str(source.get("source_csv", "unknown"))].append(i)
    for csv_name, indices in grouped.items():
        idx = np.asarray(indices, dtype=int)
        source_csv_success[csv_name] = _counter(success[idx])

    qc_csv = args.output_dir / "qc_review_samples.csv"
    _write_csv(qc_csv, qc_rows)

    report = {
        "input_dataset": str(args.dataset),
        "samples": n,
        "frames": int(data["ball_pos"].shape[1]),
        "coordinate_frame": attrs.get("coordinate_frame"),
        "success_counts": _counter(success),
        "stroke_type_rule_v2_counts": _counter(data["stroke_type_rule_v2"]) if "stroke_type_rule_v2" in data.files else {},
        "success_by_racket": success_by_racket,
        "source_csv_success_counts": source_csv_success,
        "subsets": subset_reports,
        "qc_review_count": len(qc_rows),
        "qc_review_csv": str(qc_csv),
        "percentiles": {
            "hit_dist_m": _percentiles(hit_dist),
            "landing_x_m": _percentiles(landing_pos[landing_detected, 0]),
            "landing_y_m": _percentiles(landing_pos[landing_detected, 1]),
            "landing_z_m": _percentiles(landing_pos[landing_detected, 2]),
            "ball_speed_mps": _percentiles(np.linalg.norm(data["ball_vel"], axis=2).reshape(-1)),
            "racket_speed_mps": _percentiles(np.linalg.norm(data["racket_vel"], axis=2).reshape(-1)),
        },
        "quality_counts": {
            "landing_detected": int(np.sum(landing_detected)),
            "landing_in_table": int(np.sum(landing_in_table)),
            "suspicious_hit_dist_gt_0p15m": int(np.sum(suspicious_hit_dist)),
            "abnormal_landing": int(np.sum(abnormal_landing)),
            "unknown_success": int(np.sum(unknown_success)),
            "usable_for_training_flag_true": int(sum(bool(q.get("usable_for_training")) for q in quality)),
        },
        "racket_reference_note": (
            "racket_pos is the Motive rigid-body center inside the racket, not the physical ball-contact point. "
            "Hit distance is ball center to racket rigid-body center, so nonzero distance is expected."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.output_dir / "final_dataset_audit_report.json"
    report_md = args.output_dir / "final_dataset_audit_report.md"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, report_md)
    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")
    print(f"Wrote {qc_csv}")
    for name, item in subset_reports.items():
        print(f"{name}: {item['count']} -> {item['path']}")


if __name__ == "__main__":
    main()
