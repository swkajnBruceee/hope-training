#!/usr/bin/env python3
"""Audit CleanSample trajectories for physical plausibility and duplicates."""

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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis.mocap_cleaning.derivative import compute_velocity


def _acceleration(vel: np.ndarray, time: np.ndarray) -> np.ndarray:
    return compute_velocity(vel, time)


def _safe_percentile(values: list[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def _audit_one(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    time = z["time"]
    h = int(z["hit_index"])
    ball_pos = z["ball_pos"]
    ball_vel = z["ball_vel"]
    ball_acc = _acceleration(ball_vel, time)
    racket_pos = z["racket_pos"]
    dist = np.linalg.norm(ball_pos - racket_pos, axis=1)
    ball_speed = np.linalg.norm(ball_vel, axis=1)
    ball_acc_norm = np.linalg.norm(ball_acc, axis=1)
    q = z["racket_quat"]
    q_norm = np.linalg.norm(q, axis=1)
    omega = np.linalg.norm(z["racket_omega"], axis=1)
    flags = json.loads(str(z["quality_flags_json"]))
    source = json.loads(str(z["source_json"]))

    reasons = []
    if len(time) != 201:
        reasons.append("bad_frame_count")
    if h != 120 or abs(float(z["time_rel"][h])) > 1e-9:
        reasons.append("bad_hit_alignment")
    if not np.isfinite(ball_pos).all():
        reasons.append("nonfinite_ball_pos")
    if not np.isfinite(ball_vel).all():
        reasons.append("nonfinite_ball_vel")
    if float(np.nanmax(ball_speed)) >= 50.0:
        reasons.append("ball_speed_ge_50")
    if float(np.nanpercentile(ball_acc_norm, 99)) >= 1000.0:
        reasons.append("ball_acc_p99_ge_1000")
    if dist[h] >= 0.12:
        reasons.append("hit_distance_ge_012")
    if np.nanmin(dist[max(0, h - 20): h + 21]) >= 0.12:
        reasons.append("no_near_racket_around_hit")
    if np.nanmax(np.abs(q_norm - 1.0)) > 1e-6:
        reasons.append("quat_not_unit")
    if float(np.nanmax(omega)) >= 40.0:
        reasons.append("racket_omega_ge_40")
    if not bool(flags.get("usable_for_training", False)):
        reasons.append("flagged_not_usable")

    return {
        "episode_id": str(z["episode_id"]),
        "sample_path": str(path),
        "source_csv": source["source_csv"],
        "racket": source["racket"],
        "stroke_type": str(z["stroke_type"]),
        "hit_time": float(z["hit_time"]),
        "hit_index": h,
        "usable_for_training": bool(flags.get("usable_for_training", False)),
        "max_ball_speed_mps": float(np.nanmax(ball_speed)),
        "p95_ball_speed_mps": float(np.nanpercentile(ball_speed, 95)),
        "p99_ball_acc_mps2": float(np.nanpercentile(ball_acc_norm, 99)),
        "max_ball_acc_mps2": float(np.nanmax(ball_acc_norm)),
        "hit_distance_m": float(dist[h]),
        "min_distance_near_hit_m": float(np.nanmin(dist[max(0, h - 20): h + 21])),
        "max_racket_omega_radps": float(np.nanmax(omega)),
        "max_quat_norm_error": float(np.nanmax(np.abs(q_norm - 1.0))),
        "reasons": reasons,
        "audit_ok": len(reasons) == 0,
    }


def _duplicate_groups(items: list[dict[str, Any]], min_sep_s: float) -> list[dict[str, Any]]:
    groups = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_key[(item["source_csv"], item["racket"])].append(item)
    for (csv, racket), rows in by_key.items():
        rows = sorted(rows, key=lambda x: x["hit_time"])
        current = []
        for row in rows:
            if not current or row["hit_time"] - current[-1]["hit_time"] <= min_sep_s:
                current.append(row)
            else:
                if len(current) > 1:
                    groups.append(
                        {
                            "source_csv": csv,
                            "racket": racket,
                            "count": len(current),
                            "hit_times": [x["hit_time"] for x in current],
                            "episode_ids": [x["episode_id"] for x in current],
                        }
                    )
                current = [row]
        if len(current) > 1:
            groups.append(
                {
                    "source_csv": csv,
                    "racket": racket,
                    "count": len(current),
                    "hit_times": [x["hit_time"] for x in current],
                    "episode_ids": [x["episode_id"] for x in current],
                }
            )
    return groups


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# CleanSample Audit Report",
        "",
        f"Input manifest: `{report['manifest']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Samples | {report['samples']} |",
        f"| Audit OK | {report['audit_ok_count']} |",
        f"| Audit rejected | {report['audit_rejected_count']} |",
        f"| Duplicate groups | {len(report['duplicate_groups'])} |",
        f"| Ball speed p95 of sample max | {report['summary']['max_ball_speed_p95']:.3f} |",
        f"| Ball acceleration p95 of sample p99 | {report['summary']['p99_ball_acc_p95']:.3f} |",
        "",
        "## Rejection Reasons",
        "",
    ]
    if report["reason_counts"]:
        for reason, count in sorted(report["reason_counts"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Worst Samples", ""])
    for item in report["worst_samples"][:20]:
        lines.append(
            f"- `{item['episode_id']}`: audit_ok={item['audit_ok']}, "
            f"hit_dist={item['hit_distance_m']:.3f}, max_ball_speed={item['max_ball_speed_mps']:.2f}, "
            f"p99_acc={item['p99_ball_acc_mps2']:.1f}, max_omega={item['max_racket_omega_radps']:.1f}, "
            f"reasons={','.join(item['reasons']) or 'none'}"
        )
    if report["duplicate_groups"]:
        lines.extend(["", "## Duplicate-Like Groups", ""])
        for group in report["duplicate_groups"][:20]:
            lines.append(
                f"- `{group['source_csv']}` {group['racket']}: {group['count']} hits, "
                f"times={[round(x, 3) for x in group['hit_times']]}"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duplicate-separation-s", type=float, default=0.35)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    items = [_audit_one(Path(sample["sample_path"])) for sample in manifest["samples"]]
    reason_counts = Counter(reason for item in items for reason in item["reasons"])
    duplicate_groups = _duplicate_groups(items, args.duplicate_separation_s)
    worst_samples = sorted(
        items,
        key=lambda x: (
            len(x["reasons"]),
            x["hit_distance_m"],
            x["p99_ball_acc_mps2"],
            x["max_ball_speed_mps"],
        ),
        reverse=True,
    )
    report = {
        "manifest": str(args.manifest),
        "samples": len(items),
        "audit_ok_count": sum(item["audit_ok"] for item in items),
        "audit_rejected_count": sum(not item["audit_ok"] for item in items),
        "reason_counts": dict(reason_counts),
        "summary": {
            "max_ball_speed_p50": _safe_percentile([x["max_ball_speed_mps"] for x in items], 50),
            "max_ball_speed_p95": _safe_percentile([x["max_ball_speed_mps"] for x in items], 95),
            "p99_ball_acc_p50": _safe_percentile([x["p99_ball_acc_mps2"] for x in items], 50),
            "p99_ball_acc_p95": _safe_percentile([x["p99_ball_acc_mps2"] for x in items], 95),
            "hit_distance_p95": _safe_percentile([x["hit_distance_m"] for x in items], 95),
        },
        "duplicate_groups": duplicate_groups,
        "worst_samples": worst_samples,
        "items": items,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "sample_audit_report.json"
    md_path = args.output_dir / "sample_audit_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Audited {len(items)} samples")
    print(f"Audit OK: {report['audit_ok_count']}")
    print(f"Audit rejected: {report['audit_rejected_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
