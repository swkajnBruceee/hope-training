#!/usr/bin/env python3
"""Audit target-conditioned coverage across canonical 10-D motion manifests.

This is a read-only audit.  It deliberately reports marginal and pairwise
coverage instead of pretending that a sparse finite bank covers every point
in a 10-D continuous space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DIMENSIONS = (
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "velocity_x_mps",
    "velocity_y_mps",
    "velocity_z_mps",
    "normal_x",
    "normal_y",
    "normal_z",
    "time_to_hit_s",
)


def _goal_from_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    goal = entry.get("canonical_goal_10d")
    if not isinstance(goal, dict):
        target = entry.get("strike_target")
        if isinstance(target, dict):
            goal = {
                "position_m": target.get("racket_position_m"),
                "linear_velocity_mps": target.get("racket_velocity_mps"),
                "normal_w": target.get("racket_normal_w"),
                "time_to_hit_s": entry.get("hit_event", {}).get("strike_time_s"),
            }
    if not isinstance(goal, dict):
        return None
    try:
        position = [float(x) for x in goal["position_m"]]
        velocity = [float(x) for x in goal["linear_velocity_mps"]]
        normal = [float(x) for x in (goal.get("normal_w") or goal["racket_normal"])]
        strike_time_value = goal.get("time_to_hit_s")
        if strike_time_value is None:
            strike_time_value = goal["time_to_strike_s"]
        strike_time = float(strike_time_value)
        if len(position) != 3 or len(velocity) != 3 or len(normal) != 3:
            return None
        values = position + velocity + normal + [strike_time]
        if not all(math.isfinite(x) for x in values):
            return None
        return {"values": values, "stroke": str(entry.get("stroke_type", entry.get("swing_type", "unknown")))}
    except (KeyError, TypeError, ValueError):
        return None


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(payload.get("motions", [])):
        goal = _goal_from_entry(entry)
        if goal is not None:
            rows.append({"source": str(path), "source_index": index, **goal})
    return rows


def _load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            goal_path = Path(row["normalized_goal_json"]).expanduser()
            if not goal_path.is_file():
                continue
            goal = json.loads(goal_path.read_text(encoding="utf-8"))
            item = _goal_from_entry({"canonical_goal_10d": {
                "position_m": goal["position_m"],
                "linear_velocity_mps": goal["linear_velocity_mps"],
                "normal_w": goal["racket_normal"],
                "time_to_hit_s": goal["time_to_strike_s"],
            }, "stroke_type": row.get("stroke", "unknown")})
            if item is not None:
                rows.append({"source": str(path), "source_index": index, **item})
    return rows


def _summary(rows: list[dict[str, Any]], bins: int) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows), "unique_target_count": 0, "ranges": {}, "marginal_bins": {}}
    if not rows:
        return result
    values = np.asarray([row["values"] for row in rows], dtype=np.float64)
    keys = [tuple(np.round(row["values"], 4)) for row in rows]
    result["unique_target_count"] = len(set(keys))
    for dim, column in zip(DIMENSIONS, values.T):
        low, high = float(column.min()), float(column.max())
        result["ranges"][dim] = {"min": low, "max": high}
        if math.isclose(low, high):
            result["marginal_bins"][dim] = {"occupied": 1, "empty": bins - 1, "edges": [low, high]}
            continue
        edges = np.linspace(low, high, bins + 1)
        occupied = set(np.clip(np.digitize(column, edges[1:-1], right=False), 0, bins - 1).tolist())
        result["marginal_bins"][dim] = {
            "occupied": len(occupied),
            "empty": bins - len(occupied),
            "edges": [float(x) for x in edges],
        }
    pair_specs = {
        "position_xy": (0, 1),
        "position_xz": (0, 2),
        "position_yz": (1, 2),
        "velocity_xy": (3, 4),
        "velocity_xz": (3, 5),
        "normal_xy": (6, 7),
        "normal_xz": (6, 8),
    }
    result["pairwise_bins"] = {}
    for name, (i, j) in pair_specs.items():
        occupied: set[tuple[int, int]] = set()
        for column_i, column_j in zip(values[:, i], values[:, j]):
            low_i, high_i = values[:, i].min(), values[:, i].max()
            low_j, high_j = values[:, j].min(), values[:, j].max()
            bi = 0 if math.isclose(low_i, high_i) else int(np.clip((column_i - low_i) / (high_i - low_i) * bins, 0, bins - 1))
            bj = 0 if math.isclose(low_j, high_j) else int(np.clip((column_j - low_j) / (high_j - low_j) * bins, 0, bins - 1))
            occupied.add((bi, bj))
        result["pairwise_bins"][name] = {"occupied": len(occupied), "empty": bins * bins - len(occupied), "total": bins * bins}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--csv", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()
    if not args.manifest and not args.csv:
        raise SystemExit("at least one --manifest or --csv is required")
    if args.bins < 2:
        raise SystemExit("--bins must be >= 2")

    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for path in args.manifest:
        loaded = _load_manifest(path.expanduser().resolve())
        rows.extend(loaded)
        source_counts[str(path.expanduser().resolve())] += len(loaded)
    for path in args.csv:
        loaded = _load_csv(path.expanduser().resolve())
        rows.extend(loaded)
        source_counts[str(path.expanduser().resolve())] += len(loaded)

    by_stroke: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stroke[row["stroke"]].append(row)
    dedup_keys = {(row["stroke"], tuple(np.round(row["values"], 4))) for row in rows}
    payload = {
        "schema_version": "a3_target_space_coverage_audit/v1",
        "status": "completed",
        "goal_dimensions": list(DIMENSIONS),
        "bin_count": args.bins,
        "source_counts": dict(source_counts),
        "total_loaded": len(rows),
        "unique_target_count_by_stroke": {stroke: len({tuple(np.round(r["values"], 4)) for r in values}) for stroke, values in by_stroke.items()},
        "unique_target_count_union": len(dedup_keys),
        "by_stroke": {stroke: _summary(values, args.bins) for stroke, values in sorted(by_stroke.items())},
        "interpretation": {
            "coverage_is_not_full_10d_guarantee": True,
            "empty_marginal_bins_are_regeneration_candidates": True,
            "pairwise_bins_are_diagnostics_not_hard_rejection": True,
            "normal_vector_must_remain_unit_length": True,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "total_loaded": len(rows), "unique_target_count_union": len(dedup_keys), "strokes": {k: len(v) for k, v in sorted(by_stroke.items())}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
