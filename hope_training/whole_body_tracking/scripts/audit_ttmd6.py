#!/usr/bin/env python3
"""Audit the local TTMD6 CSV contract without converting it to A3 data.

This script intentionally stops at source validation. It does not assign A3
joint names, choose a hit frame, or generate an NPZ/manifest.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NAME_RE = re.compile(
    r"^(?P<kind>bat|human)_(?P<fps>\d+)_(?P<sample>\d+)_(?P<class_id>\d+)"
    r"_(?P<group_id>\d+)_(?P<stored_len>\d+)_(?P<source_len>\d+)\.csv$"
)

CLASS_HYPOTHESIS = {
    "1": "forehand_attack",
    "2": "forehand_drive",
    "3": "forehand_push",
    "4": "backhand_attack",
    "5": "backhand_drive",
    "6": "backhand_push",
}

SKELETON_EDGES = (
    (0, 1),
    (0, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7),
    (0, 8), (8, 9), (9, 10),
    (0, 11), (11, 12), (12, 13),
)


def parse_name(path: Path) -> dict[str, str]:
    match = NAME_RE.match(path.name)
    if not match:
        raise ValueError(f"unexpected filename: {path.name}")
    return match.groupdict()


def load_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64, encoding="utf-8-sig")
    if values.ndim == 1:
        values = values[None, :]
    return values


def update_range(current: list[float], values: np.ndarray) -> list[float]:
    if values.size == 0:
        return current
    return [min(current[0], float(values.min())), max(current[1], float(values.max()))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="limit files per numeric class for a quick audit; 0 scans all files",
    )
    args = parser.parse_args()

    bat_dir = args.dataset / "TTMD_cut_bat"
    human_dir = args.dataset / "TTMD_cut_hum"
    bat_files = sorted(bat_dir.glob("bat_*.csv"))
    human_files = sorted(human_dir.glob("human_*.csv"))

    report: dict[str, object] = {
        "dataset": str(args.dataset.resolve()),
        "mode": "source_audit_only",
        "training_artifacts_written": False,
        "counts": {"bat_files": len(bat_files), "human_files": len(human_files)},
        "errors": [],
        "warnings": [],
        "class_hypothesis": CLASS_HYPOTHESIS,
        "classes": {},
        "coordinate_observations": {
            "bat_raw_range": [float("inf"), float("-inf")],
            "human_raw_range": [float("inf"), float("-inf")],
            "human_bone_lengths_raw": {},
        },
    }
    errors: list[str] = report["errors"]  # type: ignore[assignment]
    warnings: list[str] = report["warnings"]  # type: ignore[assignment]

    bat_keys = {p.name.removeprefix("bat_").removesuffix(".csv") for p in bat_files}
    human_keys = {p.name.removeprefix("human_").removesuffix(".csv") for p in human_files}
    report["pairing"] = {
        "intersection": len(bat_keys & human_keys),
        "bat_only": len(bat_keys - human_keys),
        "human_only": len(human_keys - bat_keys),
    }

    selected: dict[str, list[Path]] = defaultdict(list)
    for path in bat_files:
        try:
            meta = parse_name(path)
            selected[meta["class_id"]].append(path)
        except ValueError as exc:
            errors.append(str(exc))
    if args.max_per_class > 0:
        selected = {key: paths[: args.max_per_class] for key, paths in selected.items()}  # type: ignore[assignment]

    class_stats: dict[str, dict[str, object]] = {}
    for class_id, paths in sorted(selected.items(), key=lambda item: int(item[0])):
        lengths: list[int] = []
        speeds: list[float] = []
        ranges: list[float] = []
        bone_samples: list[np.ndarray] = []
        for bat_path in paths:
            try:
                meta = parse_name(bat_path)
                human_path = human_dir / bat_path.name.replace("bat_", "human_", 1)
                if not human_path.exists():
                    errors.append(f"missing pair: {human_path}")
                    continue
                bat = load_csv(bat_path)
                human = load_csv(human_path)
                if bat.shape != (int(meta["stored_len"]), 3):
                    errors.append(f"{bat_path.name}: bat shape {bat.shape}")
                if human.shape != (int(meta["stored_len"]), 42):
                    errors.append(f"{human_path.name}: human shape {human.shape}")
                if not np.isfinite(bat).all() or not np.isfinite(human).all():
                    errors.append(f"non-finite values: {bat_path.name}")
                report["coordinate_observations"]["bat_raw_range"] = update_range(  # type: ignore[index]
                    report["coordinate_observations"]["bat_raw_range"], bat  # type: ignore[index]
                )
                report["coordinate_observations"]["human_raw_range"] = update_range(  # type: ignore[index]
                    report["coordinate_observations"]["human_raw_range"], human  # type: ignore[index]
                )

                bat_active = np.any(bat != 0.0, axis=1)
                human_active = np.any(human != 0.0, axis=1)
                source_len = int(meta["source_len"])
                expected_nonzero_rows = min(source_len, int(meta["stored_len"]))
                if int(bat_active.sum()) != expected_nonzero_rows or int(human_active.sum()) != expected_nonzero_rows:
                    warnings.append(
                        f"stored length mismatch: {bat_path.name} "
                        f"source_length={source_len} expected_nonzero={expected_nonzero_rows} "
                        f"bat={int(bat_active.sum())} "
                        f"human={int(human_active.sum())}"
                    )
                if np.any(bat_active != human_active):
                    warnings.append(f"bat/human padding mismatch: {bat_path.name}")

                n = min(expected_nonzero_rows, len(bat), len(human))
                if n > 1:
                    active_bat = bat[:n]
                    velocity = np.linalg.norm(np.diff(active_bat, axis=0) * int(meta["fps"]), axis=1)
                    speeds.append(float(velocity.max()))
                    ranges.append(float(np.ptp(active_bat, axis=0).mean()))
                lengths.append(source_len)

                points = human[:n].reshape(n, 14, 3)
                bone_samples.append(
                    np.asarray(
                        [np.median(np.linalg.norm(points[:, i] - points[:, j], axis=1)) for i, j in SKELETON_EDGES],
                        dtype=np.float64,
                    )
                )
            except (OSError, ValueError) as exc:
                errors.append(f"{bat_path.name}: {exc}")

        stats: dict[str, object] = {
            "file_count_scanned": len(paths),
            "source_length": {
                "min": min(lengths) if lengths else None,
                "median": float(np.median(lengths)) if lengths else None,
                "max": max(lengths) if lengths else None,
            },
            "bat_speed_raw_per_s": {
                "median": float(np.median(speeds)) if speeds else None,
                "p95": float(np.percentile(speeds, 95)) if speeds else None,
            },
            "bat_path_range_raw": {
                "median": float(np.median(ranges)) if ranges else None,
            },
        }
        if bone_samples:
            bones = np.asarray(bone_samples)
            stats["human_bone_lengths_raw_median"] = np.median(bones, axis=0).round(3).tolist()
        class_stats[class_id] = stats

    report["classes"] = class_stats
    if report["coordinate_observations"]["bat_raw_range"][0] == float("inf"):  # type: ignore[index]
        report["coordinate_observations"]["bat_raw_range"] = None  # type: ignore[index]
        report["coordinate_observations"]["human_raw_range"] = None  # type: ignore[index]
    report["status"] = "structural_pass_with_warnings" if not errors else "structural_fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": len(errors), "warnings": len(warnings), "output": str(args.output)}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
