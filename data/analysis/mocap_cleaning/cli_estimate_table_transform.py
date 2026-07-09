#!/usr/bin/env python3
"""Estimate table-frame transforms from Motive table corner markers."""

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
from analysis.mocap_cleaning.motive_loader import load_motive_csv
from analysis.mocap_cleaning.units import position_scale_to_meters


def _finite_mean(pos: np.ndarray) -> np.ndarray:
    finite = np.isfinite(pos).all(axis=1)
    if not np.any(finite):
        return np.full(3, np.nan)
    return np.nanmedian(pos[finite], axis=0)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.full_like(v, np.nan)
    return v / norm


def estimate_transform(marker_points: list[np.ndarray], table_cfg: dict[str, Any]) -> dict[str, Any]:
    p1, p2, p3, p4 = marker_points
    x_axis = _normalize(p2 - p1)
    # Formal competition convention:
    # Marker 001 is P1/liang-side left origin, Marker 004 is P1-side right.
    # The positive Y axis points to P1's left, so Marker 004 has Y=-width.
    y_negative_axis = _normalize(p4 - p1)
    z_axis = _normalize(np.cross(x_axis, -y_negative_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    rotation_table_to_motive = np.stack([x_axis, y_axis, z_axis], axis=1)
    rotation_motive_to_table = rotation_table_to_motive.T
    origin = p1
    length = float(table_cfg["length_m"])
    width = float(table_cfg["width_m"])
    return {
        "origin_motive_m": [float(x) for x in origin],
        "rotation_table_to_motive": rotation_table_to_motive.tolist(),
        "rotation_motive_to_table": rotation_motive_to_table.tolist(),
        "coordinate_frame": "competition_table_m",
        "corner_mapping": {
            "table:Marker 001": [0.0, 0.0, 0.0],
            "table:Marker 002": [length, 0.0, 0.0],
            "table:Marker 003": [length, -width, 0.0],
            "table:Marker 004": [0.0, -width, 0.0],
        },
        "transform_note": "p_table = rotation_motive_to_table @ (p_motive - origin_motive_m)",
    }


def analyze_csv(path: Path, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    table_cfg = config["table"]
    marker_names = list(table_cfg["markers"])
    trial = load_motive_csv(path, rigid_bodies=[str(table_cfg["rigid_body"])], markers=marker_names)
    scale = position_scale_to_meters(trial.position_unit)
    points = [_finite_mean(trial.markers[name] * scale) for name in marker_names if name in trial.markers]
    missing = [name for name in marker_names if name not in trial.markers]
    geometry_ok = False
    transform: dict[str, Any] | None = None
    distances: dict[str, float] = {}
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing markers: {missing}")
    if len(points) == 4 and all(np.isfinite(p).all() for p in points):
        p1, p2, p3, p4 = points
        distances = {
            "m1_m2_m": float(np.linalg.norm(p2 - p1)),
            "m2_m3_m": float(np.linalg.norm(p3 - p2)),
            "m3_m4_m": float(np.linalg.norm(p4 - p3)),
            "m4_m1_m": float(np.linalg.norm(p1 - p4)),
            "diagonal_m1_m3_m": float(np.linalg.norm(p3 - p1)),
            "diagonal_m2_m4_m": float(np.linalg.norm(p4 - p2)),
        }
        length = float(table_cfg["length_m"])
        width = float(table_cfg["width_m"])
        tol = float(table_cfg["geometry_tolerance_m"])
        length_ok = abs(distances["m1_m2_m"] - length) <= tol and abs(distances["m3_m4_m"] - length) <= tol
        width_ok = abs(distances["m2_m3_m"] - width) <= tol and abs(distances["m4_m1_m"] - width) <= tol
        geometry_ok = bool(length_ok and width_ok)
        if not geometry_ok:
            reasons.append("table marker distances outside tolerance")
        transform = estimate_transform(points, table_cfg)
    else:
        reasons.append("not enough finite table marker positions")

    output_dir.mkdir(parents=True, exist_ok=True)
    transform_path = output_dir / f"{path.stem}_table_transform.json"
    item = {
        "csv": str(path),
        "take_name": trial.take_name,
        "markers": marker_names,
        "marker_points_motive_m": [[float(x) for x in p] for p in points],
        "distances": distances,
        "geometry_ok": geometry_ok,
        "transform_path": str(transform_path),
        "transform": transform,
        "reasons": reasons or ["ok"],
    }
    transform_path.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n")
    return item


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# DATA260708 Competition Table Transform Report",
        "",
        "Coordinate convention: Marker001=(0,0,0), Marker002=(+length,0,0), Marker003=(+length,-width,0), Marker004=(0,-width,0).",
        "",
        "| CSV | Geometry OK | Length A | Length B | Width A | Width B | Reasons |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in report["files"]:
        d = item["distances"]
        lines.append(
            f"| `{Path(item['csv']).name}` | {item['geometry_ok']} | "
            f"{d.get('m1_m2_m', float('nan')):.3f} | {d.get('m3_m4_m', float('nan')):.3f} | "
            f"{d.get('m2_m3_m', float('nan')):.3f} | {d.get('m4_m1_m', float('nan')):.3f} | "
            f"{'; '.join(item['reasons'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/DATA260708.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/mocap_cleaning_outputs/DATA260708/table_transforms"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])
    csv_dir = dataset_root / config["source_layout"].get("csv_dir", "CSV")
    paths = sorted(csv_dir.glob("*.csv"))
    if args.limit > 0:
        paths = paths[: args.limit]
    files = [analyze_csv(path, config, args.output_dir) for path in paths]
    report = {"config": str(args.config), "csv_dir": str(csv_dir), "files": files}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "table_transform_report.json"
    md_path = args.output_dir / "table_transform_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
