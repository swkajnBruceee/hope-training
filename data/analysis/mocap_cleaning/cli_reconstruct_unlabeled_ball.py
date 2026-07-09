#!/usr/bin/env python3
"""Reconstruct a ball trajectory from Motive unlabeled marker IDs.

DATA260708 exports the ball as ``Unlabeled N`` markers. Motive can assign a new
ID when the ball leaves and re-enters the tracked volume, usually N+1 and
occasionally N+2. This tool builds a single best-effort trajectory and records
the source marker IDs used at every frame for auditability.
"""

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
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis.mocap_cleaning.config import load_config
from analysis.mocap_cleaning.derivative import compute_velocity
from analysis.mocap_cleaning.motive_loader import list_entities, load_motive_csv, read_motive_header
from analysis.mocap_cleaning.trajectory_cleaning import clean_position_trajectory
from analysis.mocap_cleaning.units import position_scale_to_meters


_UNLABELED_RE = re.compile(r"^Unlabeled\s+(\d+)$")


@dataclass
class LinkEvent:
    frame: int
    time_s: float
    from_id: str | None
    to_id: str
    gap_s: float
    prediction_error_m: float
    id_delta: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _marker_id(name: str) -> int | None:
    match = _UNLABELED_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def _unlabeled_marker_names(path: Path, prefix: str) -> list[str]:
    entities = list_entities(read_motive_header(path))
    return sorted(name for name in entities.get("Marker", []) if name.startswith(prefix))


def _finite_rows(pos: np.ndarray) -> np.ndarray:
    return np.isfinite(pos).all(axis=1)


def _dynamic_marker_names(
    markers_m: dict[str, np.ndarray],
    time: np.ndarray,
    min_dynamic_speed_mps: float,
    max_static_ratio: float,
) -> list[str]:
    names = []
    for name, pos in markers_m.items():
        finite = _finite_rows(pos)
        if int(np.sum(finite)) < 4:
            continue
        vel = compute_velocity(pos, time)
        speed = np.linalg.norm(vel, axis=1)
        finite_speed = speed[np.isfinite(speed)]
        if len(finite_speed) == 0:
            continue
        p95 = float(np.nanpercentile(finite_speed, 95))
        static_ratio = float(np.mean(finite_speed < 0.05))
        if p95 >= min_dynamic_speed_mps and static_ratio <= max_static_ratio:
            names.append(name)
    return names


def _nearest_racket_distance(point: np.ndarray, racket_positions: dict[str, np.ndarray], frame: int) -> float:
    best = math.inf
    for pos in racket_positions.values():
        if frame < len(pos) and np.isfinite(pos[frame]).all():
            best = min(best, float(np.linalg.norm(point - pos[frame])))
    return best


def reconstruct_unlabeled_ball(
    *,
    time: np.ndarray,
    markers_m: dict[str, np.ndarray],
    racket_positions_m: dict[str, np.ndarray],
    max_id_jump: int,
    max_link_gap_s: float,
    max_link_error_m: float,
    same_id_bonus_m: float,
    id_jump_bonus_m: float,
) -> tuple[np.ndarray, np.ndarray, list[LinkEvent]]:
    n = len(time)
    out = np.full((n, 3), np.nan)
    source = np.full(n, "", dtype="<U64")
    events: list[LinkEvent] = []
    last_frame: int | None = None
    last_name: str | None = None
    last_pos: np.ndarray | None = None
    last_vel = np.zeros(3, dtype=float)

    names = sorted(markers_m)
    for frame in range(n):
        candidates: list[tuple[float, str, np.ndarray, float, int | None]] = []
        for name in names:
            pos = markers_m[name][frame]
            if not np.isfinite(pos).all():
                continue

            near_bonus = min(_nearest_racket_distance(pos, racket_positions_m, frame), 1.0) * 0.05
            id_delta = None
            if last_frame is None or last_pos is None:
                score = near_bonus
            else:
                gap_s = float(time[frame] - time[last_frame])
                predicted = last_pos + last_vel * gap_s
                err = float(np.linalg.norm(pos - predicted))
                score = err + near_bonus
                last_id = _marker_id(last_name or "")
                this_id = _marker_id(name)
                if last_name == name:
                    score -= same_id_bonus_m
                    id_delta = 0
                elif last_id is not None and this_id is not None:
                    id_delta = this_id - last_id
                    if 1 <= id_delta <= max_id_jump:
                        score -= id_jump_bonus_m
                if gap_s > max_link_gap_s and err > max_link_error_m:
                    continue
            candidates.append((score, name, pos, _nearest_racket_distance(pos, racket_positions_m, frame), id_delta))

        if not candidates:
            continue

        score, name, pos, _, id_delta = min(candidates, key=lambda item: item[0])
        if last_frame is not None and last_pos is not None:
            gap_s = float(time[frame] - time[last_frame])
            predicted = last_pos + last_vel * gap_s
            err = float(np.linalg.norm(pos - predicted))
            if name != last_name:
                events.append(
                    LinkEvent(
                        frame=frame,
                        time_s=float(time[frame]),
                        from_id=last_name,
                        to_id=name,
                        gap_s=gap_s,
                        prediction_error_m=err,
                        id_delta=id_delta,
                    )
                )
            if gap_s > 0:
                last_vel = (pos - last_pos) / gap_s
        out[frame] = pos
        source[frame] = name
        last_frame = frame
        last_name = name
        last_pos = pos

    return out, source, events


def _segments(source: np.ndarray, min_segment_frames: int) -> list[dict[str, Any]]:
    segments = []
    start = None
    current = ""
    for idx, name in enumerate(source.tolist() + [""]):
        if name != current:
            if current and start is not None and idx - start >= min_segment_frames:
                segments.append({"source_id": current, "start_frame": start, "end_frame": idx - 1, "frames": idx - start})
            start = idx if name else None
            current = name
    return segments


def analyze_csv(path: Path, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    cfg = config["unlabeled_ball"]
    racket_names = list(config["entities"]["rackets"].keys())
    marker_names = _unlabeled_marker_names(path, str(cfg["marker_prefix"]))
    trial = load_motive_csv(path, rigid_bodies=racket_names, markers=marker_names)
    scale = position_scale_to_meters(trial.position_unit)
    markers_m = {name: pos * scale for name, pos in trial.markers.items()}
    dynamic_names = _dynamic_marker_names(
        markers_m,
        trial.time,
        min_dynamic_speed_mps=float(cfg["min_dynamic_speed_mps"]),
        max_static_ratio=float(cfg["max_static_ratio"]),
    )
    dynamic_markers = {name: markers_m[name] for name in dynamic_names}
    racket_positions_m = {
        name: trial.rigid_bodies[name].pos * scale for name in racket_names if name in trial.rigid_bodies
    }
    raw_ball, source_ids, link_events = reconstruct_unlabeled_ball(
        time=trial.time,
        markers_m=dynamic_markers,
        racket_positions_m=racket_positions_m,
        max_id_jump=int(cfg["max_id_jump"]),
        max_link_gap_s=float(cfg["max_link_gap_s"]),
        max_link_error_m=float(cfg["max_link_error_m"]),
        same_id_bonus_m=float(cfg["same_id_bonus_m"]),
        id_jump_bonus_m=float(cfg["id_jump_bonus_m"]),
    )
    clean_ball, cleaning = clean_position_trajectory(
        raw_ball,
        trial.time,
        max_speed_mps=float(config["speed_thresholds"]["ball_mps"]),
        max_gap_s=float(config["gap_policy"]["interpolate_max_s"]),
        min_valid_ratio=0.10,
    )
    ball_vel = compute_velocity(clean_ball, trial.time)
    speed = np.linalg.norm(ball_vel, axis=1)
    valid = _finite_rows(clean_ball)
    id_rule_links = [
        event for event in link_events if event.id_delta is not None and 0 <= event.id_delta <= int(cfg["max_id_jump"])
    ]
    non_rule_links = [
        event for event in link_events if event.id_delta is None or event.id_delta < 0 or event.id_delta > int(cfg["max_id_jump"])
    ]
    clean_valid_ratio = float(np.mean(valid)) if len(valid) else 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    npz_path = output_dir / f"{stem}_unlabeled_ball.npz"
    np.savez_compressed(
        npz_path,
        source_csv=np.asarray(str(path)),
        take_name=np.asarray(trial.take_name),
        time=trial.time,
        ball_pos_raw=raw_ball,
        ball_pos_clean=clean_ball,
        ball_vel=ball_vel,
        source_unlabeled_id=source_ids,
    )
    return {
        "csv": str(path),
        "take_name": trial.take_name,
        "npz_path": str(npz_path),
        "unlabeled_markers_total": len(marker_names),
        "dynamic_markers_used": len(dynamic_names),
        "raw_valid_ratio": float(np.mean(_finite_rows(raw_ball))) if len(raw_ball) else 0.0,
        "clean_valid_ratio": clean_valid_ratio,
        "speed_p95_mps": float(np.nanpercentile(speed[np.isfinite(speed)], 95)) if np.isfinite(speed).any() else float("nan"),
        "speed_max_mps": float(np.nanmax(speed)) if np.isfinite(speed).any() else float("nan"),
        "segments": _segments(source_ids, int(cfg["min_segment_frames"])),
        "id_link_events": [event.to_dict() for event in link_events],
        "id_rule_link_count": len(id_rule_links),
        "non_rule_link_count": len(non_rule_links),
        "cleaning": cleaning.to_dict(),
        "decision": "usable_for_hit_candidates" if clean_valid_ratio > 0.50 else "inspect",
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# DATA260708 Unlabeled Ball Reconstruction",
        "",
        "| CSV | Dynamic IDs | Clean Valid | Speed p95 | Segments | Rule Links | Other Links | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["files"]:
        lines.append(
            f"| `{Path(item['csv']).name}` | {item['dynamic_markers_used']} | "
            f"{item['clean_valid_ratio']:.3f} | {item['speed_p95_mps']:.2f} | "
            f"{len(item['segments'])} | {item['id_rule_link_count']} | {item['non_rule_link_count']} | {item['decision']} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append("- `source_unlabeled_id` in each NPZ records the chosen Motive marker ID per frame.")
    lines.append("- ID switches are allowed when the new marker is the same ID or an N+1/N+2 continuation with plausible motion.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/DATA260708.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/mocap_cleaning_outputs/DATA260708/ball_reconstruction"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])
    csv_dir = dataset_root / config["source_layout"].get("csv_dir", "CSV")
    paths = sorted(csv_dir.glob("*.csv"))
    if args.limit > 0:
        paths = paths[: args.limit]

    files = [analyze_csv(path, config, args.output_dir) for path in paths]
    report = {
        "config": str(args.config),
        "csv_dir": str(csv_dir),
        "files": files,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "unlabeled_ball_reconstruction_report.json"
    md_path = args.output_dir / "unlabeled_ball_reconstruction_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
