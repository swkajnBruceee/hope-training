"""Cheap competition candidate profiling and coverage selection helpers."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def json_obj(raw: Any) -> dict[str, Any]:
    return json.loads(str(raw))


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def candidate_feature(data: np.lib.npyio.NpzFile, idx: int, mode: str = "full") -> np.ndarray:
    hit_index = int(data["hit_index"][idx])
    quat_hit = data["racket_pose_at_hit"][idx, 3:7].astype(np.float64)
    normal_hit = quat_xyzw_to_matrix(quat_hit)[:, 1]
    simple = np.concatenate(
        [
            data["racket_pos"][idx, hit_index].astype(np.float64),
            data["racket_vel_at_hit"][idx].astype(np.float64),
        ]
    )
    if mode == "simple":
        return simple
    return np.concatenate(
        [
            simple,
            normal_hit.astype(np.float64),
            data["ball_in_vel"][idx].astype(np.float64),
        ]
    )


def candidate_row(data: np.lib.npyio.NpzFile, idx: int, selection: dict[str, Any]) -> dict[str, Any]:
    source = json_obj(data["source_json"][idx])
    quality = json_obj(data["quality_flags_json"][idx])
    hit_index = int(data["hit_index"][idx])
    hit_ball = data["ball_pos"][idx, hit_index].astype(np.float64)
    hit_racket = data["racket_pos"][idx, hit_index].astype(np.float64)
    hit_dist = float(np.linalg.norm(hit_ball - hit_racket))
    label = str(data["stroke_type_rule_v2"][idx])
    conf = float(data["stroke_confidence_rule_v2"][idx])
    success = int(data["success"][idx])
    racket = str(source.get("racket", ""))
    preferred = [str(x) for x in selection.get("preferred_strokes", [])]
    allow_unknown = bool(selection.get("allow_unknown", False))
    min_conf = float(selection.get("min_stroke_confidence", 0.0))
    success_only = bool(selection.get("success_only", True))
    target_racket = str(selection.get("racket", racket))

    reasons: list[str] = []
    if racket != target_racket:
        reasons.append("racket_mismatch")
    if success_only and success != 1:
        reasons.append("unsuccessful")
    if not np.isfinite(hit_dist) or hit_dist > float(selection.get("max_hit_distance_m", 0.15)):
        reasons.append("hit_distance")
    if label == "unknown" and not allow_unknown:
        reasons.append("unknown_stroke")
    if preferred and label not in preferred and label != "unknown":
        reasons.append("non_preferred_stroke")
    if label != "unknown" and conf < min_conf:
        reasons.append("low_stroke_confidence")

    ball_in = data["ball_in_vel"][idx].astype(np.float64)
    ball_out = data["ball_out_vel"][idx].astype(np.float64)
    racket_vel = data["racket_vel_at_hit"][idx].astype(np.float64)
    normal = quat_xyzw_to_matrix(data["racket_pose_at_hit"][idx, 3:7].astype(np.float64))[:, 1]
    source_quality = len([v for v in quality.values() if v is False]) == 0
    score = 0.0
    score += 2.0 if label in preferred else 0.0
    score += min(conf, 1.0)
    score += max(0.0, 0.15 - hit_dist)
    score += 0.2 if success == 1 else 0.0
    score += 0.1 if source_quality else 0.0
    return {
        "dataset_index": int(idx),
        "source_index": int(data["source_index"][idx]) if "source_index" in data.files else int(idx),
        "episode_id": str(data["episode_id"][idx]),
        "racket": racket,
        "stroke_type": label,
        "stroke_confidence": conf,
        "success": success,
        "cheap_quality_pass": not reasons,
        "cheap_reject_reasons": reasons,
        "cheap_quality_score": float(score),
        "source_quality_ok": bool(source_quality),
        "hit_index": hit_index,
        "hit_time_s": float(data["hit_time"][idx]),
        "hit_pos_x": float(hit_ball[0]),
        "hit_pos_y": float(hit_ball[1]),
        "hit_pos_z": float(hit_ball[2]),
        "racket_pos_x": float(hit_racket[0]),
        "racket_pos_y": float(hit_racket[1]),
        "racket_pos_z": float(hit_racket[2]),
        "ball_racket_min_distance_m": hit_dist,
        "racket_vel_x": float(racket_vel[0]),
        "racket_vel_y": float(racket_vel[1]),
        "racket_vel_z": float(racket_vel[2]),
        "racket_speed_at_hit_mps": float(np.linalg.norm(racket_vel)),
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "normal_z": float(normal[2]),
        "ball_vin_x": float(ball_in[0]),
        "ball_vin_y": float(ball_in[1]),
        "ball_vin_z": float(ball_in[2]),
        "ball_speed_in_mps": float(np.linalg.norm(ball_in)),
        "ball_vout_x": float(ball_out[0]),
        "ball_vout_y": float(ball_out[1]),
        "ball_vout_z": float(ball_out[2]),
        "ball_speed_out_mps": float(np.linalg.norm(ball_out)),
        "motion_duration_s": float(data["time"][idx, -1] - data["time"][idx, 0]),
    }


def rows_from_dataset(data: np.lib.npyio.NpzFile, selection: dict[str, Any]) -> list[dict[str, Any]]:
    return [candidate_row(data, idx, selection) for idx in range(int(data["ball_pos"].shape[0]))]


def finite_feature_matrix(rows: list[dict[str, Any]], mode: str) -> np.ndarray:
    keys = ["hit_pos_x", "hit_pos_y", "hit_pos_z", "racket_vel_x", "racket_vel_y", "racket_vel_z"]
    if mode == "full":
        keys += ["normal_x", "normal_y", "normal_z", "ball_vin_x", "ball_vin_y", "ball_vin_z"]
    features = np.asarray([[float(row.get(key, 0.0)) for key in keys] for row in rows], dtype=np.float64)
    if not np.isfinite(features).all():
        col_median = np.nanmedian(np.where(np.isfinite(features), features, np.nan), axis=0)
        col_median = np.where(np.isfinite(col_median), col_median, 0.0)
        bad_rows, bad_cols = np.where(~np.isfinite(features))
        features[bad_rows, bad_cols] = col_median[bad_cols]
    scale = np.nanstd(features, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (features - np.nanmean(features, axis=0)) / scale


def farthest_point_select(rows: list[dict[str, Any]], count: int, feature_mode: str, quality_weight: float = 0.15) -> list[dict[str, Any]]:
    if not rows or count <= 0:
        return []
    features = finite_feature_matrix(rows, feature_mode)
    scores = np.asarray([float(row.get("cheap_quality_score", 0.0)) for row in rows], dtype=np.float64)
    score_span = max(float(np.max(scores) - np.min(scores)), 1e-9)
    score_norm = (scores - float(np.min(scores))) / score_span
    remaining = set(range(len(rows)))
    first = max(remaining, key=lambda i: (scores[i], -int(rows[i]["dataset_index"])))
    selected = [first]
    remaining.remove(first)
    while remaining and len(selected) < count:
        selected_features = features[np.asarray(selected, dtype=np.int64)]
        best = max(
            remaining,
            key=lambda i: (
                float(np.min(np.linalg.norm(selected_features - features[i][None, :], axis=1))) + quality_weight * float(score_norm[i]),
                scores[i],
                -int(rows[i]["dataset_index"]),
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return [rows[i] for i in selected]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def load_index(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_selection_markdown(report: dict[str, Any], path: Path) -> None:
    rows = report["selected"]
    lines = [
        f"# {report['stage'].replace('_', ' ').title()}",
        "",
        f"- source index: `{report['source_index']}`",
        f"- selected: `{len(rows)}`",
        f"- feature mode: `{report.get('feature_mode', '')}`",
        "",
        "## Stroke Counts",
        "",
    ]
    for stroke, count in sorted(Counter(row["stroke_type"] for row in rows).items()):
        lines.append(f"- `{stroke}`: {count}")
    lines.extend(["", "## Selected", ""])
    for rank, row in enumerate(rows, start=1):
        lines.append(f"- `{rank:03d}` `{row['episode_id']}` `{row['stroke_type']}` score={float(row.get('cheap_quality_score', 0.0)):.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_indices_from_payload(payload: dict[str, Any]) -> list[int]:
    rows: Iterable[dict[str, Any]] = payload.get("selected", payload.get("samples", []))
    return [int(row["dataset_index"]) for row in rows]
