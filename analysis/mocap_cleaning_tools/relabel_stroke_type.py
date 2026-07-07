#!/usr/bin/env python3
"""Relabel forehand/backhand from body-local swing features.

This script intentionally does not map a player/racket id to a stroke type.
Each episode is classified from the racket motion around its own hit frame.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _window_slice(center: int, radius_before: int, radius_after: int, length: int) -> slice:
    start = max(0, center - radius_before)
    end = min(length, center + radius_after + 1)
    return slice(start, end)


def _safe_counter(values: np.ndarray | list[str]) -> Counter:
    return Counter([str(v) for v in values])


def _parse_sources(source_json: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources: list[str] = []
    rackets: list[str] = []
    skeletons: list[str] = []
    for raw in source_json:
        try:
            item = json.loads(str(raw))
        except json.JSONDecodeError:
            item = {}
        source = str(item.get("source_csv", "unknown"))
        racket = str(item.get("racket", "unknown"))
        joined = " ".join(str(item.get(k, "")) for k in ("source_bvh", "source_npz", "sample_path"))
        match = re.search(r"Skeleton(\d+)", joined)
        skeleton = f"Skeleton{match.group(1)}" if match else "unknown"
        sources.append(source)
        rackets.append(racket)
        skeletons.append(skeleton)
    return np.asarray(sources), np.asarray(rackets), np.asarray(skeletons)


def compute_stroke_features(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    time = data["time"]
    racket_pos = data["racket_pos"]
    racket_vel = data["racket_vel"]
    body_center = data["body_center"]
    body_right_axis = data["body_right_axis"]
    hit_index = data["hit_index"].astype(int)

    n, t, _ = racket_pos.shape
    right = np.zeros((n, 3), dtype=np.float64)
    lateral_offset = np.full(n, np.nan, dtype=np.float64)
    lateral_velocity = np.full(n, np.nan, dtype=np.float64)
    lateral_velocity_median = np.full(n, np.nan, dtype=np.float64)
    pre_to_hit_lateral_delta = np.full(n, np.nan, dtype=np.float64)
    pre_lateral_offset = np.full(n, np.nan, dtype=np.float64)
    hit_lateral_offset_window = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        h = int(hit_index[i])
        axis = body_right_axis[i, h]
        norm = np.linalg.norm(axis)
        if not np.isfinite(norm) or norm < 1e-8:
            continue
        axis = axis / norm
        right[i] = axis

        rel_hit = racket_pos[i, h] - body_center[i, h]
        lateral_offset[i] = float(np.dot(rel_hit, axis))
        lateral_velocity[i] = float(np.dot(racket_vel[i, h], axis))

        hit_vel_slice = _window_slice(h, 4, 4, t)
        lateral_velocity_median[i] = float(np.nanmedian(racket_vel[i, hit_vel_slice] @ axis))

        # Use a pre-hit window well before contact to avoid collision-frame noise.
        pre_start = max(0, h - 24)  # 120 ms before hit at 200 Hz
        pre_end = max(pre_start + 1, h - 10)  # stop 50 ms before hit
        hit_pos_slice = _window_slice(h, 2, 2, t)

        pre_rel = racket_pos[i, pre_start:pre_end] - body_center[i, pre_start:pre_end]
        hit_rel = racket_pos[i, hit_pos_slice] - body_center[i, hit_pos_slice]
        pre_lateral_offset[i] = float(np.nanmedian(pre_rel @ axis))
        hit_lateral_offset_window[i] = float(np.nanmedian(hit_rel @ axis))
        pre_to_hit_lateral_delta[i] = hit_lateral_offset_window[i] - pre_lateral_offset[i]

    return {
        "stroke_lateral_offset_m": lateral_offset,
        "stroke_lateral_velocity_hit_mps": lateral_velocity,
        "stroke_lateral_velocity_window_mps": lateral_velocity_median,
        "stroke_pre_lateral_offset_m": pre_lateral_offset,
        "stroke_hit_lateral_offset_window_m": hit_lateral_offset_window,
        "stroke_pre_to_hit_lateral_delta_m": pre_to_hit_lateral_delta,
        "stroke_body_right_axis_at_hit": right,
    }


def score_strokes(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    vel = features["stroke_lateral_velocity_window_mps"]
    delta = features["stroke_pre_to_hit_lateral_delta_m"]
    offset = features["stroke_lateral_offset_m"]
    n = len(vel)

    forehand_score = np.zeros(n, dtype=np.float64)
    backhand_score = np.zeros(n, dtype=np.float64)
    labels = np.full(n, "unknown", dtype="<U8")
    confidence = np.zeros(n, dtype=np.float64)
    reason = np.full(n, "unclassified", dtype="<U96")

    for i in range(n):
        if not (np.isfinite(vel[i]) and np.isfinite(delta[i]) and np.isfinite(offset[i])):
            reason[i] = "nonfinite_stroke_features"
            continue

        f = 0.0
        b = 0.0

        # Right-handed body-local convention:
        # forehand generally moves from outside/right toward inside/left at contact;
        # backhand generally moves from inside/left toward outside/right.
        if vel[i] < -0.65:
            f += 3.0
        elif vel[i] < -0.30:
            f += 2.0
        elif vel[i] < -0.15:
            f += 1.0
        elif vel[i] > 0.65:
            b += 3.0
        elif vel[i] > 0.30:
            b += 2.0
        elif vel[i] > 0.15:
            b += 1.0

        if delta[i] < -0.060:
            f += 3.0
        elif delta[i] < -0.025:
            f += 2.0
        elif delta[i] < -0.012:
            f += 1.0
        elif delta[i] > 0.060:
            b += 3.0
        elif delta[i] > 0.025:
            b += 2.0
        elif delta[i] > 0.012:
            b += 1.0

        # Contact position is a weak prior only. It must not dominate swing direction.
        if offset[i] > 0.32:
            f += 0.75
        elif offset[i] < 0.16:
            b += 0.75

        forehand_score[i] = f
        backhand_score[i] = b
        margin = abs(f - b)
        max_score = max(f, b)

        if max_score < 2.0:
            reason[i] = "weak_motion_evidence"
        elif f >= b + 2.0:
            labels[i] = "forehand"
            reason[i] = "body_local_swing_forehand"
            confidence[i] = min(1.0, 0.35 + margin / 5.0)
        elif b >= f + 2.0:
            labels[i] = "backhand"
            reason[i] = "body_local_swing_backhand"
            confidence[i] = min(1.0, 0.35 + margin / 5.0)
        else:
            reason[i] = "ambiguous_body_local_swing"
            confidence[i] = min(0.45, margin / 5.0)

    return {
        "stroke_type_rule_v2": labels,
        "stroke_confidence_rule_v2": confidence,
        "stroke_score_forehand_rule_v2": forehand_score,
        "stroke_score_backhand_rule_v2": backhand_score,
        "stroke_label_reason_rule_v2": reason,
    }


def _markdown_counter_table(title: str, counter: Counter) -> list[str]:
    lines = [f"### {title}", "", "| value | count |", "|---|---:|"]
    for key, value in counter.most_common():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return lines


def _markdown_cross_table(title: str, group: np.ndarray, labels: np.ndarray) -> list[str]:
    lines = [f"### {title}", "", "| group | forehand | backhand | unknown | total |", "|---|---:|---:|---:|---:|"]
    for key in sorted(set(str(v) for v in group)):
        mask = group == key
        c = _safe_counter(labels[mask])
        total = int(mask.sum())
        lines.append(f"| {key} | {c['forehand']} | {c['backhand']} | {c['unknown']} | {total} |")
    lines.append("")
    return lines


def write_report(
    out_path: Path,
    data: np.lib.npyio.NpzFile,
    features: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> None:
    labels = scores["stroke_type_rule_v2"]
    confidence = scores["stroke_confidence_rule_v2"]
    sources, rackets, skeletons = _parse_sources(data["source_json"])
    old_labels = data["stroke_type"] if "stroke_type" in data.files else np.full(len(labels), "missing")

    lines: list[str] = [
        "# DATA260703 Stroke Relabel Report",
        "",
        "This report is generated from body-local swing features. Player/racket id is not used as a classifier input.",
        "",
        f"- samples: {len(labels)}",
        f"- input old labels: `{dict(_safe_counter(old_labels))}`",
        f"- output labels: `{dict(_safe_counter(labels))}`",
        "",
    ]

    lines += _markdown_counter_table("Output Label Counts", _safe_counter(labels))
    lines += _markdown_cross_table("Output By Racket", rackets, labels)
    lines += _markdown_cross_table("Output By Skeleton", skeletons, labels)
    lines += _markdown_cross_table("Output By Source CSV", sources, labels)

    lines += [
        "## Feature Summary",
        "",
        "| label | n | median lateral offset m | median lateral velocity m/s | median pre-to-hit delta m | median confidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("forehand", "backhand", "unknown"):
        mask = labels == label
        if not mask.any():
            continue
        lines.append(
            "| "
            + label
            + f" | {int(mask.sum())}"
            + f" | {np.nanmedian(features['stroke_lateral_offset_m'][mask]):.4f}"
            + f" | {np.nanmedian(features['stroke_lateral_velocity_window_mps'][mask]):.4f}"
            + f" | {np.nanmedian(features['stroke_pre_to_hit_lateral_delta_m'][mask]):.4f}"
            + f" | {np.nanmedian(confidence[mask]):.3f} |"
        )
    lines.append("")

    boundary = np.where((labels == "unknown") | (confidence < 0.70))[0]
    lines += [
        "## Boundary Samples",
        "",
        f"Boundary sample count (`unknown` or confidence < 0.70): {len(boundary)}",
        "",
        "| idx | episode_id | label | conf | racket | source | lat_off | lat_vel | delta | reason |",
        "|---:|---|---|---:|---|---|---:|---:|---:|---|",
    ]
    for i in boundary[:200]:
        lines.append(
            f"| {int(i)} | {data['episode_id'][i]} | {labels[i]} | {confidence[i]:.3f}"
            f" | {rackets[i]} | {sources[i]}"
            f" | {features['stroke_lateral_offset_m'][i]:.4f}"
            f" | {features['stroke_lateral_velocity_window_mps'][i]:.4f}"
            f" | {features['stroke_pre_to_hit_lateral_delta_m'][i]:.4f}"
            f" | {scores['stroke_label_reason_rule_v2'][i]} |"
        )
    lines.append("")

    lines += [
        "## Rule Notes",
        "",
        "- Positive/negative labels are inferred from motion relative to `body_right_axis`, not from Motive/table axes.",
        "- Racket id and skeleton id are used only in this report for auditing.",
        "- Low-confidence samples are intentionally kept as `unknown` to avoid contaminating training labels.",
        "- This relabeling can be run before table-frame calibration; table-frame calibration is still required for landing and success labels.",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    features = compute_stroke_features(data)
    scores = score_strokes(features)

    arrays = {key: data[key] for key in data.files}
    arrays.update(features)
    arrays.update(scores)
    arrays["stroke_type_original"] = data["stroke_type"].copy() if "stroke_type" in data.files else np.full(len(scores["stroke_type_rule_v2"]), "")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    write_report(args.report, data, features, scores)

    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    print(dict(_safe_counter(scores["stroke_type_rule_v2"])))


if __name__ == "__main__":
    main()
