#!/usr/bin/env python3
"""Prepare P2 competition-frame samples for fixed-base A3 retargeting."""

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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analysis.mocap_cleaning.competition_candidate_utils import selected_indices_from_payload
from analysis.mocap_cleaning.config import load_config


def _json_obj(raw: Any) -> dict[str, Any]:
    return json.loads(str(raw))


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
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


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.full_like(v, np.nan, dtype=np.float64)
    return v / norm


def _subset_sample(data: np.lib.npyio.NpzFile, idx: int) -> dict[str, np.ndarray]:
    n = int(data["ball_pos"].shape[0])
    out: dict[str, np.ndarray] = {}
    for key in data.files:
        value = data[key]
        if value.shape and value.shape[0] == n:
            out[key] = value[idx]
        else:
            out[key] = value
    return out


def _safe_episode(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def _candidate_feature(data: np.lib.npyio.NpzFile, idx: int) -> np.ndarray:
    hit_index = int(data["hit_index"][idx])
    quat_hit = data["racket_pose_at_hit"][idx, 3:7].astype(np.float64)
    normal_hit = _quat_xyzw_to_matrix(quat_hit)[:, 1]
    return np.concatenate(
        [
            data["racket_pos"][idx, hit_index].astype(np.float64),
            data["ball_in_vel"][idx].astype(np.float64),
            data["racket_vel_at_hit"][idx].astype(np.float64),
            normal_hit.astype(np.float64),
        ]
    )


def _diverse_order(candidates: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    if not candidates or target_count <= 0:
        return []
    features = np.asarray([item["feature"] for item in candidates], dtype=np.float64)
    finite = np.isfinite(features).all(axis=1)
    if not bool(np.all(finite)):
        features = features.copy()
        col_median = np.nanmedian(np.where(np.isfinite(features), features, np.nan), axis=0)
        col_median = np.where(np.isfinite(col_median), col_median, 0.0)
        bad_rows, bad_cols = np.where(~np.isfinite(features))
        features[bad_rows, bad_cols] = col_median[bad_cols]
    scale = np.nanstd(features, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    features = (features - np.nanmean(features, axis=0)) / scale

    scores = np.asarray([float(item["score"]) for item in candidates], dtype=np.float64)
    score_span = max(float(np.max(scores) - np.min(scores)), 1e-9)
    score_norm = (scores - float(np.min(scores))) / score_span

    remaining = set(range(len(candidates)))
    first = max(remaining, key=lambda i: (scores[i], -int(candidates[i]["idx"])))
    selected = [first]
    remaining.remove(first)
    while remaining and len(selected) < target_count:
        selected_features = features[np.asarray(selected, dtype=np.int64)]
        best = max(
            remaining,
            key=lambda i: (
                float(np.min(np.linalg.norm(selected_features - features[i][None, :], axis=1))) + 0.15 * float(score_norm[i]),
                scores[i],
                -int(candidates[i]["idx"]),
            ),
        )
        selected.append(best)
        remaining.remove(best)
    if len(selected) < target_count:
        selected.extend(sorted(remaining, key=lambda i: (-scores[i], int(candidates[i]["idx"])))[: target_count - len(selected)])
    return [candidates[i] for i in selected]


def _pick_indices(
    data: np.lib.npyio.NpzFile,
    config: dict[str, Any],
    limit: int | None,
    offset: int,
) -> list[int]:
    selection = config["selection"]
    sources = [_json_obj(x) for x in data["source_json"]]
    racket = str(selection["racket"])
    success_only = bool(selection.get("success_only", True))
    min_conf = float(selection.get("min_stroke_confidence", 0.0))
    preferred = [str(x) for x in selection.get("preferred_strokes", [])]
    allow_unknown = bool(selection.get("allow_unknown", False))
    per_stroke = int(selection.get("per_stroke_target", 0) or 0)
    first_batch = int(limit or selection.get("first_batch_size", 20))
    offset = max(int(offset), 0)
    target_count = offset + first_batch

    hit_index = data["hit_index"].astype(int)
    hit_ball = data["ball_pos"][np.arange(len(hit_index)), hit_index]
    hit_racket = data["racket_pos"][np.arange(len(hit_index)), hit_index]
    hit_dist = np.linalg.norm(hit_ball - hit_racket, axis=1)
    labels = np.asarray(data["stroke_type_rule_v2"]).astype(str)
    confidence = np.asarray(data["stroke_confidence_rule_v2"], dtype=np.float64)

    candidates = []
    for idx, src in enumerate(sources):
        if str(src.get("racket")) != racket:
            continue
        if success_only and int(data["success"][idx]) != 1:
            continue
        if not np.isfinite(hit_dist[idx]) or hit_dist[idx] > 0.15:
            continue
        label = str(labels[idx])
        conf = float(confidence[idx])
        if label == "unknown" and not allow_unknown:
            continue
        if preferred and label not in preferred and label != "unknown":
            continue
        if label != "unknown" and conf < min_conf:
            continue
        score = 0.0
        score += 2.0 if label in preferred else 0.0
        score += min(conf, 1.0)
        score += max(0.0, 0.15 - float(hit_dist[idx]))
        score += 0.2 if int(data["success"][idx]) == 1 else 0.0
        candidates.append({"idx": idx, "label": label, "score": score, "feature": _candidate_feature(data, idx)})

    selected: list[int] = []
    if bool(selection.get("diversity_sampling", False)):
        diversity_order = _diverse_order(candidates, target_count)
        selected = [int(item["idx"]) for item in diversity_order[:target_count]]
    else:
        used = set()
        if per_stroke > 0:
            for label in preferred:
                group = sorted([x for x in candidates if x["label"] == label], key=lambda x: (-float(x["score"]), int(x["idx"])))
                for item in group[:per_stroke]:
                    selected.append(int(item["idx"]))
                    used.add(int(item["idx"]))
        remaining = sorted([x for x in candidates if int(x["idx"]) not in used], key=lambda x: (-float(x["score"]), int(x["idx"])))
        for item in remaining:
            if len(selected) >= target_count:
                break
            selected.append(int(item["idx"]))
    return selected[offset:target_count]


def _target_spec(
    *,
    data: np.lib.npyio.NpzFile,
    idx: int,
    config: dict[str, Any],
    target_npz: Path,
    spec_path: Path,
) -> dict[str, Any]:
    source = _json_obj(data["source_json"][idx])
    quality = _json_obj(data["quality_flags_json"][idx])
    attrs = _json_obj(data["dataset_attrs_json"])
    hit_index = int(data["hit_index"][idx])
    quat_hit = data["racket_pose_at_hit"][idx, 3:7].astype(np.float64)
    rot_hit = _quat_xyzw_to_matrix(quat_hit)
    vel_hit = data["racket_vel_at_hit"][idx].astype(np.float64)
    base_cfg = config["robot_base"]
    return {
        "spec_version": "competition_a3_fixed_v1",
        "episode_id": str(data["episode_id"][idx]),
        "source_index": int(data["source_index"][idx]) if "source_index" in data.files else int(idx),
        "dataset_index": int(idx),
        "robot": str(config["robot"]),
        "robot_side": str(config["robot_side"]),
        "base_mode": str(config["base_mode"]),
        "coordinate_contract": {
            "position_frame": str(attrs.get("coordinate_frame", config["coordinate_frame"])),
            "orientation_frame": str(attrs.get("coordinate_frame", config["coordinate_frame"])),
            "position_unit": "m",
            "angle_unit": "rad",
            "time_unit": "s",
            "quat_order": "xyzw",
            "fps": float(config["time"]["fps"]),
            "hit_index": hit_index,
            "sequence_length_frames": int(data["ball_pos"].shape[1]),
        },
        "robot_base": {
            "position_m": [float(x) for x in base_cfg["position_m"]],
            "quat_xyzw": [float(x) for x in base_cfg["quat_xyzw"]],
            "note": "fixed P2 base, facing P1",
        },
        "source": {
            "source_csv": source.get("source_csv", ""),
            "source_bvh": source.get("source_bvh", ""),
            "racket": source.get("racket", ""),
            "stroke_type_rule_v2": str(data["stroke_type_rule_v2"][idx]),
            "stroke_confidence_rule_v2": float(data["stroke_confidence_rule_v2"][idx]),
            "success": int(data["success"][idx]),
            "landing_pos": data["landing_pos"][idx].astype(float).tolist(),
        },
        "racket_reference_point": attrs.get("racket_reference_point", {}),
        "hit_target": {
            "racket_position_m": data["racket_pose_at_hit"][idx, :3].astype(float).tolist(),
            "racket_quat_xyzw": quat_hit.astype(float).tolist(),
            "racket_normal_w": rot_hit[:, 1].astype(float).tolist(),
            "racket_tangent_w": rot_hit[:, 0].astype(float).tolist(),
            "racket_velocity_mps": vel_hit.astype(float).tolist(),
            "racket_velocity_direction_w": _normalize(vel_hit).astype(float).tolist(),
            "ball_position_m": data["hit_pos"][idx].astype(float).tolist(),
            "ball_in_velocity_mps": data["ball_in_vel"][idx].astype(float).tolist(),
            "ball_out_velocity_mps": data["ball_out_vel"][idx].astype(float).tolist(),
            "ball_to_racket_center_distance_m": float(np.linalg.norm(data["hit_pos"][idx] - data["racket_pose_at_hit"][idx, :3])),
        },
        "phase_windows": config["phase_windows"],
        "active_joints": config["ik"]["active_joints"],
        "quality_thresholds": config["quality_thresholds"],
        "artifacts": {
            "target_npz": str(target_npz),
            "target_spec_json": str(spec_path),
            "ik_init_csv": "",
            "optimized_csv": "",
            "quality_report_json": "",
            "motion_npz": "",
        },
        "quality_flags": quality,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# DATA260708 P2 Fixed-Base A3 Retarget Targets",
        "",
        f"- dataset: `{report['dataset']}`",
        f"- selected: `{report['selected_count']}`",
        f"- robot: `{report['robot']}`",
        f"- robot side: `{report['robot_side']}`",
        f"- base mode: `{report['base_mode']}`",
        "",
        "## Stroke Counts",
        "",
        "| stroke | count |",
        "|---|---:|",
    ]
    for key, value in sorted(report["stroke_counts"].items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--selected-candidates", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = Path(str(config["dataset"]))
    output_root = Path(str(config["output_root"]))
    ready_dir = output_root / "retarget_ready"
    target_dir = output_root / "target_npz"
    spec_dir = output_root / "target_specs"
    ready_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path, allow_pickle=True)
    offset = int(args.offset if args.offset is not None else config.get("selection_offset", 0))
    selected_candidates = args.selected_candidates or (Path(str(config["selection"]["selected_candidates"])) if config["selection"].get("selected_candidates") else None)
    if selected_candidates is not None:
        payload = json.loads(selected_candidates.read_text(encoding="utf-8"))
        selected = selected_indices_from_payload(payload)
        if args.limit is not None:
            selected = selected[: max(0, args.limit)]
        offset = 0
    else:
        selected = _pick_indices(data, config, args.limit, offset)
    samples = []
    for idx in selected:
        episode_id = _safe_episode(str(data["episode_id"][idx]))
        target_npz = target_dir / f"{episode_id}_target.npz"
        spec_path = spec_dir / f"{episode_id}_target_spec.json"
        arrays = _subset_sample(data, idx)
        np.savez_compressed(target_npz, **arrays)
        spec = _target_spec(data=data, idx=idx, config=config, target_npz=target_npz, spec_path=spec_path)
        spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        samples.append(
            {
                "episode_id": str(data["episode_id"][idx]),
                "dataset_index": int(idx),
                "source_index": int(data["source_index"][idx]) if "source_index" in data.files else int(idx),
                "stroke_type_rule_v2": str(data["stroke_type_rule_v2"][idx]),
                "stroke_confidence_rule_v2": float(data["stroke_confidence_rule_v2"][idx]),
                "success": int(data["success"][idx]),
                "target_npz": str(target_npz),
                "target_spec_json": str(spec_path),
            }
        )

    manifest = {
        "config": str(args.config),
        "dataset": str(dataset_path),
        "robot": str(config["robot"]),
        "robot_side": str(config["robot_side"]),
        "base_mode": str(config["base_mode"]),
        "coordinate_frame": str(config["coordinate_frame"]),
        "selected_count": len(samples),
        "selection_offset": offset,
        "selected_candidates": str(selected_candidates) if selected_candidates is not None else "",
        "samples": samples,
    }
    manifest_path = ready_dir / "retarget_target_manifest.json"
    report_path = ready_dir / "retarget_target_summary.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        **manifest,
        "stroke_counts": dict(Counter(s["stroke_type_rule_v2"] for s in samples)),
        "outputs": {
            "manifest": str(manifest_path),
            "target_npz_dir": str(target_dir),
            "target_spec_dir": str(spec_dir),
        },
    }
    _write_markdown(report, report_path)
    print(f"Selected {len(samples)} samples")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
