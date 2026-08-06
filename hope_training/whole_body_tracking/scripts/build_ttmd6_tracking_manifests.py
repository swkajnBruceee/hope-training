#!/usr/bin/env python3
"""Build replay-oriented TTMD6 motion manifests from optimized A3 outputs.

This does not claim training eligibility. It only wraps replay-ready TTMD6
optimized outputs into the same manifest family consumed by fixed-base strike
replay and later contract audits.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


CLASS_TO_STROKE = {
    1: "forehand",
    2: "forehand",
    3: "forehand",
    4: "backhand",
    5: "backhand",
    6: "backhand",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        (base_dir / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _infer_motion_npz_path(sample: dict[str, Any], optimized_manifest_path: Path) -> tuple[str, Path]:
    optimized_csv = str(sample["optimized_csv"])
    stem = Path(optimized_csv).stem
    root = optimized_manifest_path.parent
    candidates = [
        root / "optimized_motion_npz" / f"{stem}.npz",
        _resolve_path(str(Path(optimized_csv).with_suffix(".npz")).replace("optimized_csv", "optimized_motion_npz"), root),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate), candidate
    raise FileNotFoundError(f"unable to infer motion_npz for {sample['episode_id']}")


def _strike_target(target_npz: Any, hit_index: int) -> dict[str, Any]:
    position = target_npz["racket_pos"][hit_index].astype(float).tolist()
    velocity = target_npz["racket_vel"][hit_index].astype(float)
    speed = float(np.linalg.norm(velocity))
    direction = (velocity / speed).tolist() if speed > 1e-8 else [0.0, 0.0, 0.0]
    if "racket_normal_w" in target_npz.files:
        normal = target_npz["racket_normal_w"][hit_index].astype(float).tolist()
    else:
        normal = target_npz["racket_normal_base"][hit_index].astype(float).tolist()
    if "racket_tangent_w" in target_npz.files:
        tangent = target_npz["racket_tangent_w"][hit_index].astype(float).tolist()
    else:
        tangent = target_npz["racket_tangent_base"][hit_index].astype(float).tolist()
    return {
        "racket_position_m": position,
        "racket_normal_w": normal,
        "racket_tangent_w": tangent,
        "racket_velocity_mps": velocity.tolist(),
        "racket_velocity_direction_w": direction,
    }


def _hit_event(target_npz: Any, motion_fps: float, source_fps_default: float) -> dict[str, Any]:
    source_hit_index = int(target_npz["hit_index"].item())
    source_fps = float(target_npz["source_fps"].item()) if "source_fps" in target_npz.files else source_fps_default
    motion_hit = source_hit_index * motion_fps / source_fps
    motion_hit_frame = int(math.floor(motion_hit))
    motion_hit_subframe_alpha = float(motion_hit - motion_hit_frame)
    return {
        "source_hit_index": source_hit_index,
        "source_fps": source_fps,
        "hit_time_from_start_s": float(source_hit_index / source_fps),
        "motion_fps": float(motion_fps),
        "motion_hit_frame": motion_hit_frame,
        "motion_hit_subframe_alpha": motion_hit_subframe_alpha,
    }


def _build_motion_entry(
    sample: dict[str, Any],
    optimized_manifest_path: Path,
    source_fps_default: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    motion_npz_raw, motion_npz_path = _infer_motion_npz_path(sample, optimized_manifest_path)
    motion_npz = np.load(motion_npz_path)
    target_npz_raw = str(sample["target_npz"])
    target_npz_path = _resolve_path(target_npz_raw, optimized_manifest_path.parent)
    target_npz = np.load(target_npz_path)

    motion_fps = float(motion_npz["fps"].item())
    hit_event = _hit_event(target_npz, motion_fps=motion_fps, source_fps_default=source_fps_default)
    strike_target = _strike_target(target_npz, hit_index=hit_event["source_hit_index"])
    class_id = int(str(sample["source_id"]).split("_sample", 1)[0].removeprefix("class"))
    stroke_type = CLASS_TO_STROKE.get(class_id, "unknown")
    target_spec_json = str(sample["target_spec_json"])
    motion_entry = {
        "episode_id": str(sample["episode_id"]),
        "source_id": str(sample["source_id"]),
        "stroke_type": stroke_type,
        "stroke_confidence": 0.5,
        "stroke_label_status": "ttmd6_numeric_class_inferred",
        "selection_note": "ttmd6_replay_ready_auto_manifest",
        "motion_npz": motion_npz_raw,
        "fps": int(round(motion_fps)),
        "joint_pos_shape": list(motion_npz["joint_pos"].shape),
        "body_pos_w_shape": list(motion_npz["body_pos_w"].shape),
        "optimized_csv": str(sample["optimized_csv"]),
        "target_npz": target_npz_raw,
        "target_spec_json": target_spec_json,
        "hit_event": hit_event,
        "strike_target": strike_target,
    }
    npz_entry = {
        "episode_id": str(sample["episode_id"]),
        "motion_npz": motion_npz_raw,
        "fps": int(round(motion_fps)),
        "joint_pos_shape": list(motion_npz["joint_pos"].shape),
        "joint_vel_shape": list(motion_npz["joint_vel"].shape),
        "body_pos_w_shape": list(motion_npz["body_pos_w"].shape),
        "body_quat_w_shape": list(motion_npz["body_quat_w"].shape),
        "target_spec_json": target_spec_json,
        "hit_event": hit_event,
        "strike_target": strike_target,
    }
    return motion_entry, npz_entry


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _subset_manifest(payload: dict[str, Any], stroke: str) -> dict[str, Any]:
    motions = [motion for motion in payload["motions"] if motion["stroke_type"] == stroke]
    stroke_counts = Counter(motion["stroke_type"] for motion in motions)
    subset = dict(payload)
    subset["motions"] = motions
    subset["replay_ready_count"] = len(motions)
    subset["stroke_counts"] = dict(stroke_counts)
    subset["stroke"] = stroke
    if motions:
        subset["smoke_picks"] = {stroke: motions[0]["episode_id"]}
    else:
        subset["smoke_picks"] = {}
    return subset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("optimized_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument("--source-motion-npz-manifest-name", default="optimized_motion_npz_manifest.json")
    args = parser.parse_args()

    optimized_manifest_path = args.optimized_manifest.expanduser().resolve()
    optimized = _load_json(optimized_manifest_path)
    samples = optimized.get("samples", [])
    replay_ready = [sample for sample in samples if bool(sample.get("replay_ready")) and str(sample.get("optimized_status")) == "pass"]
    if not replay_ready:
        raise ValueError(f"no replay-ready pass samples in {optimized_manifest_path}")

    motions: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for sample in replay_ready:
        motion_entry, npz_entry = _build_motion_entry(
            sample=sample,
            optimized_manifest_path=optimized_manifest_path,
            source_fps_default=float(args.source_fps),
        )
        motions.append(motion_entry)
        entries.append(npz_entry)

    stroke_counts = Counter(motion["stroke_type"] for motion in motions)
    smoke_picks = {}
    for stroke in ("backhand", "forehand"):
        for motion in motions:
            if motion["stroke_type"] == stroke:
                smoke_picks[stroke] = motion["episode_id"]
                break

    source_motion_npz_manifest = {
        "stage": "ttmd6_replay_ready_motion_npz_v1",
        "count": len(entries),
        "all_joint_pos_shape": sorted({tuple(entry["joint_pos_shape"]) for entry in entries}),
        "all_body_pos_w_shape": sorted({tuple(entry["body_pos_w_shape"]) for entry in entries}),
        "all_fps": sorted({int(entry["fps"]) for entry in entries}),
        "entries": entries,
    }
    motion_npz_manifest_path = args.output_dir / args.source_motion_npz_manifest_name
    _write_json(motion_npz_manifest_path, source_motion_npz_manifest)

    tracking_manifest = {
        "source_manifest": str(optimized_manifest_path),
        "source_motion_npz_manifest": str(motion_npz_manifest_path),
        "replay_ready_count": len(motions),
        "stroke_counts": dict(stroke_counts),
        "smoke_picks": smoke_picks,
        "dataset_status": "ttmd6_replay_ready_not_training_approved",
        "motions": motions,
    }
    tracking_manifest_path = args.output_dir / "tracking_motion_manifest.json"
    _write_json(tracking_manifest_path, tracking_manifest)
    _write_json(args.output_dir / "tracking_motion_manifest_forehand.json", _subset_manifest(tracking_manifest, "forehand"))
    _write_json(args.output_dir / "tracking_motion_manifest_backhand.json", _subset_manifest(tracking_manifest, "backhand"))

    summary_lines = [
        "# TTMD6 Tracking Motion Manifest",
        "",
        f"- source optimized manifest: `{optimized_manifest_path}`",
        f"- replay-ready motions: `{len(motions)}`",
        "",
        "## Stroke Counts",
        "",
    ]
    for stroke in ("forehand", "backhand", "unknown"):
        if stroke in stroke_counts:
            summary_lines.append(f"- `{stroke}`: {stroke_counts[stroke]}")
    summary_lines += [
        "",
        "## Outputs",
        "",
        f"- motion npz manifest: `{motion_npz_manifest_path}`",
        f"- manifest: `{tracking_manifest_path}`",
        f"- forehand manifest: `{args.output_dir / 'tracking_motion_manifest_forehand.json'}`",
        f"- backhand manifest: `{args.output_dir / 'tracking_motion_manifest_backhand.json'}`",
        "",
        "## Notes",
        "",
        "- TTMD6 stroke labels remain class-inferred, not authoritative ground truth.",
        "- These manifests are replay-oriented only and remain not training-approved.",
        "- Native zero-residual calibration must be audited separately before RL use.",
    ]
    summary_path = args.output_dir / "tracking_motion_manifest.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "replay_ready_count": len(motions),
                "stroke_counts": dict(stroke_counts),
                "tracking_manifest": str(tracking_manifest_path),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
