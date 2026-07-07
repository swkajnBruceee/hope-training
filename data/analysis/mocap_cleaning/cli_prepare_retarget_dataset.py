#!/usr/bin/env python3
"""Prepare retarget-ready sample manifests from a relabeled packed dataset."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    del _ROOT

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RetargetSample:
    episode_id: str
    label: str
    confidence: float
    source_csv: str
    source_bvh: str
    sample_path: str
    source_npz: str
    debug_npz: str
    racket: str
    candidate: str
    handedness: str
    hit_time: float
    hit_time_rel: float
    hit_index: int
    frames: int
    fps: float
    max_ball_speed_mps: float
    max_racket_speed_mps: float
    max_racket_omega_radps: float
    source_label: str
    source_label_confidence: float
    v2_reason: str
    quality_flags: dict[str, Any]
    source_json: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "label": self.label,
            "confidence": self.confidence,
            "source_csv": self.source_csv,
            "source_bvh": self.source_bvh,
            "sample_path": self.sample_path,
            "source_npz": self.source_npz,
            "debug_npz": self.debug_npz,
            "racket": self.racket,
            "candidate": self.candidate,
            "handedness": self.handedness,
            "hit_time": self.hit_time,
            "hit_time_rel": self.hit_time_rel,
            "hit_index": self.hit_index,
            "frames": self.frames,
            "fps": self.fps,
            "max_ball_speed_mps": self.max_ball_speed_mps,
            "max_racket_speed_mps": self.max_racket_speed_mps,
            "max_racket_omega_radps": self.max_racket_omega_radps,
            "source_label": self.source_label,
            "source_label_confidence": self.source_label_confidence,
            "v2_reason": self.v2_reason,
            "quality_flags": self.quality_flags,
            "source_json": self.source_json,
        }


def _json_obj(raw: Any) -> dict[str, Any]:
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def _load_samples(
    dataset_path: Path,
    label_field: str,
    confidence_field: str,
    min_confidence: float,
    keep_unknown: bool,
    limit_per_label: int | None,
) -> tuple[list[RetargetSample], dict[str, Any]]:
    data = np.load(dataset_path, allow_pickle=False)
    labels = np.asarray(data[label_field]).astype(str)
    confidence = np.asarray(data[confidence_field], dtype=np.float64)

    samples: list[RetargetSample] = []
    skipped = Counter()
    label_kept = Counter()
    label_seen = Counter()

    time = data["time"]
    for idx in range(len(labels)):
        label = str(labels[idx])
        conf = float(confidence[idx])
        label_seen[label] += 1

        if label == "unknown" and not keep_unknown:
            skipped["unknown"] += 1
            continue
        if label != "unknown" and conf < min_confidence:
            skipped["low_confidence"] += 1
            continue
        if limit_per_label is not None and label_kept[label] >= limit_per_label:
            skipped["limit_per_label"] += 1
            continue

        src = _json_obj(data["source_json"][idx])
        q = _json_obj(data["quality_flags_json"][idx])
        dt = float(np.nanmedian(np.diff(time[idx])))
        fps = float(1.0 / dt)
        hit_meta = src.get("hit_metadata", {})

        sample = RetargetSample(
            episode_id=str(data["episode_id"][idx]),
            label=label,
            confidence=conf,
            source_csv=str(src.get("source_csv", "")),
            source_bvh=str(src.get("source_bvh", "")),
            sample_path=str(data["sample_path"][idx]),
            source_npz=str(src.get("source_npz", "")),
            debug_npz=str(src.get("debug_npz", "")),
            racket=str(src.get("racket", "")),
            candidate=str(src.get("candidate", "")),
            handedness=str(src.get("handedness", "")),
            hit_time=float(data["hit_time"][idx]),
            hit_time_rel=float(hit_meta.get("hit_time_rel", float("nan"))),
            hit_index=int(data["hit_index"][idx]),
            frames=int(time[idx].shape[0]),
            fps=fps,
            max_ball_speed_mps=float(q.get("max_ball_speed_mps", float("nan"))),
            max_racket_speed_mps=float(q.get("max_racket_speed_mps", float("nan"))),
            max_racket_omega_radps=float(q.get("max_racket_omega_radps", float("nan"))),
            source_label=str(data["stroke_type"][idx]) if "stroke_type" in data.files else "",
            source_label_confidence=float(q.get("stroke_label_confidence", float("nan"))),
            v2_reason=str(data["stroke_label_reason_rule_v2"][idx]) if "stroke_label_reason_rule_v2" in data.files else "",
            quality_flags=q,
            source_json=src,
        )
        samples.append(sample)
        label_kept[label] += 1

    samples.sort(key=lambda item: (item.label, -item.confidence, item.source_csv, item.episode_id))
    stats = {
        "dataset_path": str(dataset_path),
        "label_field": label_field,
        "confidence_field": confidence_field,
        "total_samples": int(len(labels)),
        "label_seen": dict(label_seen),
        "label_kept": dict(label_kept),
        "skipped": dict(skipped),
    }
    return samples, stats


def _collect_review_samples(
    dataset_path: Path,
    label_field: str,
    confidence_field: str,
    min_confidence: float,
) -> dict[str, list[RetargetSample]]:
    data = np.load(dataset_path, allow_pickle=False)
    labels = np.asarray(data[label_field]).astype(str)
    confidence = np.asarray(data[confidence_field], dtype=np.float64)
    time = data["time"]

    groups: dict[str, list[RetargetSample]] = {
        "unknown": [],
        "low_confidence": [],
    }
    for idx in range(len(labels)):
        label = str(labels[idx])
        conf = float(confidence[idx])
        src = _json_obj(data["source_json"][idx])
        q = _json_obj(data["quality_flags_json"][idx])
        dt = float(np.nanmedian(np.diff(time[idx])))
        fps = float(1.0 / dt)
        hit_meta = src.get("hit_metadata", {})
        sample = RetargetSample(
            episode_id=str(data["episode_id"][idx]),
            label=label,
            confidence=conf,
            source_csv=str(src.get("source_csv", "")),
            source_bvh=str(src.get("source_bvh", "")),
            sample_path=str(data["sample_path"][idx]),
            source_npz=str(src.get("source_npz", "")),
            debug_npz=str(src.get("debug_npz", "")),
            racket=str(src.get("racket", "")),
            candidate=str(src.get("candidate", "")),
            handedness=str(src.get("handedness", "")),
            hit_time=float(data["hit_time"][idx]),
            hit_time_rel=float(hit_meta.get("hit_time_rel", float("nan"))),
            hit_index=int(data["hit_index"][idx]),
            frames=int(time[idx].shape[0]),
            fps=fps,
            max_ball_speed_mps=float(q.get("max_ball_speed_mps", float("nan"))),
            max_racket_speed_mps=float(q.get("max_racket_speed_mps", float("nan"))),
            max_racket_omega_radps=float(q.get("max_racket_omega_radps", float("nan"))),
            source_label=str(data["stroke_type"][idx]) if "stroke_type" in data.files else "",
            source_label_confidence=float(q.get("stroke_label_confidence", float("nan"))),
            v2_reason=str(data["stroke_label_reason_rule_v2"][idx]) if "stroke_label_reason_rule_v2" in data.files else "",
            quality_flags=q,
            source_json=src,
        )
        if label == "unknown":
            groups["unknown"].append(sample)
        elif conf < min_confidence:
            groups["low_confidence"].append(sample)

    for items in groups.values():
        items.sort(key=lambda item: (item.label, item.confidence, item.source_csv, item.episode_id))
    return groups


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_csv(path: Path, samples: list[RetargetSample]) -> None:
    fields = [
        "episode_id",
        "label",
        "confidence",
        "source_csv",
        "source_bvh",
        "sample_path",
        "racket",
        "candidate",
        "handedness",
        "hit_time",
        "hit_time_rel",
        "hit_index",
        "frames",
        "fps",
        "max_ball_speed_mps",
        "max_racket_speed_mps",
        "max_racket_omega_radps",
        "source_label",
        "source_label_confidence",
        "v2_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: sample.to_dict()[field] for field in fields})


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DATA260703 Retarget-Ready Split",
        "",
        f"- dataset: `{report['dataset_path']}`",
        f"- label field: `{report['label_field']}`",
        f"- min confidence: `{report['min_confidence']}`",
        f"- keep unknown: `{report['keep_unknown']}`",
        f"- selected samples: `{report['selected_count']}`",
        "",
        "## Label Counts",
        "",
        "| label | kept | seen |",
        "|---|---:|---:|",
    ]
    seen = report["stats"]["label_seen"]
    kept = report["stats"]["label_kept"]
    for label in sorted(seen):
        lines.append(f"| {label} | {kept.get(label, 0)} | {seen[label]} |")

    lines.extend(["", "## Source CSV Counts", "", "| source | count |", "|---|---:|"])
    for source, count in sorted(report["source_csv_counts"].items()):
        lines.append(f"| {source} | {count} |")

    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Notes", ""])
    lines.append("- `forehand` and `backhand` manifests are the current retarget queue.")
    lines.append("- `unknown` is excluded from the main queue by default, but exported separately for review.")
    lines.append("- Low-confidence known-label samples are also exported separately for review.")
    lines.append("- This split still uses Motive global meters; table/world success labels remain unavailable.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("analysis/mocap_cleaning_outputs/DATA260703_combined/stroke_relabel/DATA260703_combined_train_stroke_relabel.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_ready"),
    )
    parser.add_argument("--label-field", default="stroke_type_rule_v2")
    parser.add_argument("--confidence-field", default="stroke_confidence_rule_v2")
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--keep-unknown", action="store_true")
    parser.add_argument("--limit-per-label", type=int, default=None)
    args = parser.parse_args()

    samples, stats = _load_samples(
        dataset_path=args.dataset,
        label_field=args.label_field,
        confidence_field=args.confidence_field,
        min_confidence=float(args.min_confidence),
        keep_unknown=bool(args.keep_unknown),
        limit_per_label=args.limit_per_label,
    )
    review_groups = _collect_review_samples(
        dataset_path=args.dataset,
        label_field=args.label_field,
        confidence_field=args.confidence_field,
        min_confidence=float(args.min_confidence),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_label: dict[str, list[RetargetSample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    manifest = {
        "dataset_path": str(args.dataset),
        "label_field": args.label_field,
        "confidence_field": args.confidence_field,
        "min_confidence": float(args.min_confidence),
        "keep_unknown": bool(args.keep_unknown),
        "limit_per_label": args.limit_per_label,
        "selected_count": len(samples),
        "stats": stats,
        "samples": [sample.to_dict() for sample in samples],
    }
    all_json = args.output_dir / "retarget_manifest.json"
    all_csv = args.output_dir / "retarget_samples.csv"
    _write_json(all_json, manifest)
    _write_csv(all_csv, samples)

    outputs: dict[str, str] = {
        "retarget_manifest_json": str(all_json),
        "retarget_samples_csv": str(all_csv),
    }
    source_csv_counts = Counter(sample.source_csv for sample in samples)
    for label, label_samples in sorted(by_label.items()):
        path = args.output_dir / f"{label}_manifest.json"
        _write_json(
            path,
            {
                "label": label,
                "count": len(label_samples),
                "dataset_path": str(args.dataset),
                "min_confidence": float(args.min_confidence),
                "samples": [sample.to_dict() for sample in label_samples],
            },
        )
        outputs[f"{label}_manifest_json"] = str(path)

    for review_name, review_samples in sorted(review_groups.items()):
        path = args.output_dir / f"{review_name}_review_manifest.json"
        _write_json(
            path,
            {
                "review_type": review_name,
                "count": len(review_samples),
                "dataset_path": str(args.dataset),
                "min_confidence": float(args.min_confidence),
                "samples": [sample.to_dict() for sample in review_samples],
            },
        )
        outputs[f"{review_name}_review_manifest_json"] = str(path)

    report = {
        "dataset_path": str(args.dataset),
        "label_field": args.label_field,
        "min_confidence": float(args.min_confidence),
        "keep_unknown": bool(args.keep_unknown),
        "selected_count": len(samples),
        "stats": stats,
        "source_csv_counts": dict(source_csv_counts),
        "review_counts": {key: len(value) for key, value in review_groups.items()},
        "outputs": outputs,
    }
    summary_md = args.output_dir / "retarget_summary.md"
    _write_markdown(summary_md, report)
    outputs["retarget_summary_md"] = str(summary_md)
    report["outputs"] = outputs
    _write_json(args.output_dir / "retarget_report.json", report)

    print(f"Prepared {len(samples)} retarget-ready samples")
    for label, label_samples in sorted(by_label.items()):
        print(f"{label}: {len(label_samples)}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
