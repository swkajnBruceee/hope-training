#!/usr/bin/env python3
"""Export CleanSample NPZ files from hit detection outputs."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    del _ROOT

import argparse
from collections import Counter
import json
from datetime import datetime, timezone
from pathlib import Path

from analysis.mocap_cleaning.config import load_config
from analysis.mocap_cleaning.sample_export import export_clean_sample


def _write_markdown(manifest: dict, path: Path) -> None:
    lines = [
        "# DATA260703 CleanSample Manifest",
        "",
        f"Created at: `{manifest['created_at']}`",
        "",
        "| Episode | Frames | Hit Index | Hit t_rel (s) | Usable | Sample |",
        "|---|---:|---:|---:|---|---|",
    ]
    for sample in manifest["samples"]:
        lines.append(
            f"| `{sample['episode_id']}` | {sample['frames']} | {sample['hit_index']} | "
            f"{sample['hit_time_rel']:.3f} | {sample['usable_for_training']} | `{sample['sample_path']}` |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append("- `racket_quat` comes from Motive rigid body rotation in xyzw order.")
    lines.append("- `racket_omega` is computed from frame-to-frame quaternion deltas in rad/s.")
    lines.append("- CleanSample time axes are hit-centered and resampled to the configured target FPS.")
    lines.append("- `success=-1` means unknown because table/world landing labels are not available yet.")
    reasons = Counter(
        sample["quality_flags"].get("success_label_reason", "unknown")
        for sample in manifest["samples"]
    )
    reliability = Counter(
        bool(sample["quality_flags"].get("success_label_reliable", False))
        for sample in manifest["samples"]
    )
    lines.extend(["", "## Label Status", ""])
    lines.append(f"- Reliable success labels: {reliability[True]}")
    lines.append(f"- Unreliable/unknown success labels: {reliability[False]}")
    for reason, count in sorted(reasons.items()):
        lines.append(f"- `{reason}`: {count}")
    stroke_counts = Counter(
        sample["quality_flags"].get("stroke_type", "unknown")
        for sample in manifest["samples"]
    )
    stroke_reasons = Counter(
        sample["quality_flags"].get("stroke_label_reason", "unknown")
        for sample in manifest["samples"]
    )
    lines.extend(["", "## Stroke Status", ""])
    for stroke, count in sorted(stroke_counts.items()):
        lines.append(f"- `{stroke}`: {count}")
    for reason, count in sorted(stroke_reasons.items()):
        lines.append(f"- `{reason}`: {count}")
    usable_counts = Counter(bool(sample["usable_for_training"]) for sample in manifest["samples"])
    omega_counts = Counter(
        bool(sample["quality_flags"].get("racket_omega_reasonable", False))
        for sample in manifest["samples"]
    )
    ball_speed_counts = Counter(
        bool(sample["quality_flags"].get("ball_speed_reasonable", False))
        for sample in manifest["samples"]
    )
    racket_speed_counts = Counter(
        bool(sample["quality_flags"].get("racket_speed_reasonable", False))
        for sample in manifest["samples"]
    )
    lines.extend(["", "## Quality Status", ""])
    lines.append(f"- Usable for training: {usable_counts[True]}")
    lines.append(f"- Not usable for training: {usable_counts[False]}")
    lines.append(f"- Ball speed reasonable: {ball_speed_counts[True]}")
    lines.append(f"- Racket speed reasonable: {racket_speed_counts[True]}")
    lines.append(f"- Racket omega reasonable: {omega_counts[True]}")
    lines.append(f"- Racket omega rejected: {omega_counts[False]}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/DATA260703.yaml"))
    parser.add_argument("--hit-report", type=Path, default=Path("data/analysis/mocap_cleaning_outputs/DATA260703/hit_detection_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/mocap_cleaning_outputs/DATA260703"))
    parser.add_argument("--pre-hit-s", type=float, default=None)
    parser.add_argument("--post-hit-s", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    pre_hit_s = float(args.pre_hit_s if args.pre_hit_s is not None else config["episode"]["pre_hit_s"])
    post_hit_s = float(args.post_hit_s if args.post_hit_s is not None else config["episode"]["post_hit_s"])
    hit_report = json.loads(args.hit_report.read_text())
    samples_dir = args.output_dir / "samples"
    metadata_dir = args.output_dir / "metadata"
    samples_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for item in hit_report["hits"]:
        if not item["usable_for_hit_analysis"]:
            continue
        episode_id = item["episode_id"]
        result = export_clean_sample(
            episode_id=episode_id,
            source_npz=item["source_npz"],
            debug_npz=item["debug_npz"],
            source_csv=item["source_csv"],
            source_bvh=item["source_bvh"],
            racket=item["racket"],
            candidate=item["candidate"],
            hit_metadata=item["hit"],
            cleaning_usable=item["cleaning_usable"],
            output_npz=str(samples_dir / f"{episode_id}.npz"),
            output_metadata=str(metadata_dir / f"{episode_id}.json"),
            pre_hit_s=pre_hit_s,
            post_hit_s=post_hit_s,
            target_fps=float(config["fps"]["clean_sample"]),
            max_ball_speed_mps=float(config["speed_thresholds"]["ball_mps"]),
            max_racket_speed_mps=float(config["speed_thresholds"]["racket_mps"]),
            max_racket_omega_radps=float(config["speed_thresholds"]["racket_omega_radps"]),
            table_config=config.get("table"),
            handedness=config["entities"]["rackets"][item["racket"]].get("handedness", "right"),
        )
        results.append(result.to_dict())

    manifest = {
        "dataset_id": config["dataset_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "hit_report": str(args.hit_report),
        "samples": results,
    }
    manifest_json = args.output_dir / "manifest.json"
    manifest_md = args.output_dir / "manifest.md"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(manifest, manifest_md)
    print(f"Wrote {len(results)} samples")
    print(f"Wrote {manifest_json}")
    print(f"Wrote {manifest_md}")


if __name__ == "__main__":
    main()
