#!/usr/bin/env python3
"""Merge multiple replay-oriented tracking manifests.

This preserves motion entries verbatim, checks for duplicate episode IDs, and
emits combined all/forehand/backhand manifests for later replay or audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_subset(payload: dict[str, Any], stroke: str) -> dict[str, Any]:
    motions = [motion for motion in payload["motions"] if motion.get("stroke_type") == stroke]
    subset = dict(payload)
    subset["motions"] = motions
    subset["replay_ready_count"] = len(motions)
    subset["stroke_counts"] = dict(Counter(motion["stroke_type"] for motion in motions))
    subset["stroke"] = stroke
    subset["smoke_picks"] = {stroke: motions[0]["episode_id"]} if motions else {}
    return subset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="merged_tracking_library")
    parser.add_argument("--status", default="merged_replay_ready_not_training_approved")
    args = parser.parse_args()

    merged_motions: list[dict[str, Any]] = []
    seen_ids: dict[str, Path] = {}
    source_manifests: list[str] = []
    source_motion_npz_manifests: list[str] = []

    for manifest_path_raw in args.manifest:
        manifest_path = manifest_path_raw.expanduser().resolve()
        payload = _load_json(manifest_path)
        motions = payload.get("motions", [])
        if not motions:
            raise ValueError(f"{manifest_path} contains no motions")
        source_manifests.append(str(manifest_path))
        source_motion_npz = payload.get("source_motion_npz_manifest")
        if source_motion_npz is not None:
            source_motion_npz_manifests.append(str(source_motion_npz))
        for motion in motions:
            episode_id = str(motion.get("episode_id"))
            if episode_id in seen_ids:
                raise ValueError(
                    f"duplicate episode_id {episode_id} from {manifest_path} and {seen_ids[episode_id]}"
                )
            seen_ids[episode_id] = manifest_path
            merged_motions.append(motion)

    stroke_counts = Counter(str(motion.get("stroke_type", "unknown")) for motion in merged_motions)
    smoke_picks: dict[str, str] = {}
    for stroke in ("forehand", "backhand", "unknown"):
        for motion in merged_motions:
            if motion.get("stroke_type") == stroke:
                smoke_picks[stroke] = str(motion["episode_id"])
                break

    merged_payload = {
        "dataset_name": args.dataset_name,
        "source_manifests": source_manifests,
        "source_motion_npz_manifests": source_motion_npz_manifests,
        "replay_ready_count": len(merged_motions),
        "stroke_counts": dict(stroke_counts),
        "smoke_picks": smoke_picks,
        "dataset_status": args.status,
        "motions": merged_motions,
    }

    output_dir = args.output_dir.expanduser().resolve()
    all_path = output_dir / "tracking_motion_manifest.json"
    forehand_path = output_dir / "tracking_motion_manifest_forehand.json"
    backhand_path = output_dir / "tracking_motion_manifest_backhand.json"
    _write_json(all_path, merged_payload)
    _write_json(forehand_path, _build_subset(merged_payload, "forehand"))
    _write_json(backhand_path, _build_subset(merged_payload, "backhand"))

    summary_lines = [
        f"# {args.dataset_name}",
        "",
        f"- merged manifests: `{len(source_manifests)}`",
        f"- replay-ready motions: `{len(merged_motions)}`",
        "",
        "## Stroke Counts",
        "",
    ]
    for stroke, count in sorted(stroke_counts.items()):
        summary_lines.append(f"- `{stroke}`: {count}")
    summary_lines += [
        "",
        "## Outputs",
        "",
        f"- all: `{all_path}`",
        f"- forehand: `{forehand_path}`",
        f"- backhand: `{backhand_path}`",
        "",
        "## Notes",
        "",
        "- This is a merged replay/audit library, not an automatic training approval.",
        "- Motion entries are preserved verbatim from the source manifests.",
    ]
    (output_dir / "tracking_motion_manifest.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "replay_ready_count": len(merged_motions),
                "stroke_counts": dict(stroke_counts),
                "output": str(all_path),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
