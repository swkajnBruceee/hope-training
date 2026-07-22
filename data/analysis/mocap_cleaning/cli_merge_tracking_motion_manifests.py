#!/usr/bin/env python3
"""Merge tracking motion manifests into one cumulative manifest."""

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


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Cumulative Tracking Motion Manifest",
        "",
        f"- merged manifests: `{len(report['source_manifests'])}`",
        f"- replay-ready motions: `{report['replay_ready_count']}`",
        "",
        "## Stroke Counts",
        "",
    ]
    for stroke, count in sorted(report["stroke_counts"].items()):
        lines.append(f"- `{stroke}`: {count}")
    lines.extend(["", "## Smoke Picks", ""])
    for stroke, episode_id in sorted(report["smoke_picks"].items()):
        lines.append(f"- `{stroke}`: `{episode_id}`")
    lines.extend(["", "## Sources", ""])
    for source in report["source_manifests"]:
        lines.append(f"- `{source}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-episode-id",
        action="append",
        default=[],
        help="Episode id to omit from the merged output; may be supplied more than once.",
    )
    args = parser.parse_args()

    merged: dict[str, dict] = {}
    source_manifests = []
    for manifest_path in args.manifest:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_manifests.append(str(manifest_path))
        for motion in payload["motions"]:
            merged[motion["episode_id"]] = motion

    excluded_episode_ids = {str(episode_id) for episode_id in args.exclude_episode_id}
    motions = sorted(
        (motion for episode_id, motion in merged.items() if episode_id not in excluded_episode_ids),
        key=lambda item: (item["stroke_type"], item["episode_id"]),
    )
    manifest = {
        "source_manifests": source_manifests,
        "excluded_episode_ids": sorted(excluded_episode_ids),
        "replay_ready_count": len(motions),
        "stroke_counts": dict(Counter(item["stroke_type"] for item in motions)),
        "smoke_picks": {},
        "motions": motions,
    }
    for motion in motions:
        manifest["smoke_picks"].setdefault(motion["stroke_type"], motion["episode_id"])

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = out_dir / "tracking_motion_manifest.json"
    forehand_json = out_dir / "tracking_motion_manifest_forehand.json"
    backhand_json = out_dir / "tracking_motion_manifest_backhand.json"
    summary_md = out_dir / "tracking_motion_manifest.md"

    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    forehand_json.write_text(
        json.dumps({**manifest, "stroke": "forehand", "motions": [m for m in motions if m["stroke_type"] == "forehand"]}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    backhand_json.write_text(
        json.dumps({**manifest, "stroke": "backhand", "motions": [m for m in motions if m["stroke_type"] == "backhand"]}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(manifest, summary_md)
    print(f"Wrote {manifest_json}")
    print(f"Wrote {forehand_json}")
    print(f"Wrote {backhand_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
