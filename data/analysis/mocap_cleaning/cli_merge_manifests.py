#!/usr/bin/env python3
"""Merge CleanSample manifests without modifying sample files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--only-usable", action="store_true")
    args = parser.parse_args()

    samples = []
    source_counts = Counter()
    for manifest_path in args.manifest:
        manifest = json.loads(manifest_path.read_text())
        selected = manifest["samples"]
        if args.only_usable:
            selected = [sample for sample in selected if bool(sample["usable_for_training"])]
        for sample in selected:
            merged_sample = dict(sample)
            merged_sample["source_manifest"] = str(manifest_path)
            samples.append(merged_sample)
            source_counts[str(manifest_path)] += 1

    seen = set()
    deduped = []
    duplicate_ids = []
    for sample in samples:
        episode_id = sample["episode_id"]
        if episode_id in seen:
            duplicate_ids.append(episode_id)
            continue
        seen.add(episode_id)
        deduped.append(sample)

    merged = {
        "dataset_id": args.dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifests": [str(path) for path in args.manifest],
        "only_usable": args.only_usable,
        "source_counts": dict(source_counts),
        "pre_dedup_sample_count": len(samples),
        "merged_sample_count": len(deduped),
        "duplicate_episode_ids": duplicate_ids,
        "samples": deduped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")
    print(f"Merged {len(deduped)} samples")
    print(f"Duplicates skipped {len(duplicate_ids)}")


if __name__ == "__main__":
    main()
