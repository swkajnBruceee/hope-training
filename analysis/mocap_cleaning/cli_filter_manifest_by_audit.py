#!/usr/bin/env python3
"""Create a filtered manifest from a CleanSample audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.audit_report.read_text())
    ok_by_episode = {
        item["episode_id"]: bool(item["audit_ok"])
        for item in audit["items"]
    }
    filtered_samples = [
        sample for sample in manifest["samples"]
        if ok_by_episode.get(sample["episode_id"], False)
    ]
    filtered = {
        **manifest,
        "source_manifest": str(args.manifest),
        "audit_report": str(args.audit_report),
        "filter_rule": "audit_ok == true",
        "original_sample_count": len(manifest["samples"]),
        "filtered_sample_count": len(filtered_samples),
        "removed_sample_count": len(manifest["samples"]) - len(filtered_samples),
        "samples": filtered_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(filtered, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")
    print(f"Kept {len(filtered_samples)} / {len(manifest['samples'])} samples")


if __name__ == "__main__":
    main()
