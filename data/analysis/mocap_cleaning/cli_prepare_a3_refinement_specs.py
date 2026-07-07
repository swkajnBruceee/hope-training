#!/usr/bin/env python3
"""Generate per-job A3 constrained refinement specs."""

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

from analysis.mocap_cleaning.refinement_spec import build_refinement_spec, dump_json


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# A3 Refinement Specs",
        "",
        f"- spec version: `{report['spec_version']}`",
        f"- contract version: `{report['contract_version']}`",
        f"- source jobs: `{report['source_jobs']}`",
        f"- specs generated: `{report['spec_count']}`",
        "",
        "## Label Counts",
        "",
        "| label | count |",
        "|---|---:|",
    ]
    for label, count in sorted(report["label_counts"].items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- `spec_manifest_json`: `{report['spec_manifest_json']}`")
    lines.append(f"- `specs_root`: `{report['specs_root']}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Each spec contains coordinate contract, full racket transform metadata, joint masks, phase windows, and warning/reject thresholds.")
    lines.append("- These specs are solver inputs; they do not yet produce A3 joint trajectories.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs-manifest",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3/jobs_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3"),
    )
    args = parser.parse_args()

    jobs_manifest = json.loads(args.jobs_manifest.read_text())
    specs_root = args.output_dir / "refinement_specs"
    specs_root.mkdir(parents=True, exist_ok=True)

    specs = []
    label_counts = Counter()
    for job in jobs_manifest["jobs"]:
        spec = build_refinement_spec(job)
        label = str(job["label"])
        label_dir = specs_root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        spec_path = label_dir / f"{job['episode_id']}.json"
        dump_json(spec_path, spec)
        specs.append(
            {
                "job_id": job["job_id"],
                "episode_id": job["episode_id"],
                "label": label,
                "spec_path": str(spec_path),
                "quality_report_json": spec["artifacts"]["quality_report_json"],
                "retarget_csv": spec["artifacts"]["retarget_csv"],
                "motion_npz": spec["artifacts"]["motion_npz"],
            }
        )
        label_counts[label] += 1

    manifest = {
        "spec_version": "1.1.0",
        "contract_version": "a3_refinement_contract_v1",
        "source_jobs": str(args.jobs_manifest),
        "spec_count": len(specs),
        "specs_root": str(specs_root),
        "specs": specs,
    }
    spec_manifest_json = args.output_dir / "refinement_spec_manifest.json"
    dump_json(spec_manifest_json, manifest)

    report = {
        "spec_version": manifest["spec_version"],
        "contract_version": manifest["contract_version"],
        "source_jobs": str(args.jobs_manifest),
        "spec_count": len(specs),
        "label_counts": dict(label_counts),
        "spec_manifest_json": str(spec_manifest_json),
        "specs_root": str(specs_root),
    }
    summary_md = args.output_dir / "refinement_specs_summary.md"
    _write_markdown(summary_md, report)

    print(f"Prepared {len(specs)} refinement specs")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")
    print(f"Wrote {spec_manifest_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
