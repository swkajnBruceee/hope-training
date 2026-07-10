#!/usr/bin/env python3
"""Build a cheap global candidate index before IK/optimization."""

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

import numpy as np

from analysis.mocap_cleaning.competition_candidate_utils import rows_from_dataset, write_csv
from analysis.mocap_cleaning.config import load_config


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Competition Candidate Index",
        "",
        f"- dataset: `{report['dataset']}`",
        f"- samples: `{report['sample_count']}`",
        f"- cheap quality pass: `{report['cheap_quality_pass_count']}`",
        "",
        "## Cheap Reject Reasons",
        "",
    ]
    for reason, count in sorted(report["cheap_reject_reason_counts"].items()):
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Stroke Counts", ""])
    for stroke, count in sorted(report["stroke_counts"].items()):
        lines.append(f"- `{stroke}`: {count}")
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = Path(str(config["dataset"]))
    output_dir = args.output_dir or Path(str(config["output_root"])) / "candidate_index"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(dataset_path, allow_pickle=True)
    rows = rows_from_dataset(data, config["selection"])
    reject_reasons = Counter(reason for row in rows for reason in row["cheap_reject_reasons"])
    report = {
        "stage": "competition_candidate_index",
        "config": str(args.config),
        "dataset": str(dataset_path),
        "sample_count": len(rows),
        "cheap_quality_pass_count": sum(1 for row in rows if row["cheap_quality_pass"]),
        "stroke_counts": dict(Counter(row["stroke_type"] for row in rows)),
        "cheap_reject_reason_counts": dict(reject_reasons),
        "selection": config["selection"],
        "candidates": rows,
        "outputs": {
            "json": str(output_dir / "competition_retarget_candidate_index.json"),
            "csv": str(output_dir / "competition_retarget_candidate_index.csv"),
            "summary": str(output_dir / "competition_retarget_candidate_index.md"),
        },
    }
    json_path = output_dir / "competition_retarget_candidate_index.json"
    csv_path = output_dir / "competition_retarget_candidate_index.csv"
    md_path = output_dir / "competition_retarget_candidate_index.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(rows, csv_path)
    _write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
