#!/usr/bin/env python3
"""Batch runner for A3 refinement solver modes.

Mode `passthrough` validates specs and copies generic-retarget CSV to refined CSV
when available, while still writing per-job metrics and a run summary.
"""

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
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from analysis.mocap_cleaning.a3_refinement_solver import run_refine_mode


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# A3 Refinement Solver Run",
        "",
        f"- mode: `{report['mode']}`",
        f"- manifest: `{report['manifest']}`",
        f"- processed: `{report['processed']}`",
        f"- succeeded: `{report['succeeded']}`",
        f"- skipped: `{report['skipped']}`",
        f"- failed: `{report['failed']}`",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in sorted(report["status_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_passthrough(spec: dict[str, Any], write_metrics: bool) -> tuple[str, dict[str, Any]]:
    generic_csv = Path(spec["artifacts"]["generic_retarget_csv"])
    refined_csv = Path(spec["artifacts"]["refined_retarget_csv"])
    metrics = {
        "job_id": spec["job_id"],
        "mode": "passthrough",
        "status": None,
        "input_exists": generic_csv.exists(),
        "output_path": str(refined_csv),
        "validation_status": "not_run",
        "warnings": [],
        "reject_reasons": [],
    }
    if not generic_csv.exists():
        metrics["status"] = "skipped_missing_generic_retarget_csv"
        metrics["reject_reasons"].append("generic_retarget_csv_missing")
        return metrics["status"], metrics

    refined_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(generic_csv, refined_csv)
    metrics["status"] = "passed_passthrough"
    metrics["validation_status"] = "warning"
    metrics["warnings"].append("passthrough_mode_no_refinement_applied")
    if write_metrics:
        _write_json(Path(spec["artifacts"]["quality_report_json"]), metrics)
    return metrics["status"], metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3/refinement_spec_manifest.json"),
    )
    parser.add_argument("--mode", choices=("passthrough", "refine"), default="passthrough")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--write-metrics", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3"),
    )
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    specs = manifest["specs"]
    if args.limit is not None:
        specs = specs[: max(0, int(args.limit))]

    status_counts = Counter()
    failures = []
    for item in specs:
        spec = _load_json(Path(item["spec_path"]))
        if args.mode == "passthrough":
            status, metrics = _run_passthrough(spec, write_metrics=bool(args.write_metrics))
        else:
            result = run_refine_mode(spec, write_metrics=bool(args.write_metrics))
            status = result.status
            metrics = result.metrics
        status_counts[status] += 1
        if status.startswith("failed"):
            failures.append(metrics)
            if args.fail_fast:
                break

    processed = sum(status_counts.values())
    report = {
        "mode": args.mode,
        "manifest": str(args.manifest),
        "processed": processed,
        "succeeded": int(sum(v for k, v in status_counts.items() if k.startswith("passed"))),
        "skipped": int(sum(v for k, v in status_counts.items() if k.startswith("skipped"))),
        "failed": int(sum(v for k, v in status_counts.items() if k.startswith("failed"))),
        "status_counts": dict(status_counts),
        "failures": failures[:20],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"solver_run_{args.mode}.json"
    md_path = args.output_dir / f"solver_run_{args.mode}.md"
    _write_json(json_path, report)
    _write_markdown(md_path, report)
    print(f"Processed {processed} specs")
    print(f"Succeeded {report['succeeded']}")
    print(f"Skipped {report['skipped']}")
    print(f"Failed {report['failed']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
