#!/usr/bin/env python3
"""Evaluate a representative subset of A3 retarget/refinement specs."""

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
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from analysis.mocap_cleaning.cli_generate_generic_retarget_init import build_generic_init_csv
from analysis.mocap_cleaning.a3_refinement_solver import run_refine_mode, write_retarget_csv
from analysis.mocap_cleaning.generic_init_temporal_repair import write_temporal_repair
from analysis.mocap_cleaning.refinement_spec import resolve_existing_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sample_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    sample_npz = resolve_existing_path(spec["inputs"]["source_sample_npz"])
    meta = sample_npz.parent.parent / "metadata" / f"{spec['episode_id']}.json"
    return _load_json(resolve_existing_path(meta))


def _select_specs(spec_manifest: dict[str, Any], per_label: int) -> list[dict[str, Any]]:
    by_label: dict[str, list[tuple[str, float, dict[str, Any]]]] = defaultdict(list)
    for item in spec_manifest["specs"]:
        spec = _load_json(Path(item["spec_path"]))
        meta = _sample_metadata(spec)
        source_csv = str(meta["source"]["source_csv"])
        by_label[spec["label"]].append((source_csv, float(spec["confidence"]), spec))

    selected: list[dict[str, Any]] = []
    for label, items in sorted(by_label.items()):
        per_source: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        for source_csv, confidence, spec in items:
            per_source[source_csv].append((confidence, spec))
        for values in per_source.values():
            values.sort(key=lambda x: (-x[0], x[1]["episode_id"]))

        ordered_sources = sorted(per_source.keys())
        picked = 0
        round_idx = 0
        while picked < per_label and ordered_sources:
            source_csv = ordered_sources[round_idx % len(ordered_sources)]
            if per_source[source_csv]:
                _, spec = per_source[source_csv].pop(0)
                selected.append(spec)
                picked += 1
            round_idx += 1
            if round_idx > len(ordered_sources) * (per_label + 2):
                break
    return selected


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# A3 Retarget Subset Evaluation",
        "",
        f"- selected specs: `{report['selected_count']}`",
        f"- per label target: `{report['per_label']}`",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in sorted(report["status_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Medians", ""])
    for key, value in report["medians"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Reject Reasons", ""])
    for key, value in sorted(report["reject_reason_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3/refinement_spec_manifest.json"),
    )
    parser.add_argument("--per-label", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3/subset_eval"),
    )
    args = parser.parse_args()

    spec_manifest = _load_json(args.manifest)
    selected = _select_specs(spec_manifest, per_label=int(args.per_label))

    results = []
    status_counts = Counter()
    reject_reason_counts = Counter()
    metrics_series: dict[str, list[float]] = defaultdict(list)

    for spec in selected:
        csv_data, diagnostics = build_generic_init_csv(spec)
        write_retarget_csv(spec["artifacts"]["generic_retarget_csv"], csv_data)
        diagnostics_path = Path(spec["artifacts"]["generic_retarget_csv"]).with_suffix(".diagnostics.json")
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n")
        repair_info = write_temporal_repair(
            csv_path=spec["artifacts"]["generic_retarget_csv"],
            diagnostics_path=str(diagnostics_path),
        )
        result = run_refine_mode(spec, write_metrics=True)
        metrics = result.metrics
        metadata = _sample_metadata(spec)
        source_csv = str(metadata["source"]["source_csv"])
        status_counts[result.status] += 1
        for reason in metrics.get("validation_reject_reasons", []):
            reject_reason_counts[reason] += 1
        for key in (
            "racket_position_error_at_hit_m",
            "racket_orientation_error_at_hit_deg",
            "racket_velocity_direction_error_at_hit_deg",
            "max_joint_velocity_radps",
            "max_joint_acceleration_radps2",
        ):
            value = metrics.get(key)
            if value is not None:
                metrics_series[key].append(float(value))
        results.append(
            {
                "episode_id": spec["episode_id"],
                "label": spec["label"],
                "source_csv": source_csv,
                "status": result.status,
                "kept_generic_init_baseline": bool(metrics.get("kept_generic_init_baseline", False)),
                "generic_init_max_velocity_radps": diagnostics.get("max_velocity_radps"),
                "generic_init_max_acceleration_radps2": diagnostics.get("max_acceleration_radps2"),
                "temporal_repair_max_velocity_radps": repair_info["post_temporal_repair"]["max_velocity_radps"],
                "temporal_repair_max_acceleration_radps2": repair_info["post_temporal_repair"]["max_acceleration_radps2"],
                "racket_position_error_at_hit_m": metrics.get("racket_position_error_at_hit_m"),
                "racket_orientation_error_at_hit_deg": metrics.get("racket_orientation_error_at_hit_deg"),
                "racket_velocity_direction_error_at_hit_deg": metrics.get("racket_velocity_direction_error_at_hit_deg"),
                "max_joint_velocity_radps": metrics.get("max_joint_velocity_radps"),
                "max_joint_acceleration_radps2": metrics.get("max_joint_acceleration_radps2"),
                "validation_reject_reasons": metrics.get("validation_reject_reasons", []),
            }
        )

    medians = {key: median(values) for key, values in metrics_series.items() if values}
    report = {
        "manifest": str(args.manifest),
        "selected_count": len(selected),
        "per_label": int(args.per_label),
        "status_counts": dict(status_counts),
        "reject_reason_counts": dict(reject_reason_counts),
        "medians": medians,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "subset_eval.json", report)
    _write_markdown(args.output_dir / "subset_eval.md", report)
    print(f"Evaluated {len(selected)} specs")
    for key, value in sorted(status_counts.items()):
        print(f"{key}: {value}")
    print(f"Wrote {args.output_dir / 'subset_eval.json'}")
    print(f"Wrote {args.output_dir / 'subset_eval.md'}")


if __name__ == "__main__":
    main()
