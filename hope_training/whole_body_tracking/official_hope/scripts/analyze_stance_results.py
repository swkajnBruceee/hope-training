#!/usr/bin/env python3
"""Aggregate stance result CSVs and generate the reproducible report bundle."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "mujoco_reference" / "reference"
sys.path.insert(0, str(REF))

from a3_deploy_onnx_ref_pingpong.stance_stability import aggregate_rows, write_rows  # noqa: E402
from stance_stability_test import default_model, plot_results, write_report  # noqa: E402
from a3_deploy_onnx_ref_pingpong.stance_stability import StanceMujoco  # noqa: E402


def discover(values: list[str]) -> list[Path]:
    found = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            found.extend(sorted(path.glob("stance_results.csv")))
        else:
            found.append(path)
    return found


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output-dir", default="outputs/stance_stability/combined")
    p.add_argument("--report-path", default="STANCE_STABILITY_REPORT.md")
    p.add_argument("--model-xml", default=str(default_model()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pd-profile", default="official_stand")
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in discover(args.inputs):
        with path.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    write_rows(out / "stance_results.csv", rows)
    group_keys = ("hip_deg", "knee_deg", "torso_deg", "stance_width_scale", "fore_aft_m", "lead_leg", "test_type")
    write_rows(out / "stance_summary.csv", aggregate_rows(rows, group_keys))
    plot_results(rows, out)
    sim = StanceMujoco(args.model_xml, seed=args.seed)
    report_args = SimpleNamespace(test="combined", trials="static=5, push=1, swing=1", seed=args.seed, pd_profile=args.pd_profile)
    write_report(out, rows, sim, report_args)
    report = Path(args.report_path)
    report.write_text((out / "STANCE_STABILITY_REPORT.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"combined {len(rows)} rows -> {out}")
    print(f"report -> {report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
