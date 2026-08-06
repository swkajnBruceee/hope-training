#!/usr/bin/env python3
"""Aggregate one-shot external-target grid reports across motion anchors."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def summarize_group(paths: list[Path]) -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not reports:
        raise ValueError("no target-response reports matched")
    if any(report.get("audit") != "external_racket_position_conditioning" for report in reports):
        raise ValueError("all inputs must be external racket position audit reports")

    trials = [
        trial
        for report in reports
        for trial in report["trials"]
        if trial is not None
    ]
    nonzero = [
        trial
        for trial in trials
        if _norm(trial["requested_offset_b_m"]) > 1.0e-9
    ]
    baselines = [
        trial
        for trial in trials
        if _norm(trial["requested_offset_b_m"]) <= 1.0e-9
    ]
    pairs = [pair for report in reports for pair in report["axis_pairs"]]

    useful = [
        trial
        for trial in nonzero
        if trial["nominal_paired_response"]["directional_cosine"] > 0.5
        and trial["nominal_paired_response"]["along_command_gain"] > 0.2
    ]
    actor_initial_delta = [
        trial["nominal_paired_response"]["initial_actor_response_l2"]
        for trial in nonzero
    ]
    actor_hit_delta = []
    for report in reports:
        complete_trials = [trial for trial in report["trials"] if trial is not None]
        baseline = next(
            trial
            for trial in complete_trials
            if _norm(trial["requested_offset_b_m"]) <= 1.0e-9
        )
        base_action = baseline["hit_upper_actor_output"]
        actor_hit_delta.extend(
            _norm(
                [
                    value - base
                    for value, base in zip(
                        trial["hit_upper_actor_output"], base_action
                    )
                ]
            )
            for trial in complete_trials
            if trial is not baseline
        )

    position_errors = [trial["position_error_m"] for trial in trials]
    baseline_position_errors = [
        trial["position_error_m"] for trial in baselines
    ]
    central_diagonal = [
        pair["position_jacobian_column"]["xyz".index(pair["axis"])]
        for pair in pairs
    ]
    radii = sorted({round(float(pair["radius_m"]), 6) for pair in pairs})
    radius_summaries = {}
    for radius in radii:
        selected = [
            pair
            for pair in pairs
            if abs(float(pair["radius_m"]) - radius) <= 2.0e-6
        ]
        total_diag = [
            pair["position_jacobian_column"]["xyz".index(pair["axis"])]
            for pair in selected
        ]
        radius_summary = {
            "pair_count": len(selected),
            "positive_total_diagonal_count": sum(value > 0.0 for value in total_diag),
            "median_total_diagonal_gain": _median(total_diag),
        }
        if selected and "root_position_jacobian_column" in selected[0]:
            root_diag = [
                pair["root_position_jacobian_column"]["xyz".index(pair["axis"])]
                for pair in selected
            ]
            arm_diag = [
                pair["racket_relative_root_jacobian_column"][
                    "xyz".index(pair["axis"])
                ]
                for pair in selected
            ]
            radius_summary.update(
                {
                    "median_root_diagonal_gain": _median(root_diag),
                    "median_racket_relative_root_diagonal_gain": _median(
                        arm_diag
                    ),
                }
            )
        radius_summaries[f"{radius:.3f}"] = radius_summary

    actor_median = _median(actor_initial_delta)
    useful_rate = len(useful) / max(len(nonzero), 1)
    position_10cm_rate = sum(error < 0.10 for error in position_errors) / max(
        len(position_errors), 1
    )
    if actor_median <= 1.0e-5:
        classification = "actor_ignores_external_position"
    elif useful_rate >= 0.60 and position_10cm_rate >= 0.80:
        classification = "locally_target_conditioned"
    else:
        classification = "actor_changes_but_racket_mapping_is_unreliable"

    return {
        "report_count": len(reports),
        "motion_ids": [int(report["motion_id"]) for report in reports],
        "complete_report_count": sum(bool(report["complete"]) for report in reports),
        "trial_count": len(trials),
        "nonzero_target_count": len(nonzero),
        "physical_termination_count": sum(
            int(report["physical_termination_count"]) for report in reports
        ),
        "position_error": {
            "mean_m": sum(position_errors) / len(position_errors),
            "max_m": max(position_errors),
            "under_0.10m_count": sum(error < 0.10 for error in position_errors),
            "under_0.10m_rate": position_10cm_rate,
            "baseline_mean_m": (
                sum(baseline_position_errors) / len(baseline_position_errors)
            ),
            "baseline_under_0.10m_count": sum(
                error < 0.10 for error in baseline_position_errors
            ),
        },
        "target_response": {
            "positive_direction_count": sum(
                trial["nominal_paired_response"]["directional_cosine"] > 0.0
                for trial in nonzero
            ),
            "useful_direction_and_gain_count": len(useful),
            "useful_direction_and_gain_rate": useful_rate,
            "central_diagonal_positive_count": sum(
                value > 0.0 for value in central_diagonal
            ),
            "central_pair_count": len(central_diagonal),
            "central_diagonal_median": _median(central_diagonal),
            "by_radius_m": radius_summaries,
        },
        "actor_response": {
            "initial_l2_median": actor_median,
            "hit_l2_median": _median(actor_hit_delta),
        },
        "classification": classification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-glob", required=True)
    parser.add_argument("--full-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixed_paths = [Path(path) for path in sorted(glob.glob(args.fixed_glob))]
    full_paths = [Path(path) for path in sorted(glob.glob(args.full_glob))]
    report = {
        "schema_version": 1,
        "audit": "external_racket_position_conditioning_summary",
        "thresholds": {
            "position_acceptance_m": 0.10,
            "useful_directional_cosine_min": 0.5,
            "useful_along_command_gain_min": 0.2,
            "local_conditioning_useful_rate_min": 0.60,
        },
        "fixed_model900": summarize_group(fixed_paths),
        "full_v30_stack": summarize_group(full_paths),
    }
    report["decision"] = (
        "train_local_target_residual_adapter"
        if report["fixed_model900"]["classification"]
        != "locally_target_conditioned"
        else "measure_existing_local_coverage_before_training"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
