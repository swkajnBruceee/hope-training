#!/usr/bin/env python3
"""Aggregate a bounded step bundle using nominal zero baselines from another bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import a3_base_contract as contract
import a3_base_zero_baseline_comparison as comparison
from analyze_a3_base_low_zoh_bundle import (
    _distribution,
    _metadata_index,
    _mirror_summary,
    _npz,
    _result_index,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-bundle-dir", type=Path, required=True)
    parser.add_argument("--zero-bundle-dir", type=Path, required=True)
    parser.add_argument("--isaac-step-results-dir", type=Path, required=True)
    parser.add_argument("--isaac-zero-results-dir", type=Path, required=True)
    parser.add_argument("--mujoco-step-results-dir", type=Path, required=True)
    parser.add_argument("--mujoco-zero-results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_report(
    *,
    step_bundle_dir: Path,
    zero_bundle_dir: Path,
    isaac_step_results_dir: Path,
    isaac_zero_results_dir: Path,
    mujoco_step_results_dir: Path,
    mujoco_zero_results_dir: Path,
) -> dict[str, Any]:
    step_bundle_dir = step_bundle_dir.resolve()
    zero_bundle_dir = zero_bundle_dir.resolve()
    step_bundle = json.loads(
        (step_bundle_dir / "bundle_report.json").read_text(encoding="utf-8")
    )
    step_metadata = _metadata_index(step_bundle_dir)
    zero_metadata = _metadata_index(zero_bundle_dir)
    isaac_step = _result_index(isaac_step_results_dir)
    isaac_zero = _result_index(isaac_zero_results_dir)
    mujoco_step = _result_index(mujoco_step_results_dir)
    mujoco_zero = _result_index(mujoco_zero_results_dir)
    expected_step_ids = set(step_bundle["case_ids"])
    for label, index in (("Isaac", isaac_step), ("MuJoCo", mujoco_step)):
        if set(index) != expected_step_ids:
            raise ValueError(f"{label} step coverage differs from bundle")
    matrix_hashes = {
        item[0]["matrix_sha256"]
        for index in (isaac_step, isaac_zero, mujoco_step, mujoco_zero)
        for item in index.values()
    }
    if matrix_hashes != {step_bundle["matrix_sha256"]}:
        raise ValueError("step and zero artifacts do not share one matrix")
    zero_by_joint = {
        value[0]["runner_facts"]["selected_joint_name"]: case_id
        for case_id, value in isaac_zero.items()
        if value[0]["case_validation"]["category"] == "joint_zero_baseline"
    }
    if len(zero_by_joint) != 14:
        raise ValueError("nominal zero bundle must cover all 14 Base joints")

    rows = []
    source_paths: set[Path] = {
        step_bundle_dir / "bundle_report.json",
        zero_bundle_dir / "bundle_report.json",
    }
    for case_id in step_bundle["case_ids"]:
        if isaac_step[case_id][0]["case_validation"]["category"] != "base_action_step":
            raise ValueError("step bundle contains a non-step case")
        joint = isaac_step[case_id][0]["runner_facts"]["selected_joint_name"]
        zero_id = zero_by_joint[joint]
        if zero_id not in mujoco_zero or zero_id not in zero_metadata:
            raise ValueError(f"missing cross-engine zero baseline for {joint}")
        compared = comparison.compare_step_with_zero_baselines(
            isaac_step_result=isaac_step[case_id][0],
            isaac_step_evidence=_npz(isaac_step[case_id][1]),
            isaac_zero_result=isaac_zero[zero_id][0],
            isaac_zero_evidence=_npz(isaac_zero[zero_id][1]),
            mujoco_step_result=mujoco_step[case_id][0],
            mujoco_step_evidence=_npz(mujoco_step[case_id][1]),
            mujoco_zero_result=mujoco_zero[zero_id][0],
            mujoco_zero_evidence=_npz(mujoco_zero[zero_id][1]),
            step_trace_metadata=step_metadata[case_id][0],
            zero_trace_metadata=zero_metadata[zero_id][0],
            classification_color="yellow",
            difference_labels=["expected_actuator_difference"],
            rationale=(
                "The low-amplitude causal ablation identified official MuJoCo "
                "passive frictionloss as the primary displacement-gain difference."
            ),
        )
        rows.append(
            {
                "step_case_id": case_id,
                "zero_case_id": zero_id,
                "selected_joint_name": joint,
                "commanded_joint_delta_rad": compared["commanded_joint_delta_rad"],
                "direction_consistent": compared[
                    "baseline_corrected_response_direction_consistent"
                ],
                "isaac_gain": compared["isaac_baseline_corrected_gain"],
                "mujoco_official_gain": compared["mujoco_baseline_corrected_gain"],
                "gain_symmetric_difference": compared["gain_symmetric_difference"],
                "isaac_end_window_slope_radps": compared[
                    "isaac_response_end_window_slope_radps"
                ],
                "mujoco_end_window_slope_radps": compared[
                    "mujoco_response_end_window_slope_radps"
                ],
                "all_safety_envelopes_passed": compared[
                    "all_safety_envelopes_passed"
                ],
            }
        )
        source_paths.update(
            {
                isaac_step[case_id][1], isaac_step[case_id][2],
                isaac_zero[zero_id][1], isaac_zero[zero_id][2],
                mujoco_step[case_id][1], mujoco_step[case_id][2],
                mujoco_zero[zero_id][1], mujoco_zero[zero_id][2],
                step_metadata[case_id][1], zero_metadata[zero_id][1],
            }
        )
    differences = [float(row["gain_symmetric_difference"]) for row in rows]
    worst = max(rows, key=lambda row: float(row["gain_symmetric_difference"]))
    report = {
        "schema_version": 1,
        "artifact_status": "phase0_step_bundle_diagnostic_not_promotion_evidence",
        "scope": step_bundle["scope"],
        "matrix_sha256": step_bundle["matrix_sha256"],
        "case_count": len(rows),
        "all_step_directions_consistent": all(row["direction_consistent"] for row in rows),
        "all_safety_envelopes_passed": all(
            row["all_safety_envelopes_passed"] for row in rows
        ),
        "gain_symmetric_difference": _distribution(differences),
        "case_count_at_or_below_d_gain_0p25_for_diagnostics_only": sum(
            value <= 0.25 for value in differences
        ),
        "worst_case": worst,
        "left_right_mirror": {
            "isaac": _mirror_summary(rows, "isaac_gain"),
            "mujoco_official": _mirror_summary(rows, "mujoco_official_gain"),
        },
        "classification": {
            "primary_difference_label": "expected_actuator_difference",
            "classification_basis": "low_zoh_passive_friction_ablation",
            "unexplained_symbol_or_order_error_present": False,
            "training_joint_friction_contract_frozen": False,
        },
        "qualification_status": {
            "fixture_runner_qualified": True,
            "fixture_matrix_approved": False,
            "stand_task_approved": False,
            "locomotion_command_approved": False,
            "deployment_approved": False,
        },
        "automatic_promotion": False,
        "cases": rows,
        "source_sha256": {
            str(path): contract.file_sha256(path) for path in sorted(source_paths)
        },
    }
    report["source_sha256"][str(Path(__file__).resolve())] = contract.file_sha256(
        Path(__file__).resolve()
    )
    return report


def main() -> None:
    args = _parser().parse_args()
    report = build_report(
        step_bundle_dir=args.step_bundle_dir,
        zero_bundle_dir=args.zero_bundle_dir,
        isaac_step_results_dir=args.isaac_step_results_dir,
        isaac_zero_results_dir=args.isaac_zero_results_dir,
        mujoco_step_results_dir=args.mujoco_step_results_dir,
        mujoco_zero_results_dir=args.mujoco_zero_results_dir,
    )
    output = args.output.expanduser().resolve()
    if output.suffix != ".json":
        raise ValueError("--output must end in .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "case_count": report["case_count"],
                "all_step_directions_consistent": report[
                    "all_step_directions_consistent"
                ],
                "gain_symmetric_difference": report["gain_symmetric_difference"],
                "worst_case": report["worst_case"],
                "qualification_status": report["qualification_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
