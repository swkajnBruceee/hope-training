#!/usr/bin/env python3
"""Analyze waist-pitch residual response across Strike reference workpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

import a3_base_contract as contract
import a3_base_zero_baseline_comparison as comparison
from analyze_a3_base_low_zoh_bundle import (
    _distribution,
    _metadata_index,
    _npz,
    _result_index,
    _symmetric_difference,
)


CASE_PATTERN = re.compile(
    r"^waist_pitch__strike_(?P<strike>[+-]\d+\.\d+)__base_(?P<base>[+-]\d+\.\d+)__r01$"
)
ZERO_PATTERN = re.compile(
    r"^waist_pitch_zero__strike_(?P<strike>[+-]\d+\.\d+)__r01$"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--isaac-results-dir", type=Path, required=True)
    parser.add_argument("--mujoco-results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_report(
    *, bundle_dir: Path, isaac_results_dir: Path, mujoco_results_dir: Path
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    bundle = json.loads(
        (bundle_dir / "bundle_report.json").read_text(encoding="utf-8")
    )
    metadata = _metadata_index(bundle_dir)
    isaac = _result_index(isaac_results_dir)
    mujoco = _result_index(mujoco_results_dir)
    expected = set(bundle["case_ids"])
    if set(isaac) != expected or set(mujoco) != expected:
        raise ValueError("waist workpoint result coverage differs from bundle")

    zero_by_workpoint = {}
    step_descriptors = []
    for case_id in bundle["case_ids"]:
        zero_match = ZERO_PATTERN.match(case_id)
        step_match = CASE_PATTERN.match(case_id)
        if zero_match:
            zero_by_workpoint[float(zero_match.group("strike"))] = case_id
        elif step_match:
            step_descriptors.append(
                (
                    case_id,
                    float(step_match.group("strike")),
                    float(step_match.group("base")),
                )
            )
        else:
            raise ValueError(f"unexpected waist workpoint case: {case_id}")
    if len(zero_by_workpoint) != 3 or len(step_descriptors) != 6:
        raise ValueError("waist workpoint bundle must contain 3 zeros and 6 residual steps")

    rows = []
    source_paths: set[Path] = {bundle_dir / "bundle_report.json"}
    for case_id, strike_reference, base_action in step_descriptors:
        zero_id = zero_by_workpoint[strike_reference]
        compared = comparison.compare_step_with_zero_baselines(
            isaac_step_result=isaac[case_id][0],
            isaac_step_evidence=_npz(isaac[case_id][1]),
            isaac_zero_result=isaac[zero_id][0],
            isaac_zero_evidence=_npz(isaac[zero_id][1]),
            mujoco_step_result=mujoco[case_id][0],
            mujoco_step_evidence=_npz(mujoco[case_id][1]),
            mujoco_zero_result=mujoco[zero_id][0],
            mujoco_zero_evidence=_npz(mujoco[zero_id][1]),
            step_trace_metadata=metadata[case_id][0],
            zero_trace_metadata=metadata[zero_id][0],
            classification_color="yellow",
            difference_labels=["expected_actuator_difference"],
            rationale=(
                "The paired zero baseline removes workpoint drift; low-ZOH friction "
                "ablation identified official MuJoCo passive frictionloss as causal."
            ),
        )
        rows.append(
            {
                "case_id": case_id,
                "zero_case_id": zero_id,
                "strike_reference_rad": strike_reference,
                "base_action": base_action,
                "commanded_residual_rad": compared["commanded_joint_delta_rad"],
                "direction_consistent": compared[
                    "baseline_corrected_response_direction_consistent"
                ],
                "isaac_gain": compared["isaac_baseline_corrected_gain"],
                "mujoco_official_gain": compared["mujoco_baseline_corrected_gain"],
                "gain_symmetric_difference": compared["gain_symmetric_difference"],
                "isaac_effort_rms_nm": isaac[case_id][0]["metrics"][
                    "selected_joint_effort_rms_nm"
                ],
                "mujoco_effort_rms_nm": mujoco[case_id][0]["metrics"][
                    "selected_joint_effort_rms_nm"
                ],
                "isaac_saturation_duration_s": isaac[case_id][0]["metrics"][
                    "selected_joint_saturation_duration_s"
                ],
                "mujoco_saturation_duration_s": mujoco[case_id][0]["metrics"][
                    "selected_joint_saturation_duration_s"
                ],
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
                isaac[case_id][1], isaac[case_id][2],
                isaac[zero_id][1], isaac[zero_id][2],
                mujoco[case_id][1], mujoco[case_id][2],
                mujoco[zero_id][1], mujoco[zero_id][2],
                metadata[case_id][1], metadata[zero_id][1],
            }
        )

    asymmetry = []
    for workpoint in sorted(zero_by_workpoint):
        pair = [row for row in rows if row["strike_reference_rad"] == workpoint]
        if len(pair) != 2:
            raise ValueError("each workpoint must have positive and negative residuals")
        positive = next(row for row in pair if row["base_action"] > 0.0)
        negative = next(row for row in pair if row["base_action"] < 0.0)
        asymmetry.append(
            {
                "strike_reference_rad": workpoint,
                "isaac_positive_negative_gain_symmetric_difference": (
                    _symmetric_difference(positive["isaac_gain"], negative["isaac_gain"])
                ),
                "mujoco_positive_negative_gain_symmetric_difference": (
                    _symmetric_difference(
                        positive["mujoco_official_gain"],
                        negative["mujoco_official_gain"],
                    )
                ),
                "lower_gain_direction": (
                    "positive"
                    if positive["mujoco_official_gain"] < negative["mujoco_official_gain"]
                    else "negative"
                ),
            }
        )
    isaac_gains = [float(row["isaac_gain"]) for row in rows]
    mujoco_gains = [float(row["mujoco_official_gain"]) for row in rows]
    report = {
        "schema_version": 1,
        "artifact_status": "phase0_waist_workpoint_diagnostic_not_promotion_evidence",
        "scope": bundle["scope"],
        "matrix_sha256": bundle["matrix_sha256"],
        "case_counts": {"zero": 3, "residual_step": 6},
        "all_step_directions_consistent": all(row["direction_consistent"] for row in rows),
        "all_safety_envelopes_passed": all(
            row["all_safety_envelopes_passed"] for row in rows
        ),
        "any_actuator_saturation": any(
            row["isaac_saturation_duration_s"] > 0.0
            or row["mujoco_saturation_duration_s"] > 0.0
            for row in rows
        ),
        "gain_distribution": {
            "isaac": _distribution(isaac_gains),
            "mujoco_official": _distribution(mujoco_gains),
            "cross_engine_symmetric_difference": _distribution(
                [float(row["gain_symmetric_difference"]) for row in rows]
            ),
        },
        "directional_asymmetry_by_workpoint": asymmetry,
        "diagnostic_conclusion": {
            "action_sign_or_composer_error_supported": False,
            "effort_saturation_supported": False,
            "residual_limit_expansion_supported": False,
            "workpoint_dependent_passive_load_interaction_supported": True,
            "reason": (
                "Isaac gain is nearly invariant while official MuJoCo gain and the "
                "lower-response direction vary strongly by workpoint without saturation."
            ),
            "waist_residual_limit_frozen": False,
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
        bundle_dir=args.bundle_dir,
        isaac_results_dir=args.isaac_results_dir,
        mujoco_results_dir=args.mujoco_results_dir,
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
                "gain_distribution": report["gain_distribution"],
                "directional_asymmetry_by_workpoint": report[
                    "directional_asymmetry_by_workpoint"
                ],
                "diagnostic_conclusion": report["diagnostic_conclusion"],
                "qualification_status": report["qualification_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
