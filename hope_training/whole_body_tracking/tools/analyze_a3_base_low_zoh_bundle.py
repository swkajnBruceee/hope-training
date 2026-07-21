#!/usr/bin/env python3
"""Aggregate the 14-DOF low-amplitude ZOH fixture and friction ablation.

This report is diagnostic evidence only.  It deliberately does not contain an
automatic threshold that can promote the Stand task.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import a3_base_contract as contract
import a3_base_zero_baseline_comparison as comparison


EXPECTED_ZERO_COUNT = 14
EXPECTED_STEP_COUNT = 28


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--isaac-results-dir", type=Path, required=True)
    parser.add_argument("--mujoco-results-dir", type=Path, required=True)
    parser.add_argument("--mujoco-friction-zero-results-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _result_index(directory: Path) -> dict[str, tuple[dict[str, Any], Path, Path]]:
    index: dict[str, tuple[dict[str, Any], Path, Path]] = {}
    for result_path in sorted(directory.resolve().glob("*.json")):
        if result_path.name.endswith("bundle_report.json"):
            continue
        payload = _json(result_path)
        case_id = payload.get("case_id")
        if not isinstance(case_id, str):
            continue
        evidence_path = result_path.with_suffix(".trace.npz")
        if not evidence_path.is_file():
            raise ValueError(f"missing evidence for {result_path}")
        if case_id in index:
            raise ValueError(f"duplicate result case: {case_id}")
        index[case_id] = (payload, evidence_path, result_path)
    return index


def _metadata_index(bundle_dir: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    index: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted((bundle_dir.resolve() / "traces").glob("*.trace.json")):
        payload = _json(path)
        case_id = payload.get("case_id")
        if not isinstance(case_id, str) or case_id in index:
            raise ValueError(f"invalid or duplicate trace metadata: {path}")
        index[case_id] = (payload, path)
    return index


def _symmetric_difference(first: float, second: float) -> float:
    denominator = 0.5 * (abs(first) + abs(second))
    return abs(first - second) / denominator if denominator > 1.0e-12 else 0.0


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution requires finite values")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "maximum": float(np.max(array)),
    }


def _mirror_joint(name: str) -> str | None:
    if name.startswith("left_"):
        return "right_" + name.removeprefix("left_")
    return None


def _mirror_summary(rows: list[dict[str, Any]], gain_key: str) -> dict[str, Any]:
    by_key = {
        (row["selected_joint_name"], math.copysign(1.0, row["commanded_joint_delta_rad"])): row
        for row in rows
    }
    pairs = []
    for row in rows:
        right_name = _mirror_joint(row["selected_joint_name"])
        if right_name is None:
            continue
        sign = math.copysign(1.0, row["commanded_joint_delta_rad"])
        other = by_key.get((right_name, sign))
        if other is None:
            raise ValueError(f"missing mirror case for {row['step_case_id']}")
        pairs.append(
            {
                "left_case_id": row["step_case_id"],
                "right_case_id": other["step_case_id"],
                "symmetric_gain_difference": _symmetric_difference(
                    float(row[gain_key]), float(other[gain_key])
                ),
            }
        )
    return {
        "pair_count": len(pairs),
        "symmetric_gain_difference": _distribution(
            [item["symmetric_gain_difference"] for item in pairs]
        ),
        "pairs": pairs,
    }


def build_report(
    *,
    bundle_dir: Path,
    isaac_results_dir: Path,
    mujoco_results_dir: Path,
    mujoco_friction_zero_results_dir: Path | None,
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    bundle = _json(bundle_dir / "bundle_report.json")
    metadata = _metadata_index(bundle_dir)
    isaac = _result_index(isaac_results_dir)
    mujoco = _result_index(mujoco_results_dir)
    friction_zero = (
        _result_index(mujoco_friction_zero_results_dir)
        if mujoco_friction_zero_results_dir is not None
        else None
    )
    expected_ids = set(bundle["case_ids"])
    for label, index in (("Isaac", isaac), ("MuJoCo", mujoco)):
        if set(index) != expected_ids:
            raise ValueError(f"{label} result coverage differs from bundle")
    if friction_zero is not None and set(friction_zero) != expected_ids:
        raise ValueError("friction-zero result coverage differs from bundle")

    zero_by_joint: dict[str, str] = {}
    step_ids = []
    for case_id in bundle["case_ids"]:
        result = isaac[case_id][0]
        category = result["case_validation"]["category"]
        joint = result["runner_facts"]["selected_joint_name"]
        if category == "joint_zero_baseline":
            if joint in zero_by_joint:
                raise ValueError(f"multiple nominal zero baselines for {joint}")
            zero_by_joint[joint] = case_id
        elif category == "base_action_step":
            step_ids.append(case_id)
        else:
            raise ValueError(f"unexpected low-ZOH category: {category}")
    if len(zero_by_joint) != EXPECTED_ZERO_COUNT or len(step_ids) != EXPECTED_STEP_COUNT:
        raise ValueError("low-ZOH bundle must contain 14 zeros and 28 steps")

    rows = []
    official_differences = []
    ablated_differences = []
    source_paths: set[Path] = {bundle_dir / "bundle_report.json"}
    rationale = (
        "The no-contact v3 trace, mapping, command sign, and PD scale are shared; "
        "the diagnostic MuJoCo run changes only passive frictionloss scale."
    )
    for case_id in step_ids:
        joint = isaac[case_id][0]["runner_facts"]["selected_joint_name"]
        zero_id = zero_by_joint[joint]
        step_metadata, step_metadata_path = metadata[case_id]
        zero_metadata, zero_metadata_path = metadata[zero_id]
        kwargs = {
            "isaac_step_result": isaac[case_id][0],
            "isaac_step_evidence": _npz(isaac[case_id][1]),
            "isaac_zero_result": isaac[zero_id][0],
            "isaac_zero_evidence": _npz(isaac[zero_id][1]),
            "mujoco_step_result": mujoco[case_id][0],
            "mujoco_step_evidence": _npz(mujoco[case_id][1]),
            "mujoco_zero_result": mujoco[zero_id][0],
            "mujoco_zero_evidence": _npz(mujoco[zero_id][1]),
            "step_trace_metadata": step_metadata,
            "zero_trace_metadata": zero_metadata,
            "classification_color": "yellow",
            "difference_labels": ["expected_actuator_difference"],
            "rationale": rationale,
        }
        official = comparison.compare_step_with_zero_baselines(**kwargs)
        row = {
            "step_case_id": case_id,
            "zero_case_id": zero_id,
            "selected_joint_name": joint,
            "commanded_joint_delta_rad": official["commanded_joint_delta_rad"],
            "direction_consistent": official[
                "baseline_corrected_response_direction_consistent"
            ],
            "isaac_gain": official["isaac_baseline_corrected_gain"],
            "mujoco_official_gain": official["mujoco_baseline_corrected_gain"],
            "official_gain_symmetric_difference": official[
                "gain_symmetric_difference"
            ],
            "isaac_end_window_slope_radps": official[
                "isaac_response_end_window_slope_radps"
            ],
            "mujoco_official_end_window_slope_radps": official[
                "mujoco_response_end_window_slope_radps"
            ],
            "all_official_safety_envelopes_passed": official[
                "all_safety_envelopes_passed"
            ],
        }
        official_differences.append(float(official["gain_symmetric_difference"]))
        source_paths.update(
            {
                isaac[case_id][1], isaac[zero_id][1],
                isaac[case_id][2], isaac[zero_id][2],
                mujoco[case_id][1], mujoco[zero_id][1],
                mujoco[case_id][2], mujoco[zero_id][2],
                step_metadata_path, zero_metadata_path,
            }
        )
        if friction_zero is not None:
            kwargs.update(
                {
                    "mujoco_step_result": friction_zero[case_id][0],
                    "mujoco_step_evidence": _npz(friction_zero[case_id][1]),
                    "mujoco_zero_result": friction_zero[zero_id][0],
                    "mujoco_zero_evidence": _npz(friction_zero[zero_id][1]),
                }
            )
            ablated = comparison.compare_step_with_zero_baselines(**kwargs)
            row.update(
                {
                    "mujoco_friction_zero_gain": ablated[
                        "mujoco_baseline_corrected_gain"
                    ],
                    "friction_zero_gain_symmetric_difference": ablated[
                        "gain_symmetric_difference"
                    ],
                    "mujoco_friction_zero_end_window_slope_radps": ablated[
                        "mujoco_response_end_window_slope_radps"
                    ],
                    "friction_ablation_reduced_gain_difference": ablated[
                        "gain_symmetric_difference"
                    ] < official["gain_symmetric_difference"],
                }
            )
            ablated_differences.append(float(ablated["gain_symmetric_difference"]))
            source_paths.update(
                {
                    friction_zero[case_id][1], friction_zero[zero_id][1],
                    friction_zero[case_id][2], friction_zero[zero_id][2],
                }
            )
        rows.append(row)

    zero_metrics = {}
    for engine, index in (("isaac", isaac), ("mujoco_official", mujoco)):
        drifts = [
            abs(float(index[case_id][0]["metrics"]["peak_abs_drift_from_baseline_rad"]))
            for case_id in zero_by_joint.values()
        ]
        slopes = [
            abs(float(index[case_id][0]["metrics"]["end_window_slope_radps"]))
            for case_id in zero_by_joint.values()
        ]
        zero_metrics[engine] = {
            "peak_abs_drift_rad": _distribution(drifts),
            "abs_end_window_slope_radps": _distribution(slopes),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_status": "phase0_low_zoh_diagnostic_not_promotion_evidence",
        "scope": bundle["scope"],
        "matrix_sha256": bundle["matrix_sha256"],
        "case_counts": {"zero": len(zero_by_joint), "step": len(rows)},
        "all_step_directions_consistent": all(row["direction_consistent"] for row in rows),
        "all_safety_envelopes_passed": all(
            row["all_official_safety_envelopes_passed"] for row in rows
        ),
        "official_gain_symmetric_difference": _distribution(official_differences),
        "zero_baselines": zero_metrics,
        "left_right_mirror": {
            "isaac": _mirror_summary(rows, "isaac_gain"),
            "mujoco_official": _mirror_summary(rows, "mujoco_official_gain"),
        },
        "classification": {
            "low_zoh_primary_difference_label": "expected_actuator_difference",
            "primary_evidence": (
                "Removing only MuJoCo passive frictionloss collapses the cross-engine "
                "gain difference; PhysX and MuJoCo friction parameters have different units "
                "and load dependence."
            ),
            "low_zoh_primary_difference_classification_frozen": friction_zero is not None,
            "training_joint_friction_contract_frozen": False,
            "copy_mujoco_frictionloss_value_into_isaac_allowed": False,
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
    }
    if friction_zero is not None:
        report["friction_ablation"] = {
            "mujoco_passive_friction_scale": 0.0,
            "gain_symmetric_difference": _distribution(ablated_differences),
            "improved_case_count": sum(
                row["friction_ablation_reduced_gain_difference"] for row in rows
            ),
            "case_count": len(rows),
            "median_difference_reduction_fraction": float(
                1.0
                - np.median(np.asarray(ablated_differences))
                / max(np.median(np.asarray(official_differences)), 1.0e-12)
            ),
        }
        report["left_right_mirror"]["mujoco_friction_zero"] = _mirror_summary(
            rows, "mujoco_friction_zero_gain"
        )
    report["source_sha256"] = {
        str(path): contract.file_sha256(path) for path in sorted(source_paths)
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
        mujoco_friction_zero_results_dir=args.mujoco_friction_zero_results_dir,
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
                "case_counts": report["case_counts"],
                "official_gain_symmetric_difference": report[
                    "official_gain_symmetric_difference"
                ],
                "friction_ablation": report.get("friction_ablation"),
                "qualification_status": report["qualification_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
