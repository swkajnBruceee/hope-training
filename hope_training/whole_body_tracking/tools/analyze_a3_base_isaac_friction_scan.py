#!/usr/bin/env python3
"""Analyze bounded PhysX joint-friction scans against official MuJoCo results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import a3_base_contract as contract


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--official-mujoco-results-dir", type=Path, required=True)
    parser.add_argument(
        "--isaac-scan",
        action="append",
        required=True,
        metavar="COEFFICIENT=RESULT_DIR",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _index(directory: Path) -> dict[str, tuple[dict[str, Any], Path, Path]]:
    result = {}
    for path in sorted(directory.resolve().glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_id = payload.get("case_id")
        if not isinstance(case_id, str):
            continue
        evidence = path.with_suffix(".trace.npz")
        if not evidence.is_file() or case_id in result:
            raise ValueError(f"invalid result/evidence pair: {path}")
        result[case_id] = (payload, evidence, path)
    return result


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _symmetric_difference(first: float, second: float) -> float:
    denominator = 0.5 * (abs(first) + abs(second))
    return abs(first - second) / denominator if denominator > 1.0e-12 else 0.0


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("distribution requires finite values")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "maximum": float(np.max(array)),
    }


def _gain(
    step_result: dict[str, Any],
    step_evidence_path: Path,
    zero_evidence_path: Path,
    metric_window: dict[str, Any],
) -> float:
    step = _arrays(step_evidence_path)
    zero = _arrays(zero_evidence_path)
    time_s = np.asarray(step["time_s"], dtype=np.float64)
    if not np.array_equal(time_s, np.asarray(zero["time_s"], dtype=np.float64)):
        raise ValueError("step and zero timestamps differ")
    active_end = float(metric_window["active_end_s"])
    active_start = float(metric_window["active_start_s"])
    end_window = float(metric_window["end_window_s"])
    tail = (time_s > max(active_start, active_end - end_window)) & (
        time_s <= active_end
    )
    response = np.asarray(step["joint_q_rad"]) - np.asarray(zero["joint_q_rad"])
    command = float(step_result["metrics"]["commanded_joint_delta_rad"])
    if command == 0.0 or not np.any(tail):
        raise ValueError("invalid step command or metric window")
    return float(np.mean(response[tail]) / command)


def build_report(
    *,
    bundle_dir: Path,
    official_mujoco_results_dir: Path,
    scans: dict[float, Path],
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    bundle = json.loads((bundle_dir / "bundle_report.json").read_text(encoding="utf-8"))
    expected = set(bundle["case_ids"])
    mujoco = _index(official_mujoco_results_dir)
    scan_indices = {value: _index(path) for value, path in scans.items()}
    if 0.0 not in scan_indices:
        raise ValueError("scan must include coefficient 0")
    for coefficient, index in scan_indices.items():
        if set(index) != expected:
            raise ValueError(f"Isaac coefficient {coefficient} coverage mismatch")
    if not expected <= set(mujoco):
        raise ValueError("official MuJoCo results do not cover diagnostic bundle")

    metadata = {}
    metadata_paths = {}
    for path in (bundle_dir / "traces").glob("*.trace.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata[payload["case_id"]] = payload
        metadata_paths[payload["case_id"]] = path
    baseline = scan_indices[0.0]
    zeros = {
        value[0]["runner_facts"]["selected_joint_name"]: case_id
        for case_id, value in baseline.items()
        if value[0]["case_validation"]["category"] == "joint_zero_baseline"
    }
    step_ids = [
        case_id
        for case_id, value in baseline.items()
        if value[0]["case_validation"]["category"] == "base_action_step"
    ]
    source_paths: set[Path] = {bundle_dir / "bundle_report.json"}
    rows = []
    summaries = []
    for coefficient in sorted(scan_indices):
        index = scan_indices[coefficient]
        gain_differences = []
        trajectory_changes = []
        cases = []
        for case_id in sorted(step_ids):
            joint = index[case_id][0]["runner_facts"]["selected_joint_name"]
            zero_id = zeros[joint]
            window = metadata[case_id]["metric_window"]
            isaac_gain = _gain(
                index[case_id][0], index[case_id][1], index[zero_id][1], window
            )
            mujoco_gain = _gain(
                mujoco[case_id][0], mujoco[case_id][1], mujoco[zero_id][1], window
            )
            base_step = _arrays(baseline[case_id][1])["joint_q_rad"]
            base_zero = _arrays(baseline[zero_id][1])["joint_q_rad"]
            scan_step = _arrays(index[case_id][1])["joint_q_rad"]
            scan_zero = _arrays(index[zero_id][1])["joint_q_rad"]
            trajectory_change = float(
                np.max(np.abs((scan_step - scan_zero) - (base_step - base_zero)))
            )
            gain_difference = _symmetric_difference(isaac_gain, mujoco_gain)
            gain_differences.append(gain_difference)
            trajectory_changes.append(trajectory_change)
            cases.append(
                {
                    "case_id": case_id,
                    "selected_joint_name": joint,
                    "isaac_gain": isaac_gain,
                    "mujoco_official_gain": mujoco_gain,
                    "gain_symmetric_difference": gain_difference,
                    "max_response_trajectory_change_from_coefficient_zero_rad": (
                        trajectory_change
                    ),
                }
            )
            source_paths.update(
                {
                    index[case_id][1], index[zero_id][1],
                    index[case_id][2], index[zero_id][2],
                    mujoco[case_id][1], mujoco[zero_id][1],
                    mujoco[case_id][2], mujoco[zero_id][2],
                    metadata_paths[case_id], metadata_paths[zero_id],
                }
            )
        active_values = {
            float(item[0]["runner_facts"]["active_selected_joint_friction_coefficient"])
            for item in index.values()
        }
        if len(active_values) != 1 or not math.isclose(
            next(iter(active_values)), coefficient, rel_tol=1.0e-6, abs_tol=1.0e-9
        ):
            raise ValueError("recorded PhysX coefficient differs from requested scan")
        summary = {
            "coefficient": coefficient,
            "case_count": len(cases),
            "safety_envelope_pass_count": sum(
                bool(index[case_id][0]["case_validation"]["safety_envelope_passed"])
                for case_id in index
            ),
            "gain_symmetric_difference_to_official_mujoco": _distribution(
                gain_differences
            ),
            "max_response_trajectory_change_from_coefficient_zero_rad": _distribution(
                trajectory_changes
            ),
            "cases": cases,
        }
        summaries.append(summary)
        rows.extend(cases)

    zero_median = next(
        item["gain_symmetric_difference_to_official_mujoco"]["median"]
        for item in summaries
        if item["coefficient"] == 0.0
    )
    best = min(
        summaries,
        key=lambda item: item["gain_symmetric_difference_to_official_mujoco"]["median"],
    )
    report = {
        "schema_version": 1,
        "artifact_status": "phase0_joint_friction_diagnostic_not_promotion_evidence",
        "scope": bundle["scope"],
        "matrix_sha256": bundle["matrix_sha256"],
        "physx_friction_semantics": (
            "unitless coefficient; resisting force is bounded by coefficient times "
            "the transmitted parent-child spatial force"
        ),
        "mujoco_frictionloss_semantics": "fixed generalized passive friction force",
        "direct_numeric_mapping_allowed": False,
        "scan": summaries,
        "diagnostic_conclusion": {
            "best_scanned_coefficient": best["coefficient"],
            "baseline_median_gain_symmetric_difference": zero_median,
            "best_median_gain_symmetric_difference": best[
                "gain_symmetric_difference_to_official_mujoco"
            ]["median"],
            "single_global_physx_coefficient_reproduces_official_mujoco_frictionloss": False,
            "reason": (
                "The bounded scan changes transient trajectories but does not reproduce "
                "the joint-dependent low-amplitude attenuation caused by fixed MuJoCo "
                "frictionloss. Further coefficient fitting is stopped."
            ),
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
    scans = {}
    for item in args.isaac_scan:
        coefficient_text, separator, directory = item.partition("=")
        if not separator:
            raise ValueError("--isaac-scan must be COEFFICIENT=RESULT_DIR")
        coefficient = float(coefficient_text)
        if not math.isfinite(coefficient) or coefficient < 0.0 or coefficient in scans:
            raise ValueError("scan coefficients must be unique, finite, and non-negative")
        scans[coefficient] = Path(directory).expanduser().resolve()
    report = build_report(
        bundle_dir=args.bundle_dir,
        official_mujoco_results_dir=args.official_mujoco_results_dir,
        scans=scans,
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
                "scan": [
                    {
                        "coefficient": item["coefficient"],
                        "median_gain_symmetric_difference": item[
                            "gain_symmetric_difference_to_official_mujoco"
                        ]["median"],
                        "maximum_trajectory_change_rad": item[
                            "max_response_trajectory_change_from_coefficient_zero_rad"
                        ]["maximum"],
                    }
                    for item in report["scan"]
                ],
                "diagnostic_conclusion": report["diagnostic_conclusion"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
