#!/usr/bin/env python3
"""Build a read-only trace bundle for bounded A3 fixture pilot scopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import a3_base_calibration as calibration
import a3_base_command_trace as command_trace
import a3_base_contract as contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-dir", type=Path, default=contract.contract_dir_from_script()
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=(
            "low_zoh",
            "representative_medium",
            "waist_working_point",
            "transport_200hz",
            "friction_diagnostic",
            "stand_fixture_approval",
            "low_zoh_repeat1",
            "representative_medium_repeat1",
            "waist_working_point_repeat1",
            "transport_200hz_repeat1",
            "friction_diagnostic_repeat1",
        ),
        required=True,
    )
    parser.add_argument(
        "--repeat-number",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="One-based deterministic repeat to extract from the frozen matrix.",
    )
    return parser.parse_args()


def _select(matrix: dict, scope: str, repeat_number: int = 1) -> list[dict]:
    if scope.endswith("_repeat1"):
        if repeat_number != 1:
            raise ValueError("legacy *_repeat1 scope aliases require --repeat-number 1")
        scope = scope.removesuffix("_repeat1")
    cases = [
        case for case in matrix["cases"]
        if case["repeat_index"] == repeat_number - 1
    ]
    if scope == "low_zoh":
        return [
            case
            for case in cases
            if case["case_family"] in {"action_zero", "action_low"}
        ]
    if scope == "stand_fixture_approval":
        representative_medium = {
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        }
        return [
            case
            for case in cases
            if case["case_family"] in {"action_zero", "action_low"}
            or (
                case["case_family"] == "action_medium"
                and case["inputs"]["selected_joint_name"] in representative_medium
            )
            or case["case_family"] in {
                "working_point_zero",
                "working_point",
                "waist_composition",
            }
            or (
                case["case_family"] == "transport"
                and float(case["inputs"]["physics_rate_hz"]) == 200.0
            )
        ]
    if scope == "friction_diagnostic":
        diagnostic_joints = {
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_roll_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        }
        return [
            case
            for case in cases
            if case["case_family"] in {"action_zero", "action_low"}
            and case["inputs"]["selected_joint_name"] in diagnostic_joints
        ]
    representative = {
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    }
    if scope == "representative_medium":
        return [
            case
            for case in cases
            if case["case_family"] == "action_medium"
            and case["inputs"]["selected_joint_name"] in representative
        ]
    if scope == "waist_working_point":
        return [
            case
            for case in cases
            if case["case_family"] in {
                "working_point_zero",
                "working_point",
                "waist_composition",
            }
        ]
    if scope == "transport_200hz":
        return [
            case
            for case in cases
            if case["case_family"] == "transport"
            and float(case["inputs"]["physics_rate_hz"]) == 200.0
        ]
    raise ValueError(f"unknown scope {scope}")


def main() -> None:
    args = parse_args()
    contracts = contract.load_contracts(args.contract_dir)
    contract.validate_contracts(contracts)
    matrix_path = args.matrix.expanduser().resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    calibration.validate_matrix(matrix, contracts)
    selected = _select(matrix, args.scope, args.repeat_number)
    if not selected:
        raise ValueError("fixture trace bundle scope selected no cases")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be fresh: {output_dir}")
    trace_dir = output_dir / "traces"
    isaac_result_dir = output_dir / "isaac_results"
    trace_dir.mkdir(parents=True, exist_ok=True)
    isaac_result_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    trace_hashes = {}
    for index, case in enumerate(selected, start=1):
        rate = float(case["inputs"].get("physics_rate_hz", 200.0))
        trace, metadata = command_trace.build_trace(case, contracts, rate)
        command_trace.validate_trace(trace, metadata, contracts)
        metadata["matrix_sha256"] = matrix["matrix_sha256"]
        metadata["contract_payload_sha256"] = matrix["contract_payload_sha256"]
        stem = f"{index:03d}_{case['case_id']}"
        trace_path = trace_dir / f"{stem}.npz"
        metadata_path = trace_dir / f"{stem}.trace.json"
        output_path = isaac_result_dir / f"{stem}.json"
        np.savez(trace_path, **trace)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "trace": str(trace_path),
                "trace_metadata": str(metadata_path),
                "output": str(output_path),
            }
        )
        trace_hashes[case["case_id"]] = metadata["trace_sha256"]
    manifest_path = output_dir / "trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "artifact_status": "fixture_trace_bundle_not_execution_or_promotion_evidence",
        "scope": (
            args.scope
            if args.scope.endswith("_repeat1")
            else f"{args.scope}_repeat{args.repeat_number}"
        ),
        "repeat_number": args.repeat_number,
        "matrix_sha256": matrix["matrix_sha256"],
        "case_count": len(selected),
        "case_ids": [case["case_id"] for case in selected],
        "trace_sha256_by_case": trace_hashes,
        "fixture_runner_qualified": True,
        "fixture_matrix_approved": False,
        "stand_task_approved": False,
        "automatic_promotion": False,
    }
    report_path = output_dir / "bundle_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "report": str(report_path),
                "scope": report["scope"],
                "case_count": len(selected),
                "matrix_sha256": matrix["matrix_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
