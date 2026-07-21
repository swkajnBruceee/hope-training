#!/usr/bin/env python3
"""Run the frozen repeat-1 A3 Base fixture pilot in isolated native MuJoCo."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

import a3_base_calibration as calibration
import a3_base_command_trace as command_trace
import a3_base_contract as contract
import a3_base_mujoco_calibration as native


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-dir", type=Path, default=contract.contract_dir_from_script()
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mujoco-python-path")
    parser.add_argument(
        "--scope",
        choices=("repeat1_low_bundle", "repeat1_medium_steps"),
        default="repeat1_low_bundle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contracts = contract.load_contracts(args.contract_dir)
    contract.validate_contracts(contracts)
    matrix_path = args.matrix.expanduser().resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    calibration.validate_matrix(matrix, contracts)
    model_path = args.model.expanduser().resolve()
    expected_model_hash = contracts["command_composer_contract.json"][
        "source_assets"
    ]["official_mujoco_xml"]["sha256"]
    if contract.file_sha256(model_path) != expected_model_hash:
        raise ValueError("official MuJoCo model hash mismatch")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be fresh: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scope == "repeat1_low_bundle":
        selected = [
            case
            for case in matrix["cases"]
            if case["repeat_index"] == 0
            and (
                case["stage"] == "base_action_step_low_amplitude"
                or case["category"] in {"waist_pitch_residual", "target_transport"}
            )
        ]
    else:
        selected = [
            case
            for case in matrix["cases"]
            if case["repeat_index"] == 0
            and case["stage"] == "base_action_step_medium_amplitude"
        ]
    results = []
    stage_counts: Counter[str] = Counter()
    violations_by_stage: dict[str, list[str]] = defaultdict(list)
    for index, case in enumerate(selected, start=1):
        shared_trace, trace_metadata = command_trace.build_trace(
            case,
            contracts,
            float(case["inputs"].get("physics_rate_hz", 1000.0)),
        )
        result, trace = native.run_shared_trace(
            case,
            shared_trace,
            trace_metadata,
            contracts,
            str(model_path),
            args.mujoco_python_path,
        )
        validation = calibration.validate_case_result(
            result, case["category"], contracts
        )
        result["case_validation"] = validation
        result["matrix_sha256"] = matrix["matrix_sha256"]
        result["model_sha256"] = expected_model_hash
        result["contract_status"] = contracts["calibration_contract.json"][
            "native_mujoco_runner"
        ]["result_status"]
        case_dir = output_dir / f"{index:03d}_{case['case_id']}"
        case_dir.mkdir()
        (case_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        np.savez(case_dir / "trace.npz", **trace)
        results.append(result)
        stage_counts[case["stage"]] += 1
        violations_by_stage[case["stage"]].extend(validation["violations"])

    passed = sum(item["case_validation"]["safety_envelope_passed"] for item in results)
    report = {
        "schema_version": 1,
        "artifact_status": "phase0_fixture_pilot_not_promotion_evidence",
        "matrix_sha256": matrix["matrix_sha256"],
        "model_sha256": expected_model_hash,
        "contract_payload_sha256": matrix["contract_payload_sha256"],
        "runner_source_sha256": contract.file_sha256(Path(native.__file__).resolve()),
        "pilot_driver_source_sha256": contract.file_sha256(Path(__file__).resolve()),
        "selected_case_count": len(selected),
        "pilot_scope": args.scope,
        "full_matrix_case_count": matrix["case_count"],
        "matrix_coverage_complete": False,
        "repeat_coverage_complete": False,
        "safety_envelope_pass_count": int(passed),
        "safety_envelope_fail_count": len(results) - int(passed),
        "stage_counts": dict(sorted(stage_counts.items())),
        "violations_by_stage": dict(sorted(violations_by_stage.items())),
        "automatic_promotion": False,
        "free_base_stability_evidence": False,
        "results": results,
    }
    report_path = output_dir / "pilot_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "selected_case_count": len(selected),
                "pilot_scope": args.scope,
                "safety_envelope_pass_count": int(passed),
                "safety_envelope_fail_count": len(results) - int(passed),
                "matrix_coverage_complete": False,
                "automatic_promotion": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
