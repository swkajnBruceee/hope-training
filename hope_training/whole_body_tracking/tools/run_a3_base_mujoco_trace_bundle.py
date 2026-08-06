#!/usr/bin/env python3
"""Consume a pre-generated A3 fixture trace manifest in one MuJoCo process."""

from __future__ import annotations

import argparse
from collections import Counter
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
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mujoco-python-path")
    parser.add_argument(
        "--diagnostic-passive-friction-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--diagnostic-passive-damping-scale", type=float, default=1.0
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contracts = contract.load_contracts(args.contract_dir)
    contract.validate_contracts(contracts)
    matrix = json.loads(args.matrix.expanduser().resolve().read_text(encoding="utf-8"))
    calibration.validate_matrix(matrix, contracts)
    cases_by_id = {case["case_id"]: case for case in matrix["cases"]}
    manifest = json.loads(
        args.trace_manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("trace manifest must be a non-empty list")
    model_path = args.model.expanduser().resolve()
    expected_model_hash = contracts["command_composer_contract.json"]["source_assets"][
        "official_mujoco_xml"
    ]["sha256"]
    if contract.file_sha256(model_path) != expected_model_hash:
        raise ValueError("official MuJoCo model hash mismatch")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be fresh: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    categories: Counter[str] = Counter()
    for index, item in enumerate(manifest, start=1):
        trace_path = Path(item["trace"]).expanduser().resolve()
        metadata_path = Path(item["trace_metadata"]).expanduser().resolve()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        case = cases_by_id.get(metadata.get("case_id"))
        if case is None:
            raise ValueError("trace manifest references an unknown case")
        with np.load(trace_path, allow_pickle=False) as archive:
            shared_trace = {name: archive[name] for name in command_trace.ARRAY_ORDER}
        command_trace.validate_trace(shared_trace, metadata, contracts)
        result, evidence = native.run_shared_trace(
            case,
            shared_trace,
            metadata,
            contracts,
            str(model_path),
            args.mujoco_python_path,
            args.diagnostic_passive_friction_scale,
            args.diagnostic_passive_damping_scale,
        )
        validation = calibration.validate_case_result(
            result, case["category"], contracts
        )
        result["matrix_sha256"] = matrix["matrix_sha256"]
        result["model_sha256"] = expected_model_hash
        result["contract_status"] = contracts["calibration_contract.json"][
            "native_mujoco_runner"
        ]["result_status"]
        result["case_validation"] = validation
        stem = f"{index:03d}_{case['case_id']}"
        result_path = output_dir / f"{stem}.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        np.savez(output_dir / f"{stem}.trace.npz", **evidence)
        results.append(result)
        categories[case["category"]] += 1
    passed = sum(
        bool(result["case_validation"]["safety_envelope_passed"])
        for result in results
    )
    report = {
        "schema_version": 1,
        "artifact_status": "phase0_fixture_bundle_not_promotion_evidence",
        "matrix_sha256": matrix["matrix_sha256"],
        "model_sha256": expected_model_hash,
        "trace_manifest_sha256": contract.file_sha256(
            args.trace_manifest.expanduser().resolve()
        ),
        "diagnostic_model_overrides": {
            "passive_friction_scale": args.diagnostic_passive_friction_scale,
            "passive_damping_scale": args.diagnostic_passive_damping_scale,
        },
        "case_count": len(results),
        "category_counts": dict(sorted(categories.items())),
        "safety_envelope_pass_count": int(passed),
        "safety_envelope_fail_count": len(results) - int(passed),
        "fixture_matrix_approved": False,
        "stand_task_approved": False,
        "automatic_promotion": False,
        "results": results,
    }
    report_path = output_dir / "bundle_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "case_count": len(results),
                "safety_envelope_pass_count": int(passed),
                "safety_envelope_fail_count": len(results) - int(passed),
                "automatic_promotion": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
