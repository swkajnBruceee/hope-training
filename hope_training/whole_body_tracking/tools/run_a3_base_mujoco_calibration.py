#!/usr/bin/env python3
"""Run one frozen A3 Base calibration case in an isolated MuJoCo process."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mujoco-python-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contracts = contract.load_contracts(args.contract_dir)
    contract.validate_contracts(contracts)
    matrix_path = args.matrix.expanduser().resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    calibration.validate_matrix(matrix, contracts)
    trace_metadata = json.loads(
        args.trace_metadata.expanduser().resolve().read_text(encoding="utf-8")
    )
    cases = [
        case
        for case in matrix["cases"]
        if case["case_id"] == trace_metadata.get("case_id")
    ]
    if len(cases) != 1:
        raise ValueError("trace metadata does not select exactly one matrix case")
    with np.load(args.trace.expanduser().resolve(), allow_pickle=False) as archive:
        shared_trace = {name: archive[name] for name in command_trace.ARRAY_ORDER}
    command_trace.validate_trace(shared_trace, trace_metadata, contracts)
    model_path = args.model.expanduser().resolve()
    expected_model = contracts["command_composer_contract.json"]["source_assets"][
        "official_mujoco_xml"
    ]["sha256"]
    if contract.file_sha256(model_path) != expected_model:
        raise ValueError("official MuJoCo model hash mismatch")
    result, trace = native.run_shared_trace(
        cases[0],
        shared_trace,
        trace_metadata,
        contracts,
        str(model_path),
        args.mujoco_python_path,
    )
    case_validation = calibration.validate_case_result(
        result, cases[0]["category"], contracts
    )
    result["matrix_sha256"] = matrix["matrix_sha256"]
    result["model_sha256"] = expected_model
    result["runner_source_sha256"] = contract.file_sha256(
        Path(native.__file__).resolve()
    )
    result["contract_status"] = contracts["calibration_contract.json"][
        "native_mujoco_runner"
    ]["result_status"]
    result["case_validation"] = case_validation
    output = args.output.expanduser().resolve()
    if output.suffix != ".json":
        raise ValueError("--output must end in .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    trace_path = output.with_suffix(".trace.npz")
    np.savez(trace_path, **trace)
    rendered = {
        "output": str(output),
        "trace": str(trace_path),
        "case_id": result["case_id"],
        "metrics": result["metrics"],
        "runner_facts": result["runner_facts"],
        "case_validation": case_validation,
    }
    print(json.dumps(rendered, indent=2))


if __name__ == "__main__":
    main()
