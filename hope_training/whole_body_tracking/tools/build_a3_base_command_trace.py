#!/usr/bin/env python3
"""Build one immutable cross-simulator A3 Base command trace."""

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
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--physics-rate-hz", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contracts = contract.load_contracts(args.contract_dir)
    contract.validate_contracts(contracts)
    matrix_path = args.matrix.expanduser().resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    calibration.validate_matrix(matrix, contracts)
    matches = [case for case in matrix["cases"] if case["case_id"] == args.case_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one matrix case named {args.case_id!r}")
    trace, metadata = command_trace.build_trace(
        matches[0], contracts, args.physics_rate_hz
    )
    command_trace.validate_trace(trace, metadata, contracts)
    metadata["matrix_sha256"] = matrix["matrix_sha256"]
    metadata["contract_payload_sha256"] = matrix["contract_payload_sha256"]
    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **trace)
    metadata_path = output.with_suffix(".trace.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metadata": str(metadata_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
