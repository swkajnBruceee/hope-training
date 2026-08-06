#!/usr/bin/env python3
"""Validate an A3 Base Phase 0 calibration result artifact fail-closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import a3_base_calibration as calibration
import a3_base_contract as contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--contract-dir", type=Path, default=contract.contract_dir_from_script()
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contracts = contract.load_contracts(args.contract_dir)
    contract.validate_contracts(contracts)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    result = calibration.validate_result_artifact(artifact, matrix, contracts)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["safety_envelope_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

