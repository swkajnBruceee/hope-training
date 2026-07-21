#!/usr/bin/env python3
"""Build the deterministic A3 Base Phase 0 calibration experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import a3_base_calibration as calibration
import a3_base_contract as contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-dir", type=Path, default=contract.contract_dir_from_script()
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional output JSON; stdout emits a compact summary when set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contracts = contract.load_contracts(args.contract_dir)
    contract.validate_contracts(contracts)
    matrix = calibration.build_matrix(contracts)
    calibration.validate_matrix(matrix, contracts)
    rendered = json.dumps(matrix, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "case_count": matrix["case_count"],
                    "matrix_sha256": matrix["matrix_sha256"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
