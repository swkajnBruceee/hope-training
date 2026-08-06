#!/usr/bin/env python3
"""Validate the Phase 0 A3 Base/Strike/Composer contract fail-closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import a3_base_contract as contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=contract.contract_dir_from_script())
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--skip-source-assets", action="store_true", help="only validate JSON and golden vectors")
    parser.add_argument(
        "--require-training-approved",
        action="store_true",
        help="fail until calibration blockers are resolved and approval flags are true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contracts = contract.load_contracts(args.contract_dir)
    summary = contract.validate_contracts(contracts)
    summary["golden_cases"] = contract.validate_golden_vectors(contracts)
    if not args.skip_source_assets:
        summary["asset_audit"] = contract.validate_source_assets(
            contracts["command_composer_contract.json"], args.repo_root
        )
    summary["structural_validation_passed"] = True
    summary["training_gate_passed"] = bool(summary["training_approved"])
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_training_approved and not summary["training_approved"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

