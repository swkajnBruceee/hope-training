#!/usr/bin/env python3
"""Compose a reference A3 31-DOF command from Base and Strike inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import a3_base_contract as contract


def _json_value(raw: str) -> Any:
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=contract.contract_dir_from_script())
    parser.add_argument("--base-action", required=True, help="JSON array or path to JSON")
    parser.add_argument("--strike-q-reference", required=True, help="JSON array or path to JSON")
    parser.add_argument("--output", type=Path, help="optional output JSON; stdout is always emitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contracts = contract.load_contracts(args.contract_dir)
    contract.validate_contracts(contracts)
    command = contract.compose_command(
        contracts["command_composer_contract.json"],
        _json_value(args.base_action),
        _json_value(args.strike_q_reference),
    )
    rendered = json.dumps(command, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

