#!/usr/bin/env python3
"""Audit local A3 ONNX assets against the deployment policy contract.

This is intentionally an interface audit only.  It never treats a model as
deployable based on DOF count alone: tensor names, dimensions, and the
configured role must all match the deployment contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_INPUT_NAME = "obs_dict"
EXPECTED_OUTPUT_NAME = "action"
EXPECTED_INPUT_DIM = 1570
EXPECTED_ACTION_DIM = 29


def _shape(value: Any) -> list[Any]:
    dims: list[Any] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def _flat_dim(shape: list[Any]) -> int | None:
    result = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            return None
        result *= dim
    return result


def inspect_model(path: Path, root: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    inputs = [
        {"name": value.name, "shape": _shape(value)}
        for value in model.graph.input
    ]
    outputs = [
        {"name": value.name, "shape": _shape(value)}
        for value in model.graph.output
    ]
    input_dim = _flat_dim(inputs[0]["shape"]) if len(inputs) == 1 else None
    action_dim = _flat_dim(outputs[0]["shape"]) if len(outputs) == 1 else None
    compatible = (
        len(inputs) == 1
        and len(outputs) == 1
        and inputs[0]["name"] == EXPECTED_INPUT_NAME
        and outputs[0]["name"] == EXPECTED_OUTPUT_NAME
        and input_dim == EXPECTED_INPUT_DIM
        and action_dim == EXPECTED_ACTION_DIM
    )
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "inputs": inputs,
        "outputs": outputs,
        "flat_input_dim": input_dim,
        "flat_action_dim": action_dim,
        "deployment_contract_compatible": compatible,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="A3 official package root; defaults to this repository's package.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    root = (
        args.root.expanduser().resolve()
        if args.root
        else Path(__file__).resolve().parents[4]
        / "third_party/aimsim_official/motion_control_humble"
    )
    if not root.exists():
        raise SystemExit(f"official package root does not exist: {root}")

    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.onnx")):
        try:
            rows.append(inspect_model(path, root))
        except Exception as exc:  # keep one bad asset from hiding the rest
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "error": f"{type(exc).__name__}: {exc}",
                    "deployment_contract_compatible": False,
                }
            )

    compatible = [row for row in rows if row.get("deployment_contract_compatible")]
    summary = {
        "contract": {
            "input_name": EXPECTED_INPUT_NAME,
            "output_name": EXPECTED_OUTPUT_NAME,
            "input_dim": EXPECTED_INPUT_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
        },
        "official_root": str(root),
        "model_count": len(rows),
        "compatible_count": len(compatible),
        "compatible_models": compatible,
        "models": rows,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
