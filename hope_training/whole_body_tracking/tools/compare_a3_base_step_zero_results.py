#!/usr/bin/env python3
"""Compare an Isaac/MuJoCo step pair using same-workpoint zero baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import a3_base_contract as contract
import a3_base_zero_baseline_comparison as comparison


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for engine in ("isaac", "mujoco"):
        for kind in ("step", "zero"):
            parser.add_argument(f"--{engine}-{kind}-result", type=Path, required=True)
            parser.add_argument(f"--{engine}-{kind}-evidence", type=Path, required=True)
    parser.add_argument("--step-trace-metadata", type=Path, required=True)
    parser.add_argument("--zero-trace-metadata", type=Path, required=True)
    parser.add_argument(
        "--classification-color",
        choices=sorted(comparison.CLASSIFICATION_COLORS),
        required=True,
    )
    parser.add_argument(
        "--difference-label",
        action="append",
        required=True,
        choices=sorted(comparison.ALLOWED_DIFFERENCE_LABELS),
    )
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _json(path: Path) -> dict:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def main() -> None:
    args = _parser().parse_args()
    kwargs = {}
    source_paths: dict[str, Path] = {}
    for engine in ("isaac", "mujoco"):
        for kind in ("step", "zero"):
            result_path = getattr(args, f"{engine}_{kind}_result")
            evidence_path = getattr(args, f"{engine}_{kind}_evidence")
            kwargs[f"{engine}_{kind}_result"] = _json(result_path)
            kwargs[f"{engine}_{kind}_evidence"] = _npz(evidence_path)
            source_paths[f"{engine}_{kind}_result"] = result_path.expanduser().resolve()
            source_paths[f"{engine}_{kind}_evidence"] = evidence_path.expanduser().resolve()
    kwargs["step_trace_metadata"] = _json(args.step_trace_metadata)
    kwargs["zero_trace_metadata"] = _json(args.zero_trace_metadata)
    kwargs["classification_color"] = args.classification_color
    kwargs["difference_labels"] = args.difference_label
    kwargs["rationale"] = args.rationale
    payload = comparison.compare_step_with_zero_baselines(**kwargs)
    source_paths["step_trace_metadata"] = args.step_trace_metadata.expanduser().resolve()
    source_paths["zero_trace_metadata"] = args.zero_trace_metadata.expanduser().resolve()
    payload["source_sha256"] = {
        name: contract.file_sha256(path) for name, path in source_paths.items()
    }
    payload["source_sha256"]["comparison_source"] = contract.file_sha256(
        Path(__file__).resolve()
    )
    output = args.output.expanduser().resolve()
    if output.suffix != ".json":
        raise ValueError("--output must end in .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
