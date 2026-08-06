#!/usr/bin/env python3
"""Compare one aligned Isaac/MuJoCo A3 Base fixture result pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import a3_base_contract as contract
import a3_base_fixture_comparison as comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-result", type=Path, required=True)
    parser.add_argument("--isaac-evidence", type=Path, required=True)
    parser.add_argument("--mujoco-result", type=Path, required=True)
    parser.add_argument("--mujoco-evidence", type=Path, required=True)
    parser.add_argument("--trace-metadata", type=Path, required=True)
    parser.add_argument(
        "--difference-label",
        action="append",
        required=True,
        choices=sorted(comparison.ALLOWED_DIFFERENCE_LABELS),
    )
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def main() -> None:
    args = parse_args()
    isaac_result_path = args.isaac_result.expanduser().resolve()
    mujoco_result_path = args.mujoco_result.expanduser().resolve()
    isaac_evidence_path = args.isaac_evidence.expanduser().resolve()
    mujoco_evidence_path = args.mujoco_evidence.expanduser().resolve()
    metadata_path = args.trace_metadata.expanduser().resolve()
    isaac_result = json.loads(isaac_result_path.read_text(encoding="utf-8"))
    mujoco_result = json.loads(mujoco_result_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = comparison.compare_pair(
        isaac_result=isaac_result,
        isaac_evidence=_load_npz(isaac_evidence_path),
        mujoco_result=mujoco_result,
        mujoco_evidence=_load_npz(mujoco_evidence_path),
        trace_metadata=metadata,
        difference_labels=args.difference_label,
        rationale=args.rationale,
    )
    payload["source_sha256"] = {
        "isaac_result": contract.file_sha256(isaac_result_path),
        "isaac_evidence": contract.file_sha256(isaac_evidence_path),
        "mujoco_result": contract.file_sha256(mujoco_result_path),
        "mujoco_evidence": contract.file_sha256(mujoco_evidence_path),
        "trace_metadata": contract.file_sha256(metadata_path),
        "comparison_source": contract.file_sha256(Path(__file__).resolve()),
    }
    output = args.output.expanduser().resolve()
    if output.suffix != ".json":
        raise ValueError("--output must end in .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
