#!/usr/bin/env python3
"""Create a diagnostic right-wrist-yaw variant of a canonical A3 command.

The input command is never modified.  The output is explicitly marked as a
local-SIL calibration artifact, so it cannot silently replace the immutable
source command or be promoted to training evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


JOINT_NAME = "right_wrist_yaw_joint"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raised_cosine_window(timestamps_s: np.ndarray, center_s: float, half_width_s: float) -> np.ndarray:
    """Return a zero-ended, unit-peak smooth window centred at ``center_s``."""

    phase = (timestamps_s - center_s) / half_width_s
    return np.where(np.abs(phase) < 1.0, 0.5 * (1.0 + np.cos(np.pi * phase)), 0.0)


def build_variant(
    input_path: Path, *, center_s: float, half_width_s: float, bias_rad: float
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(input_path, allow_pickle=False) as archive:
        required = {"timestamps_s", "q_des", "joint_names"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{input_path}: missing {', '.join(sorted(missing))}")
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
    timestamps = np.asarray(payload["timestamps_s"], dtype=np.float64)
    q_des = np.asarray(payload["q_des"], dtype=np.float64)
    names = [str(name) for name in np.asarray(payload["joint_names"]).tolist()]
    if timestamps.ndim != 1 or len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("timestamps_s must be strictly increasing")
    if q_des.shape != (len(timestamps), len(names)) or not np.all(np.isfinite(q_des)):
        raise ValueError("q_des must be finite [T, joint_names]")
    if names.count(JOINT_NAME) != 1:
        raise ValueError(f"command must contain exactly one {JOINT_NAME}")
    if half_width_s <= 0.0 or not np.isfinite(center_s) or not np.isfinite(bias_rad):
        raise ValueError("bias centre, half width, and amplitude must be finite; half width must be positive")
    index = names.index(JOINT_NAME)
    window = raised_cosine_window(timestamps, center_s, half_width_s)
    q_des[:, index] += bias_rad * window
    payload["timestamps_s"] = timestamps
    payload["q_des"] = q_des
    metadata: dict[str, object] = {
        "schema_version": 1,
        "artifact_status": "local_sil_diagnostic_only",
        "source_command": str(input_path),
        "source_command_sha256": sha256_file(input_path),
        "edited_joint": JOINT_NAME,
        "bias_rad": bias_rad,
        "center_s": center_s,
        "half_width_s": half_width_s,
        "peak_sample_index": int(np.argmax(window)),
        "peak_sample_time_s": float(timestamps[int(np.argmax(window))]),
        "note": "Smooth wrist-orientation experiment; not the immutable canonical command and not training data.",
    }
    return payload, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Immutable/source command NPZ to leave untouched.")
    parser.add_argument("--output", required=True, type=Path, help="New diagnostic command NPZ; must not already exist.")
    parser.add_argument("--center-s", required=True, type=float, help="Replay time of the peak wrist bias.")
    parser.add_argument("--half-width-s", type=float, default=0.22, help="Half duration of the raised-cosine bias window.")
    parser.add_argument("--bias-rad", required=True, type=float, help="Peak additive right-wrist-yaw bias in radians.")
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"--input does not exist: {input_path}")
    if output_path.exists():
        parser.error(f"--output must be new to preserve provenance: {output_path}")
    payload, metadata = build_variant(
        input_path, center_s=args.center_s, half_width_s=args.half_width_s, bias_rad=args.bias_rad
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    metadata["output_command"] = str(output_path)
    metadata["output_command_sha256"] = sha256_file(output_path)
    output_path.with_suffix(".command.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
