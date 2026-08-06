#!/usr/bin/env python3
"""Convert project-side RobotIOBackend replay CSV evidence to qualification NPZ."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _matrix(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    return np.asarray([[float(row[f"{prefix}_{joint}"]) for joint in range(31)] for row in rows], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-state-csv", type=Path, required=True)
    parser.add_argument("--command-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    states = list(csv.DictReader(args.raw_state_csv.open(encoding="utf-8")))
    commands = list(csv.DictReader(args.command_csv.open(encoding="utf-8")))
    if len(states) < 2 or len(commands) < 2:
        raise ValueError("replay must contain at least two state and command samples")
    raw_state_timestamp_s = np.asarray([float(row["state_data_ready_ns"]) / 1e9 for row in states], dtype=np.float64)
    backend_sync_timestamp_s = np.asarray([float(row["state_sync_ready_ns"]) / 1e9 for row in states], dtype=np.float64)
    command_timestamp_s = np.asarray([float(row["command_monotonic_ns"]) / 1e9 for row in commands], dtype=np.float64)
    if not (np.all(np.diff(raw_state_timestamp_s) > 0.0) and np.all(np.diff(backend_sync_timestamp_s) > 0.0) and np.all(np.diff(command_timestamp_s) > 0.0)):
        raise ValueError("replay clocks are not strictly increasing")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        raw_state_timestamp_s=raw_state_timestamp_s,
        backend_sync_timestamp_s=backend_sync_timestamp_s,
        command_timestamp_s=command_timestamp_s,
        q_actual=_matrix(states, "q"),
        dq_actual=_matrix(states, "dq"),
        tau_est=_matrix(states, "tau"),
    )
    print(output)


if __name__ == "__main__":
    main()
