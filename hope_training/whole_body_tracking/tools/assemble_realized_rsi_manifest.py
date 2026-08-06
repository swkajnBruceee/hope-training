#!/usr/bin/env python3
"""Assemble the per-motion realized native rollout traces into a bank index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    entries = []
    for motion_id in range(8):
        case = cases[motion_id * 8]
        trace = args.directory / f"motion_{motion_id}_trace.npz"
        if not trace.exists():
            raise FileNotFoundError(trace)
        with np.load(trace, allow_pickle=False) as data:
            steps = np.asarray(data["motion_step"])
            wrap = np.flatnonzero(np.diff(steps) <= 0)
            keep = int(wrap[0] + 1) if wrap.size else int(len(steps))
            trimmed_name = f"realized_{motion_id}.npz"
            np.savez_compressed(args.directory / trimmed_name, **{key: np.asarray(data[key])[:keep] for key in data.files})
        entries.append({
            "episode_id": case["episode_id"],
            "state_file": trimmed_name,
            "source_case": case,
            "realized_state": True,
            "captured_frames": keep,
        })
    payload = {
        "schema_version": 1,
        "stage": "native_strike_realized_rsi_bank_v1",
        "training_eligible": False,
        "state_semantics": "actual_native_zero_residual_simulation_rollout",
        "captured_context": ["motion_step", "root_state_w", "joint_pos", "joint_vel", "joint_pos_target", "applied_torque", "time_to_strike_s"],
        "missing_context": ["actuator_hidden_state_not_exposed", "contact_sensor_not_configured", "ball_state_not_present_in_native_strike_task", "filter_state_not_present", "observation_history_not_captured"],
        "entries": entries,
    }
    out = args.directory / "realized_rsi_manifest.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"entries": len(entries), "output": str(out), "training_eligible": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
