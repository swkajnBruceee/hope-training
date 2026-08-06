#!/usr/bin/env python3
"""Produce bounded +/- smooth command perturbations for Phase 3A.

It does not run rollouts or rank candidates.  Every emitted command retains
the original immutable target and must be measured in standalone before a
dimension can enter any optimizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from a3_strike_contract import command_sha256_from_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--target-spec", type=Path, required=True)
    parser.add_argument("--joint-name", required=True)
    parser.add_argument("--center-s", type=float, required=True)
    parser.add_argument("--width-s", type=float, default=0.08)
    parser.add_argument("--epsilon-rad", type=float, default=0.02)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.width_s <= 0.0 or args.epsilon_rad <= 0.0:
        raise ValueError("width and epsilon must be positive")
    target = json.loads(args.target_spec.read_text(encoding="utf-8"))
    with np.load(args.command, allow_pickle=False) as data:
        names = [str(x) for x in data["joint_names"].tolist()]
        if args.joint_name not in names:
            raise ValueError(f"joint is not in command: {args.joint_name}")
        payload = {key: np.asarray(data[key]) for key in data.files}
    index = names.index(args.joint_name)
    time_s = np.asarray(payload["timestamps_s"], dtype=np.float64)
    envelope = np.exp(-0.5 * ((time_s - args.center_s) / args.width_s) ** 2)
    envelope[(time_s < args.center_s - 3.0 * args.width_s) | (time_s > args.center_s + 3.0 * args.width_s)] = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for sign, label in ((1.0, "plus"), (-1.0, "minus")):
        candidate = {key: value.copy() for key, value in payload.items()}
        candidate["q_des"][:, index] += sign * args.epsilon_rad * envelope
        output = args.out_dir / f"{args.joint_name}_{label}.npz"
        np.savez_compressed(output, **candidate)
        index_rows.append({
            "command": str(output), "command_sha256": command_sha256_from_npz(output),
            "joint_name": args.joint_name, "sign": label, "epsilon_rad": args.epsilon_rad,
            "center_s": args.center_s, "width_s": args.width_s,
            "source_target_sha256": target.get("source_target_sha256"),
            "status": "requires_official_standalone_rollout",
        })
    (args.out_dir / "sensitivity_index.json").write_text(json.dumps({"candidates": index_rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
