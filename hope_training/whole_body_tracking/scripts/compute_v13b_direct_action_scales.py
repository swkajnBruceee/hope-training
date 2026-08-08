#!/usr/bin/env python3
"""Compute offline direct-action envelopes for V1.3B.

This script is deliberately offline: it may read reviewed reference data to
estimate a safe joint envelope, but the resulting YAML is a fixed runtime
artifact and the V1.3B environment never loads a motion/reference file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NPZ with q_joint or joint_pos [N,22]")
    parser.add_argument("--ready", required=True, help="READY vector [22] as .npy or comma-separated values")
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin", type=float, default=1.15)
    args = parser.parse_args()
    data = np.load(args.input)
    for key in ("q_joint", "joint_pos", "joint_positions"):
        if key in data:
            q = np.asarray(data[key], dtype=np.float64)
            break
    else:
        raise SystemExit("input NPZ must contain q_joint, joint_pos, or joint_positions")
    if q.ndim != 2 or q.shape[1] < 22:
        raise SystemExit(f"expected [N,22+] joint array, got {q.shape}")
    ready_path = Path(args.ready)
    if ready_path.exists():
        ready = np.asarray(np.load(ready_path), dtype=np.float64).reshape(-1)
    else:
        ready = np.asarray([float(x) for x in args.ready.split(",")], dtype=np.float64)
    if ready.size != 22:
        raise SystemExit(f"READY must contain 22 values, got {ready.size}")
    delta = np.abs(q[:, :22] - ready[None, :])
    scale = np.maximum(1.0e-3, float(args.margin) * np.percentile(delta, 99.5, axis=0))
    payload = {
        "contract_version": "v13b_direct_action_scale_v1",
        "status": "generated_offline_envelope_pending_physx_qualification",
        "source_is_offline_only": True,
        "joint_order": "A3_REFERENCE_TRACKER_JOINTS (legs12 + waist3 + right_arm7)",
        "scale_rad": [float(x) for x in scale],
        "lower_scale_rad": [float(x) for x in scale[:12]],
        "upper_scale_rad": [float(x) for x in scale[12:22]],
        "percentile": 99.5,
        "margin": float(args.margin),
    }
    Path(args.output).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.output}; max_scale={scale.max():.6f} rad")


if __name__ == "__main__":
    main()
