#!/usr/bin/env python3
"""Rebuild a prefix-bank index from completed NPZ files after simulator shutdown stalls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--frame-z-offset", type=float, required=True)
    parser.add_argument("--ground-z-offset", type=float, default=0.0)
    parser.add_argument("--task-id")
    parser.add_argument("--action-joint-names", help="Comma-separated override")
    args = parser.parse_args()

    template = json.loads(args.template.read_text(encoding="utf-8"))
    entries = []
    captured_context = None
    all_finite = True
    for path in sorted(args.bank.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if captured_context is None:
                captured_context = list(data.files)
            elif list(data.files) != captured_context:
                raise RuntimeError(f"Field mismatch in {path}")
            finite = all(
                np.isfinite(data[key]).all()
                for key in data.files
                if data[key].dtype.kind in "fc"
            )
            all_finite = all_finite and finite
            entries.append(
                {
                    "episode_id": path.stem,
                    "state_file": path.name,
                    "captured_frames": int(data["motion_step"].shape[0]),
                }
            )

    if not entries:
        raise RuntimeError(f"No NPZ files in {args.bank}")
    manifest = {
        **template,
        "manifest_frame_z_offset_m": float(args.frame_z_offset),
        "manifest_ground_z_offset_m": float(args.ground_z_offset),
        "all_values_finite": bool(all_finite),
        "continuous_prefix_handoff_candidate": bool(all_finite),
        "captured_context": captured_context,
        "entries": entries,
        "reconstructed_after_simulator_shutdown_stall": True,
    }
    if args.task_id:
        manifest["task_id"] = args.task_id
    if args.action_joint_names:
        manifest["action_joint_names"] = [name.strip() for name in args.action_joint_names.split(",")]
        manifest["action_dim"] = len(manifest["action_joint_names"])
    output = args.bank / "rsi_capture_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"entries": len(entries), "finite": all_finite, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
