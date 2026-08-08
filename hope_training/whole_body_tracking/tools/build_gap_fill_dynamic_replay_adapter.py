#!/usr/bin/env python3
"""Adapt the A3 FK bank to the existing floating-base P1 replay contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = json.loads((args.bank / "manifest.json").read_text(encoding="utf-8"))
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    motions = []
    for index, entry in enumerate(bank["motions"]):
        source = (args.bank / entry["motion_npz"]).resolve()
        target = out / f"motion_{index:02d}.npz"
        with np.load(source, allow_pickle=False) as data:
            body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
            root0 = body_pos[0, 0].copy()
            body_pos_b0 = body_pos - root0
            arrays = {
                "fps": np.asarray(data["fps"]),
                "joint_names": np.asarray(data["joint_names"]),
                "joint_pos": np.asarray(data["joint_pos"], dtype=np.float32),
                "joint_vel": np.asarray(data["joint_vel"], dtype=np.float32),
                "body_pos_b0": body_pos_b0,
                "body_quat_b0_wxyz": np.asarray(data["body_quat_w"], dtype=np.float32),
                "body_lin_vel_b0": np.asarray(data["body_lin_vel_w"], dtype=np.float32),
                "body_ang_vel_b0": np.asarray(data["body_ang_vel_w"], dtype=np.float32),
                "hit_frame": np.asarray(data["hit_frame"]),
            }
        np.savez_compressed(target, **arrays)
        motions.append({
            "motion_id": index,
            "motion_name": entry["motion_id"],
            "canonical_motion_npz": target.name,
            "hit_frame": int(entry["hit_event"]["motion_hit_frame"]),
            "source_goal_id": entry["source_goal_id"],
            "stroke_type": entry["stroke_type"],
        })
    payload = {
        "schema_version": "a3_gap_fill_dynamic_replay_adapter/v1",
        "status": "floating_base_replay_pending",
        "source_manifest": str((args.bank / "manifest.json").resolve()),
        "momentum_preview_contract": {"joint_names": motions and list(np.load(out / motions[0]["canonical_motion_npz"], allow_pickle=False)["joint_names"].astype(str))},
        "motions": motions,
    }
    (out / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "motions": len(motions)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
