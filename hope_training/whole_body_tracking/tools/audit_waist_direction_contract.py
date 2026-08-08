#!/usr/bin/env python3
"""Audit A3 waist pitch direction: forward-only, never backward tilt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MAX_FORWARD_DEG = 20.0
TOLERANCE_DEG = 1.0e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    all_rows = []
    for manifest_arg in args.manifests:
        manifest_path = manifest_arg.expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("motions", [])
        rows = []
        for entry in entries:
            motion = Path(str(entry["motion_npz"])).expanduser()
            if not motion.is_absolute():
                motion = manifest_path.parent / motion
            with np.load(motion, allow_pickle=False) as data:
                names = [str(x) for x in np.asarray(data["joint_names"]).reshape(-1)]
                q = np.asarray(data["joint_pos"], dtype=np.float64)
            ip = names.index("waist_pitch_joint")
            ir = names.index("waist_roll_joint")
            iy = names.index("waist_yaw_joint")
            pitch = np.degrees(q[:, ip])
            roll = np.degrees(q[:, ir])
            yaw = np.degrees(q[:, iy])
            backward = float(pitch.min()) < -TOLERANCE_DEG
            over_forward = float(pitch.max()) > MAX_FORWARD_DEG + TOLERANCE_DEG
            over_roll = float(np.max(np.abs(roll))) > MAX_FORWARD_DEG + TOLERANCE_DEG
            row = {
                "episode_id": entry.get("episode_id"),
                "waist_pitch_min_deg": float(pitch.min()),
                "waist_pitch_max_deg": float(pitch.max()),
                "waist_roll_abs_max_deg": float(np.max(np.abs(roll))),
                "waist_yaw_min_deg": float(yaw.min()),
                "waist_yaw_max_deg": float(yaw.max()),
                "backward_tilt_detected": backward,
                "forward_tilt_over_20deg": over_forward,
                "roll_over_20deg": over_roll,
                "status": "PASS" if not (backward or over_forward or over_roll) else "REJECT",
            }
            rows.append(row)
            all_rows.append({"manifest": str(manifest_path), **row})
        reports.append(
            {
                "manifest": str(manifest_path),
                "count": len(rows),
                "backward_tilt_count": sum(r["backward_tilt_detected"] for r in rows),
                "forward_over_20deg_count": sum(r["forward_tilt_over_20deg"] for r in rows),
                "roll_over_20deg_count": sum(r["roll_over_20deg"] for r in rows),
                "pitch_range_deg": [min(r["waist_pitch_min_deg"] for r in rows), max(r["waist_pitch_max_deg"] for r in rows)] if rows else None,
            }
        )
    report = {
        "schema_version": "a3_waist_direction_contract_audit/v1",
        "contract": {
            "waist_pitch": "forward_only_nonnegative_joint_pitch",
            "backward_tilt_allowed": False,
            "maximum_forward_tilt_deg": MAX_FORWARD_DEG,
            "maximum_roll_abs_deg": MAX_FORWARD_DEG,
        },
        "status": "PASS" if all(r["backward_tilt_count"] == 0 and r["forward_over_20deg_count"] == 0 and r["roll_over_20deg_count"] == 0 for r in reports) else "FAIL",
        "manifests": reports,
        "rows": all_rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "manifests": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
