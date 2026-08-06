#!/usr/bin/env python3
"""Convert the offline wrapper CSV into a compact 10-DOF upper-body NPZ.

This intentionally does not invent Isaac body states or manifest metadata.
Use the project's official CSV->NPZ builder after adapting the column map if
that richer format is required.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
JOINTS = WAIST + ARM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--goal", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("trajectory CSV is empty")

    q = np.asarray([[float(row[name]) for name in JOINTS] for row in rows], dtype=np.float64)
    dq = np.asarray(
        [[float(row[f"{name}_velocity"]) for name in JOINTS] for row in rows],
        dtype=np.float64,
    )
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    hit_flags = np.asarray([int(row["is_hit_frame"]) for row in rows], dtype=np.int8)
    hit_indices = np.flatnonzero(hit_flags)
    if len(hit_indices) != 1:
        raise ValueError(f"expected exactly one hit frame, found {len(hit_indices)}")
    if len(times) < 2:
        raise ValueError("trajectory must contain at least two samples")
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, atol=1e-9, rtol=1e-7):
        raise ValueError("trajectory timestamps are not uniform")

    goal = json.loads(args.goal.read_text(encoding="utf-8"))
    diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray("a3_ik_point_upper_reference/v2"),
        joint_names=np.asarray(JOINTS),
        joint_pos=q,
        joint_vel=dq,
        time_s=times,
        control_dt=np.asarray(dt, dtype=np.float64),
        hit_frame=np.asarray(int(hit_indices[0]), dtype=np.int64),
        canonical_position=np.asarray(goal["position_m"], dtype=np.float64),
        canonical_normal=np.asarray(goal["racket_normal"], dtype=np.float64),
        canonical_velocity=np.asarray(goal["linear_velocity_mps"], dtype=np.float64),
        requested_strike_time_s=np.asarray(goal["time_to_strike_s"], dtype=np.float64),
        planned_strike_time_s=np.asarray(
            diagnostics["planned_strike_time_s"], dtype=np.float64
        ),
        source_goal_id=np.asarray(goal["goal_id"]),
        source_generator=np.asarray("IkPointArmSource/offline-wrapper-v2"),
        requested_swing_type=np.asarray(diagnostics["requested_swing_type"]),
        selected_swing_type=np.asarray(diagnostics["selected_swing_type"]),
        ready_id=np.asarray(diagnostics["ready_id"]),
        ready_swing_type=np.asarray(diagnostics["ready_swing_type"]),
        candidate_status=np.asarray(diagnostics["status"]),
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
