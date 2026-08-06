#!/usr/bin/env python3
"""Apply one bounded ILC update to a copied A3 command trajectory.

The desired motion remains the source NPZ.  A previous fixed-base zero-action
rollout provides q_actual at every command frame.  This tool updates only the
position-command NPZ used by the servo:

    q_command_next = q_command + gain * (q_desired - q_actual)

It deliberately does not relabel the strike target, alter body-reference
arrays, or modify source files.  Output must be re-evaluated in the same
declared actuator contract before another update or any promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


NPZ_COLUMN_BY_NATIVE_JOINT = {
    "waist_yaw_joint": 2,
    "waist_roll_joint": 5,
    "waist_pitch_joint": 8,
    "right_shoulder_pitch_joint": 13,
    "right_shoulder_roll_joint": 18,
    "right_shoulder_yaw_joint": 22,
    "right_elbow_joint": 24,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 28,
    "right_wrist_yaw_joint": 30,
}


def _read_rollout_errors(path: Path, episode_id: str, frame_count: int) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    values: dict[str, np.ndarray] = {}
    for joint_name in NPZ_COLUMN_BY_NATIVE_JOINT:
        prefix = joint_name.replace("_joint", "")
        field = f"{prefix}_actual"
        series = np.full(frame_count, np.nan, dtype=np.float64)
        for row in rows:
            if row.get("mode") != "zero" or row.get("episode_id") != episode_id:
                continue
            if row.get("post_step_reset") == "True":
                continue
            step = int(row["command_motion_step"])
            if 0 <= step < frame_count and np.isnan(series[step]):
                series[step] = float(row[field])
        if np.isnan(series).any():
            missing = np.flatnonzero(np.isnan(series))
            # The diagnostic intentionally excludes the terminal post-step
            # reset.  A single missing final command frame therefore has no
            # measured response and must keep its existing command, rather
            # than blocking an otherwise complete ILC iteration.
            if np.array_equal(missing, np.asarray([frame_count - 1])):
                series[-1] = series[-2]
            else:
                raise ValueError(
                    f"{joint_name}: missing rollout q_actual at command frames {missing[:10].tolist()}"
                )
        values[joint_name] = series
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desired-npz", type=Path, required=True)
    parser.add_argument("--command-npz", type=Path, required=True)
    parser.add_argument("--timeseries-csv", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, help="Optional retained-target pilot manifest to copy and repoint.")
    parser.add_argument("--output-manifest", type=Path, help="Required with --pilot-manifest.")
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--max-waist-offset-rad", type=float, default=0.12)
    parser.add_argument("--max-arm-offset-rad", type=float, default=0.16)
    parser.add_argument("--smoothing-sigma-frames", type=float, default=1.0)
    args = parser.parse_args()

    if not 0.0 < args.gain <= 1.0:
        raise ValueError("--gain must be in (0, 1]")
    if args.smoothing_sigma_frames < 0.0:
        raise ValueError("--smoothing-sigma-frames must be non-negative")

    desired_path = args.desired_npz.expanduser().resolve()
    command_path = args.command_npz.expanduser().resolve()
    desired = np.load(desired_path, allow_pickle=False)
    command = np.load(command_path, allow_pickle=False)
    desired_q = np.asarray(desired["joint_pos"], dtype=np.float64)
    command_q = np.asarray(command["joint_pos"], dtype=np.float64)
    if desired_q.shape != command_q.shape or desired_q.ndim != 2 or desired_q.shape[1] != 31:
        raise ValueError("desired and command joint_pos must both be [T,31] with identical shapes")

    actual = _read_rollout_errors(args.timeseries_csv.expanduser().resolve(), args.episode_id, desired_q.shape[0])
    next_q = command_q.copy()
    report = {"episode_id": args.episode_id, "gain": args.gain, "joints": {}}
    for joint_name, column in NPZ_COLUMN_BY_NATIVE_JOINT.items():
        desired_series = desired_q[:, column]
        command_series = command_q[:, column]
        error = desired_series - actual[joint_name]
        update = args.gain * error
        if args.smoothing_sigma_frames > 0.0:
            update = gaussian_filter1d(update, sigma=args.smoothing_sigma_frames, mode="nearest")
        # Keep reset identical to the desired motion, and cap against the
        # desired trajectory rather than permitting cumulative command drift.
        update[0] = 0.0
        cap = args.max_waist_offset_rad if joint_name.startswith("waist_") else args.max_arm_offset_rad
        next_series = np.clip(command_series + update, desired_series - cap, desired_series + cap)
        next_q[:, column] = next_series
        report["joints"][joint_name] = {
            "q_actual_minus_desired_max_abs_rad": float(np.max(np.abs(actual[joint_name] - desired_series))),
            "command_delta_max_abs_rad": float(np.max(np.abs(next_series - desired_series))),
            "cap_rad": float(cap),
        }

    fps = int(np.asarray(command["fps"]).reshape(-1)[0])
    next_vel = np.gradient(next_q, axis=0) * float(fps)
    payload = {key: np.asarray(command[key]) for key in command.files}
    payload["joint_pos"] = next_q.astype(np.float32)
    payload["joint_vel"] = next_vel.astype(np.float32)
    output_npz = args.output_npz.expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    report_path = output_npz.with_suffix(".ilc_update.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[ilc] wrote {output_npz}")
    print(f"[ilc] wrote {report_path}")
    if (args.pilot_manifest is None) != (args.output_manifest is None):
        raise ValueError("--pilot-manifest and --output-manifest must be supplied together")
    if args.pilot_manifest is not None:
        pilot_manifest = args.pilot_manifest.expanduser().resolve()
        out_manifest = args.output_manifest.expanduser().resolve()
        data = json.loads(pilot_manifest.read_text(encoding="utf-8"))
        motions = list(data.get("motions", []))
        if len(motions) != 1 or str(motions[0].get("episode_id")) != args.episode_id:
            raise ValueError("pilot manifest must contain exactly the updated episode")
        motions[0]["motion_npz"] = str(output_npz)
        motions[0]["actuator_aware_pilot"]["status"] = "ilc_command_update_unvalidated"
        motions[0]["actuator_aware_pilot"]["ilc_report"] = str(report_path)
        data["motions"] = motions
        data["dataset_status"] = "actuator_aware_ilc_pilot_not_for_training"
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[ilc] wrote {out_manifest}")


if __name__ == "__main__":
    main()
