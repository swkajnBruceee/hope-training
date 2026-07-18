#!/usr/bin/env python3
"""Summarize a non-publishing A3 ONNX standing-policy dump.

The binary layout is defined by a3_deploy_onnx_ref.  This tool has no robot or
network access; it only reads the dump emitted by --probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OBS_FLOATS = 1570
ACTION_FLOATS = 29
QDES_FLOATS = 29
JOINT_POS_FLOATS = 29
JOINT_VEL_FLOATS = 29
VEC3_FLOATS = 3
RECORD_BYTES = (
    (OBS_FLOATS + ACTION_FLOATS) * np.dtype(np.float32).itemsize
    + (QDES_FLOATS + JOINT_POS_FLOATS + JOINT_VEL_FLOATS + 2 * VEC3_FLOATS)
    * np.dtype(np.float64).itemsize
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.dump.read_bytes()
    if not raw or len(raw) % RECORD_BYTES:
        raise SystemExit(
            f"{args.dump}: {len(raw)} bytes is not a non-empty whole number "
            f"of {RECORD_BYTES}-byte records"
        )
    count = len(raw) // RECORD_BYTES
    cursor = 0
    actions: list[np.ndarray] = []
    q_des: list[np.ndarray] = []
    joint_pos: list[np.ndarray] = []
    gravity: list[np.ndarray] = []
    angular_velocity: list[np.ndarray] = []
    for _ in range(count):
        cursor += OBS_FLOATS * 4
        actions.append(np.frombuffer(raw, dtype=np.float32, count=ACTION_FLOATS, offset=cursor).copy())
        cursor += ACTION_FLOATS * 4
        q_des.append(np.frombuffer(raw, dtype=np.float64, count=QDES_FLOATS, offset=cursor).copy())
        cursor += QDES_FLOATS * 8
        joint_pos.append(np.frombuffer(raw, dtype=np.float64, count=JOINT_POS_FLOATS, offset=cursor).copy())
        cursor += JOINT_POS_FLOATS * 8
        cursor += JOINT_VEL_FLOATS * 8
        gravity.append(np.frombuffer(raw, dtype=np.float64, count=VEC3_FLOATS, offset=cursor).copy())
        cursor += VEC3_FLOATS * 8
        angular_velocity.append(np.frombuffer(raw, dtype=np.float64, count=VEC3_FLOATS, offset=cursor).copy())
        cursor += VEC3_FLOATS * 8

    q_des_array = np.asarray(q_des)
    joint_pos_array = np.asarray(joint_pos)
    action_array = np.asarray(actions)
    report = {
        "scope": "local_sil_onnx_stand_shadow_nonpublishing",
        "dump": str(args.dump),
        "records": count,
        "raw_action_abs_max": float(np.abs(action_array).max()),
        "q_des_abs_max_rad": float(np.abs(q_des_array).max()),
        "q_des_step_abs_max_rad": float(np.abs(np.diff(q_des_array, axis=0)).max()) if count > 1 else 0.0,
        "q_des_minus_measured_abs_max_rad": float(np.abs(q_des_array - joint_pos_array).max()),
        "gravity_norm_min": float(np.linalg.norm(np.asarray(gravity), axis=1).min()),
        "gravity_norm_max": float(np.linalg.norm(np.asarray(gravity), axis=1).max()),
        "base_angular_speed_max_rad_s": float(np.linalg.norm(np.asarray(angular_velocity), axis=1).max()),
        "not_a_balance_validation": True,
        "not_a_command_publication_test": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
