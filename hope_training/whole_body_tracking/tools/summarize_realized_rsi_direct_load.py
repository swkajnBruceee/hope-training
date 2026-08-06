#!/usr/bin/env python3
"""Compare realized RSI direct-load first steps with continuous rollout steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--direct-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.bank / "realized_rsi_manifest.json").read_text(encoding="utf-8"))
    state_by_episode = {
        str(entry["episode_id"]): args.bank / str(entry["state_file"])
        for entry in manifest["entries"]
    }
    direct = json.loads(args.direct_result.read_text(encoding="utf-8"))
    rows = []
    for row in direct["cases"]:
        with np.load(state_by_episode[str(row["episode_id"])], allow_pickle=False) as state:
            steps = np.asarray(state["motion_step"], dtype=np.int64)
            indexes = np.flatnonzero(steps == int(row["loaded_step"]))
            if indexes.size != 1 or int(indexes[0]) + 1 >= len(steps):
                continue
            index = int(indexes[0])
            continuous_q_delta = float(
                np.linalg.norm(state["joint_pos"][index + 1] - state["joint_pos"][index])
            )
            continuous_torque_max = float(np.abs(state["applied_torque"][index + 1]).max())
        direct_q_delta = float(row["first_joint_delta_norm"])
        direct_torque_max = float(row["first_torque_abs_max_nm"])
        rows.append(
            {
                "episode_id": row["episode_id"],
                "phase": row["phase"],
                "loaded_step": int(row["loaded_step"]),
                "direct_joint_delta_norm": direct_q_delta,
                "continuous_joint_delta_norm": continuous_q_delta,
                "joint_delta_ratio": direct_q_delta / max(continuous_q_delta, 1.0e-9),
                "direct_torque_abs_max_nm": direct_torque_max,
                "continuous_torque_abs_max_nm": continuous_torque_max,
                "torque_ratio": direct_torque_max / max(continuous_torque_max, 1.0e-9),
            }
        )

    max_joint_ratio = max((row["joint_delta_ratio"] for row in rows), default=float("inf"))
    max_torque_ratio = max((row["torque_ratio"] for row in rows), default=float("inf"))
    passed = bool(rows) and max_joint_ratio <= 1.25 and max_torque_ratio <= 1.50
    payload = {
        "schema_version": 1,
        "stage": "native_strike_realized_rsi_direct_load_continuity_audit_v1",
        "training_eligible": False,
        "passed": passed,
        "criteria": {
            "max_first_step_joint_delta_ratio": 1.25,
            "max_first_step_torque_ratio": 1.50,
        },
        "summary": {
            "cases": len(rows),
            "max_joint_delta_ratio": max_joint_ratio,
            "max_torque_ratio": max_torque_ratio,
            "diagnosis": (
                "direct load preserves q/dq/root/PD target but not the full PhysX contact and solver history"
                if not passed
                else "first-step continuation is within the provisional continuity envelope"
            ),
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, **payload["summary"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
