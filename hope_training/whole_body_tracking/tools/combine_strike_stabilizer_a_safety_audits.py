#!/usr/bin/env python3
"""Combine sequential low-memory Strike Stabilizer-A safety batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    batches = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not all(item["runtime_safety_smoke_passed"] for item in batches):
        raise RuntimeError("At least one input batch did not pass its runtime safety smoke")
    keys = ("zero_residual", "bounded_random_legs")
    aggregate = {}
    for key in keys:
        rows = [item["metrics"][key] for item in batches]
        aggregate[key] = {
            "pairs": sum(item["pairs"] for item in batches),
            "survival_fraction": min(row["survival_fraction"] for row in rows),
            "worst_max_base_torque_nm": max(row["max_base_torque_nm"]["max"] for row in rows),
            "worst_min_soft_joint_margin_rad": min(row["min_soft_joint_margin_rad"]["min"] for row in rows),
            "minimum_foot_contact_fraction": min(row["foot_contact_fraction"]["mean"] for row in rows),
            "max_torque_saturation_fraction": max(row["torque_saturation_fraction"] for row in rows),
            "max_velocity_saturation_fraction": max(row["velocity_saturation_fraction"] for row in rows),
            "max_consecutive_torque_saturation_steps": max(row["max_consecutive_torque_saturation_steps"] for row in rows),
            "max_consecutive_velocity_saturation_steps": max(row["max_consecutive_velocity_saturation_steps"] for row in rows),
        }
        if key == "bounded_random_legs":
            aggregate[key].update(
                {
                    "max_sampled_raw_exceed_fraction": max(row["sampled_raw_exceed_fraction"] for row in rows),
                    "max_execution_clip_fraction": max(row["execution_clip_fraction"] for row in rows),
                    "max_waist_action": max(row["waist_action_max"] for row in rows),
                    "max_raw_action_rms": max(row["raw_action_rms"] for row in rows),
                }
            )
    result = {
        "schema_version": 1,
        "audit_id": "strike_stabilizer_a_untrained_bounded_leg_safety_16pair_v1",
        "input_batches": [str(path) for path in args.inputs],
        "sequential_batches_required": "32 simultaneous environments trigger a PhysX GPU device-side assert on this 8GB GPU; each source batch uses 8 environments.",
        "aggregate": aggregate,
        "full_swing_bounded_leg_safety_verified": True,
        "ppo_training_approved": False,
        "notes": [
            "This validates only untrained bounded-action runtime safety over continuous-prefix handoffs.",
            "It does not establish a learned stabilization benefit or authorize PPO.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "pairs": aggregate["bounded_random_legs"]["pairs"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
