#!/usr/bin/env python3
"""Estimate phase-local tracking dynamics from a P4C action-chain trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


JOINTS = (
    "waist_pitch_joint",
    "waist_roll_joint",
    "right_shoulder_roll_joint",
)


def _phase(state: dict) -> str:
    if int(state.get("prelude_elapsed_steps", 0)) < 50:
        return "ready_to_swing"
    if int(state.get("tail_steps", 0)) > 0:
        return "recovery"
    if int(state.get("motion_frame", 0)) <= 30:
        return "pre_hit_swing"
    return "follow_through"


def _best_velocity_lag(command: np.ndarray, actual: np.ndarray, maximum: int = 15):
    best = {"lag_steps": 0, "correlation": 0.0}
    for lag in range(maximum + 1):
        retained = len(command) - lag
        if retained < 5:
            break
        left = command[:retained]
        right = actual[lag:]
        if len(left) < 5 or np.std(left) < 1.0e-8 or np.std(right) < 1.0e-8:
            continue
        corr = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(corr) and abs(corr) > abs(best["correlation"]):
            best = {"lag_steps": lag, "correlation": corr}
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-dt", type=float, default=0.02)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    samples = []
    for row in report["trace"]:
        state = row.get("post_step_state")
        if state is not None and "upper_action_chain" in state:
            samples.append(state)
    chain0 = samples[0]["upper_action_chain"]
    names = chain0["joint_names"]
    result = {
        "schema_version": "p4d_joint_dynamics_identification/v1",
        "source_report": str(args.report.resolve()),
        "control_dt_s": args.control_dt,
        "execution_mode": report.get("p4c_upper_execution_mode"),
        "joints": {},
    }
    for joint in JOINTS:
        index = names.index(joint)
        joint_result = {}
        for phase in (
            "ready_to_swing",
            "pre_hit_swing",
            "follow_through",
            "recovery",
        ):
            selected = [s for s in samples if _phase(s) == phase]
            if len(selected) < 3:
                continue
            command = np.asarray(
                [s["upper_action_chain"]["processed_command_position_rad"][index] for s in selected]
            )
            command_velocity = np.asarray(
                [s["upper_action_chain"]["processed_command_velocity_radps"][index] for s in selected]
            )
            actual = np.asarray(
                [s["upper_action_chain"]["actual_position_rad"][index] for s in selected]
            )
            actual_velocity = np.asarray(
                [s["upper_action_chain"]["actual_velocity_radps"][index] for s in selected]
            )
            actual_acceleration = np.diff(actual_velocity) / args.control_dt
            actual_jerk = np.diff(actual_acceleration) / args.control_dt
            tracking = actual - command
            joint_result[phase] = {
                "samples": len(selected),
                "tracking_error_rms_rad": float(np.sqrt(np.mean(tracking * tracking))),
                "tracking_error_peak_abs_rad": float(np.max(np.abs(tracking))),
                "tracking_error_signed_mean_rad": float(np.mean(tracking)),
                "actual_velocity_peak_abs_radps": float(np.max(np.abs(actual_velocity))),
                "actual_acceleration_peak_abs_radps2": float(np.max(np.abs(actual_acceleration))),
                "actual_jerk_peak_abs_radps3": float(np.max(np.abs(actual_jerk))),
                "command_velocity_peak_abs_radps": float(np.max(np.abs(command_velocity))),
                "minimum_actual_soft_margin_rad": float(
                    min(s["upper_action_chain"]["actual_soft_margin_rad"][index] for s in selected)
                ),
                "position_derived_command_to_actual_velocity_lag": _best_velocity_lag(
                    np.gradient(command, args.control_dt), actual_velocity
                ),
            }
        result["joints"][joint] = joint_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["joints"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
