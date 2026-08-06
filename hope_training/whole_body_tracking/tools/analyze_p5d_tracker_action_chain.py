#!/usr/bin/env python3
"""Audit P5D's reference, tracker-residual, safety, and actual-state chain.

Unlike the legacy P4D executor, a P5D processed command intentionally differs
from the safe reference when the generic tracker acts.  This report therefore
keeps the learned residual separate from the safety projection before measuring
the remaining physical tracking error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _matrix(samples: list[dict], field: str, width: int) -> np.ndarray:
    values = np.asarray([row[field] for row in samples], dtype=np.float64)
    if values.shape != (len(samples), width):
        raise ValueError(f"{field} has shape {values.shape}, expected {(len(samples), width)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{field} contains a non-finite value")
    return values


def _summary(values: np.ndarray, window: np.ndarray) -> dict[str, float]:
    return {
        "full_cycle_max_abs_rad": float(np.max(np.abs(values))),
        "hit_window_max_abs_rad": float(np.max(np.abs(window))),
        "hit_window_rms_rad": _rms(window),
    }


def _per_joint(names: list[str], values: np.ndarray, window: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {
            "full_cycle_max_abs_rad": float(np.max(np.abs(values[:, index]))),
            "hit_window_max_abs_rad": float(np.max(np.abs(window[:, index]))),
            "hit_window_rms_rad": _rms(window[:, index]),
        }
        for index, name in enumerate(names)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hit-radius", type=int, default=8)
    parser.add_argument("--safety-tolerance-rad", type=float, default=1.0e-6)
    args = parser.parse_args()
    if args.hit_radius < 0 or args.safety_tolerance_rad < 0.0:
        raise ValueError("hit radius and safety tolerance must be non-negative")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    trace = report.get("trace", [])
    post_states = [
        row["post_step_state"]
        for row in trace
        if isinstance(row.get("post_step_state"), dict)
    ]
    direct_samples = [
        state["reference_tracker_action_chain"]
        for state in post_states
        if "reference_tracker_action_chain" in state
    ]
    prior_samples = [
        state["upper_action_chain"]
        for state in post_states
        if "upper_action_chain" in state and "frozen_stage_a_support_chain" in state
    ]
    if direct_samples:
        mode = "direct_safe_reference_residual"
        samples = direct_samples
        prior = None
        names = list(samples[0].get("joint_names", []))
        width = len(names)
        safe_reference = _matrix(samples, "safe_reference_position_rad", width)
        residual = _matrix(samples, "effective_tracker_residual_rad", width)
        safety = _matrix(samples, "safety_override_rad", width)
        command = _matrix(samples, "processed_command_position_rad", width)
        actual = _matrix(samples, "actual_position_rad", width)
        reconstructed = safe_reference + residual + safety
        sample_steps = [
            int(row["control_step"])
            for row in trace
            if isinstance(row.get("post_step_state"), dict)
            and "reference_tracker_action_chain" in row["post_step_state"]
        ]
    elif prior_samples:
        mode = "frozen_900_3396_prior_plus_upper_residual"
        samples = prior_samples
        names = list(samples[0].get("joint_names", []))
        width = len(names)
        safe_reference = _matrix(samples, "safe_reference_position_rad", width)
        prior = _matrix(samples, "frozen_actor_contribution_rad", width)
        residual = _matrix(samples, "coordinator_contribution_rad", width)
        safety = _matrix(samples, "safety_override_rad", width)
        command = _matrix(samples, "processed_command_position_rad", width)
        actual = _matrix(samples, "actual_position_rad", width)
        reconstructed = safe_reference + prior + residual + safety
        sample_steps = [
            int(row["control_step"])
            for row in trace
            if isinstance(row.get("post_step_state"), dict)
            and "upper_action_chain" in row["post_step_state"]
            and "frozen_stage_a_support_chain" in row["post_step_state"]
        ]
    else:
        raise ValueError(
            "report has neither reference_tracker_action_chain nor prior-guided upper action-chain samples"
        )
    if not names:
        raise ValueError("P5D action chain has no joint_names")
    # JSON round-tripping of the CUDA float32 action fields can introduce a
    # few 1e-8 rad last-bit differences; this is still an exact execution
    # decomposition at the controller's precision.
    if not np.allclose(command, reconstructed, atol=1.0e-6, rtol=0.0):
        raise RuntimeError("processed command does not match its recorded execution decomposition")

    hit_record = report.get("strike") or {}
    hit_step_value = report.get("all_hit_control_step", hit_record.get("control_step"))
    if hit_step_value is None:
        raise ValueError("report has neither all_hit_control_step nor strike.control_step")
    hit_step = int(hit_step_value)
    steps = np.asarray(sample_steps)
    mask = np.abs(steps - hit_step) <= args.hit_radius
    if not np.any(mask):
        raise ValueError("hit window contains no action-chain samples")
    tracking = actual - command
    total = actual - safe_reference
    result = {
        "schema_version": "p5d_tracker_action_chain/v2",
        "source_report": str(args.report.resolve()),
        "motion_id": int(report["motion_id"]),
        "execution_mode": mode,
        "control_steps": len(samples),
        "tagged_hit_control_step": hit_step,
        "hit_window_radius_control_steps": args.hit_radius,
        "definitions": {
            "frozen_prior": "frozen model_900 upper contribution after its phase gate",
            "tracker_residual": "P5 PPO contribution after residual scale and phase gate",
            "safety_override": "processed_command - (safe_reference + tracker_residual)",
            "tracking": "actual - processed_command",
            "total": "actual - safe_reference",
        },
        "decomposition": {
            "identity": (
                "actual - safe_reference = frozen_prior + tracker_residual + "
                "safety_override + tracking"
                if prior is not None
                else "actual - safe_reference = tracker_residual + safety_override + tracking"
            ),
            "tracker_residual": {"aggregate": _summary(residual, residual[mask]), "per_joint": _per_joint(names, residual, residual[mask])},
            "safety_override": {"aggregate": _summary(safety, safety[mask]), "per_joint": _per_joint(names, safety, safety[mask])},
            "tracking": {"aggregate": _summary(tracking, tracking[mask]), "per_joint": _per_joint(names, tracking, tracking[mask])},
            "total": {"aggregate": _summary(total, total[mask]), "per_joint": _per_joint(names, total, total[mask])},
        },
        "safety_filter": {
            "tolerance_rad": args.safety_tolerance_rad,
            "triggered_control_steps": int(np.count_nonzero(np.any(np.abs(safety) > args.safety_tolerance_rad, axis=1))),
            "hit_window_triggered_control_steps": int(np.count_nonzero(np.any(np.abs(safety[mask]) > args.safety_tolerance_rad, axis=1))),
        },
    }
    if prior is not None:
        result["decomposition"]["frozen_prior"] = {
            "aggregate": _summary(prior, prior[mask]),
            "per_joint": _per_joint(names, prior, prior[mask]),
        }
        support = [
            state["frozen_stage_a_support_chain"]
            for state in post_states
            if "upper_action_chain" in state and "frozen_stage_a_support_chain" in state
        ]
        support_raw = np.asarray([sample["stage_a_raw_action"] for sample in support], dtype=np.float64)
        support_bounded = np.asarray(
            [sample["stage_a_masked_bounded_action"] for sample in support], dtype=np.float64
        )
        result["frozen_stage_a_support"] = {
            "checkpoint_role": "frozen_model_3396_leg_support",
            "action_dim": int(support_raw.shape[1]),
            "full_cycle_raw_max_abs": float(np.max(np.abs(support_raw))),
            "full_cycle_masked_bounded_max_abs": float(np.max(np.abs(support_bounded))),
            "hit_window_raw_max_abs": float(np.max(np.abs(support_raw[mask]))),
            "hit_window_masked_bounded_max_abs": float(np.max(np.abs(support_bounded[mask]))),
            "mask": support[0]["stage_a_action_mask"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
