#!/usr/bin/env python3
"""Summarize P4A formal-scene dynamic qualification runs.

This report intentionally separates three questions:

1. Can the legacy stabilizer execute the shape in the nominal P1 scene?
2. Is the reference itself limit-safe and fully observable for safety?
3. Is the resulting hit state close enough to use as a repair seed?

Only (1) is a dynamic-policy replay result.  A motion cannot receive an A
qualification while its reference violates a soft limit or while self
collision is not observable in the formal asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MOTION_IDS = (0, 2, 3, 4, 5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _state_series(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [row["pre_step_state"] for row in report.get("trace", []) if row.get("pre_step_state")]


def _minimum(states: list[dict[str, Any]], key: str) -> tuple[float | None, str | None]:
    rows = [(float(state[key]), state.get(key.replace("_rad", "_joint"))) for state in states if key in state]
    if not rows:
        return None, None
    return min(rows, key=lambda item: item[0])


def _extreme(states: list[dict[str, Any]], key: str, fn=max) -> float | None:
    values = [float(state[key]) for state in states if key in state]
    return fn(values) if values else None


def _nominal_summary(path: Path) -> dict[str, Any]:
    report = _load(path)
    trial = report["trials"][0]
    states = _state_series(report)
    min_actual_soft, min_actual_soft_joint = _minimum(states, "minimum_actual_soft_joint_margin_rad")
    min_actual_hard, min_actual_hard_joint = _minimum(states, "minimum_actual_hard_joint_margin_rad")
    min_reference_soft, min_reference_soft_joint = _minimum(states, "minimum_reference_soft_joint_margin_rad")
    min_reference_hard, min_reference_hard_joint = _minimum(states, "minimum_reference_hard_joint_margin_rad")
    hit_step = int(trial["control_step"])
    post_hit_states = [
        row["pre_step_state"]
        for row in report.get("trace", [])
        if int(row.get("control_step", -1)) >= hit_step and row.get("pre_step_state")
    ]
    rearm_ready_and_stable = any(
        bool(state.get("stage_a_rearm_ready")) and bool(state.get("stage_a_rearm_stable"))
        for state in post_hit_states
    )
    return {
        "source": str(path),
        "sha256": _sha256(path),
        "complete": bool(report.get("complete")),
        "control_steps": int(report.get("control_steps", 0)),
        "physical_termination_count": int(report.get("physical_termination_count", 0)),
        "timeout_count": int(report.get("timeout_count", 0)),
        "hit": {
            "position_error_m": float(trial["position_error_m"]),
            "normal_error_deg": float(trial["normal_error_deg"]),
            "velocity_error_mps": float(trial["velocity_error_mps"]),
        },
        "stability": {
            "max_root_tilt_deg": _extreme(states, "root_tilt_deg"),
            "max_loaded_foot_tangential_speed_mps": _extreme(states, "loaded_foot_tangential_speed_max_mps"),
            "max_effort_limit_ratio": _extreme(states, "max_effort_limit_ratio"),
            "rearm_ready_and_stable_seen_after_hit": rearm_ready_and_stable,
        },
        "limits": {
            "minimum_actual_soft_margin_rad": min_actual_soft,
            "minimum_actual_soft_margin_joint": min_actual_soft_joint,
            "minimum_actual_hard_margin_rad": min_actual_hard,
            "minimum_actual_hard_margin_joint": min_actual_hard_joint,
            "minimum_reference_soft_margin_rad": min_reference_soft,
            "minimum_reference_soft_margin_joint": min_reference_soft_joint,
            "minimum_reference_hard_margin_rad": min_reference_hard,
            "minimum_reference_hard_margin_joint": min_reference_hard_joint,
            "minimum_actual_waist_roll_soft_margin_rad": _extreme(states, "waist_roll_soft_margin_rad", min),
            "minimum_reference_waist_roll_soft_margin_rad": _extreme(
                states, "waist_roll_reference_soft_margin_rad", min
            ),
        },
    }


def _robustness_summary(path: Path) -> dict[str, Any]:
    report = _load(path)
    total = int(report.get("physical_termination_count", 0)) + int(report.get("timeout_count", 0))
    return {
        "source": str(path),
        "sha256": _sha256(path),
        "startup_physics_randomized": True,
        "environments": total,
        "physical_termination_count": int(report.get("physical_termination_count", 0)),
        "timeout_count": int(report.get("timeout_count", 0)),
    }


def _classify(nominal: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if nominal["physical_termination_count"] or not nominal["complete"]:
        return "D", ["nominal_formal_scene_execution_failed"]

    limits = nominal["limits"]
    if limits["minimum_reference_soft_margin_rad"] is None or limits["minimum_reference_soft_margin_rad"] <= 0.0:
        reasons.append("reference_soft_limit_violation_requires_deterministic_repair")
    reasons.append("self_collision_not_observable_in_formal_asset")

    hit = nominal["hit"]
    stability = nominal["stability"]
    seed_gates = {
        "position_error_le_0p08_m": hit["position_error_m"] <= 0.08,
        "normal_error_le_10_deg": hit["normal_error_deg"] <= 10.0,
        "velocity_error_le_1p5_mps": hit["velocity_error_mps"] <= 1.5,
        "recovery_seen_within_audit_horizon": stability["rearm_ready_and_stable_seen_after_hit"],
    }
    failed_seed_gates = [name for name, passed in seed_gates.items() if not passed]
    if failed_seed_gates:
        reasons.extend(f"repair_seed_gate_failed:{name}" for name in failed_seed_gates)
        return "C", reasons
    return "B", reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("eval_outputs/strike_goal_p4"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_outputs/strike_goal_p4/p4a_policy_qualification_summary.json"),
    )
    args = parser.parse_args()

    randomized_names = {
        0: "p1_policy_replay_motion0_smoke.json",
        2: "p1_policy_replay_motion2.json",
        3: "p1_policy_replay_motion3.json",
        4: "p1_policy_replay_motion4.json",
        5: "p1_policy_replay_motion5.json",
    }
    rows = []
    for motion_id in MOTION_IDS:
        nominal_path = args.input_dir / f"p1_policy_nominal_motion{motion_id}.json"
        randomized_path = args.input_dir / randomized_names[motion_id]
        nominal = _nominal_summary(nominal_path)
        classification, reasons = _classify(nominal)
        rows.append(
            {
                "motion_id": motion_id,
                "classification": classification,
                "classification_is_provisional": True,
                "classification_reasons": reasons,
                "nominal_formal_scene_policy_replay": nominal,
                "startup_randomized_robustness_probe": _robustness_summary(randomized_path),
            }
        )

    output = {
        "schema_version": "p4a_dynamic_qualification/v1",
        "purpose": "Formal P1 full-trajectory qualification of canonical legacy motion shape priors",
        "training_started": False,
        "ppo_allowed": False,
        "policy_replay_contract": {
            "scene": "formal P1 table/net/collision scene with P1-equivalent A3 placement",
            "controller": "existing floating-base target-conditioned upper policy and lower stabilizer",
            "nominal_startup_physics_randomization": False,
            "reference": "scene-placed canonical prior; joint trajectory and timing unchanged",
        },
        "interpretation": {
            "bare_pd_replay": "plant/controller baseline only; it is not used to classify a motion prior",
            "A": "fully qualified safe prior; unavailable until all limit and collision gates pass",
            "B": "dynamically usable repair seed; deterministic reference repair still required",
            "C": "shape prior only; material task-state/recovery reconstruction required",
            "D": "nominal formal-scene execution unsafe; exclude from teacher library",
        },
        "repair_seed_triage_gates": {
            "position_error_max_m": 0.08,
            "normal_error_max_deg": 10.0,
            "velocity_error_max_mps": 1.5,
            "recovery_ready_and_stable_required_within_audit_horizon": True,
            "note": "These gates only distinguish B from C; they are not final task acceptance tolerances.",
        },
        "global_blockers": [
            "all references have non-positive soft-limit margin",
            "formal A3 asset has self collision disabled, so self collision is not observable",
            "canonical ball-center strike goal still requires explicit ball-contact-to-policy-TCP conversion",
            "canonical goal, legacy calibrated center, adapted reference, and actual execution are not yet separated in one trace",
            "velocity target is not achieved accurately enough for 10D qualification",
        ],
        "motions": rows,
        "classification_counts": {
            label: sum(row["classification"] == label for row in rows) for label in ("A", "B", "C", "D")
        },
        "next_stage": "P4B deterministic limit-safe canonical prior repair; PPO remains disabled",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(output["classification_counts"], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
