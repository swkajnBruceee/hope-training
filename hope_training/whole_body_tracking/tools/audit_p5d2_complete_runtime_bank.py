#!/usr/bin/env python3
"""Fail-closed audit for the regenerated complete P5D-2 bank.

The audit separates static/offline admission from runtime evidence.  A static
reference can be geometrically valid and still cannot be published as a
tracker reference until the frozen model_900/model_3396 prior and the exact
P1 safety filter have replay evidence.  No training is launched.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_runtime_reference_bank/p5d2_complete_all_reference_bank_manifest.json"
LIMITS = ROOT / "cfg/p5_reference_dynamics_v1.json"
OUT = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_runtime_reference_audit"
WORLD_ANCHOR = np.asarray((-0.5, -0.7625, 1.04), dtype=np.float64)


def finite3(x: object) -> bool:
    a = np.asarray(x, dtype=np.float64)
    return a.shape == (3,) and bool(np.isfinite(a).all())


def main() -> None:
    payload = json.loads(BANK.read_text())
    limits = json.loads(LIMITS.read_text())
    audited = copy.deepcopy(payload)
    rows = []
    reason_counts: dict[str, int] = {}
    static_eligible = 0
    published_eligible = 0
    for entry in audited["motions"]:
        checks: dict[str, bool] = {}
        reasons: list[str] = []
        goal = entry.get("canonical_goal_10d", {})
        checks["canonical_goal_10d_contract"] = bool(finite3(goal.get("position_b0_m")) and finite3(goal.get("normal_b0")) and finite3(goal.get("linear_velocity_b0_mps")) and np.isfinite(float(goal.get("time_to_hit_s", np.nan))) and float(goal.get("time_to_hit_s", 0.0)) > 0.0)
        if not checks["canonical_goal_10d_contract"]: reasons.append("invalid_canonical_goal_10d")
        target = entry.get("strike_target", {})
        if checks["canonical_goal_10d_contract"]:
            target_pos = np.asarray(target.get("racket_position_m", [np.nan] * 3), dtype=float)
            target_vel = np.asarray(target.get("racket_velocity_mps", [np.nan] * 3), dtype=float)
            target_n = np.asarray(target.get("racket_normal_w", [np.nan] * 3), dtype=float)
            checks["frame_tcp_contract"] = bool(np.linalg.norm(target_pos - (WORLD_ANCHOR + np.asarray(goal["position_b0_m"]))) <= 2e-3 and np.linalg.norm(target_vel - np.asarray(goal["linear_velocity_b0_mps"])) <= 2e-3 and np.degrees(np.arccos(np.clip(np.dot(target_n, np.asarray(goal["normal_b0"])), -1.0, 1.0))) <= 0.2)
        else:
            checks["frame_tcp_contract"] = False
        if not checks["frame_tcp_contract"]: reasons.append("frame_or_tcp_contract_mismatch")

        path = Path(entry["motion_npz"])
        checks["runtime_npz_contract"] = path.exists()
        dyn = {}
        z = None
        if checks["runtime_npz_contract"]:
            z = np.load(path, allow_pickle=False)
            required = {"fps", "joint_pos", "joint_vel", "body_pos_b0", "body_quat_b0_wxyz", "body_lin_vel_b0", "body_ang_vel_b0", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "hit_frame", "canonical_goal_position_b0_m", "canonical_goal_normal_b0", "canonical_goal_linear_velocity_b0_mps", "canonical_goal_time_to_hit_s"}
            checks["runtime_npz_contract"] = required.issubset(z.files) and z["joint_pos"].ndim == 2 and z["joint_pos"].shape[1] == 31 and z["joint_pos"].shape[0] == z["joint_vel"].shape[0] and z["joint_pos"].shape[0] > int(np.asarray(z["hit_frame"]).reshape(-1)[0]) + 1
            if checks["runtime_npz_contract"]:
                q = np.asarray(z["joint_pos"], dtype=float); qd = np.asarray(z["joint_vel"], dtype=float); fps = float(np.asarray(z["fps"]).reshape(-1)[0]); qdd = np.diff(qd, axis=0) * fps; jerk = np.diff(qdd, axis=0) * fps
                dyn = {"frames": int(q.shape[0]), "hit_frame": int(np.asarray(z["hit_frame"]).reshape(-1)[0]), "fps": fps, "max_abs_joint_velocity_radps": float(np.max(np.abs(qd))), "max_abs_joint_acceleration_radps2": float(np.max(np.abs(qdd))) if qdd.size else 0.0, "max_abs_joint_jerk_radps3": float(np.max(np.abs(jerk))) if jerk.size else 0.0}
        if not checks["runtime_npz_contract"]: reasons.append("runtime_npz_contract_missing_or_incomplete")
        checks["finite_dynamics"] = bool(dyn) and all(np.isfinite(float(v)) for k, v in dyn.items() if k not in ("frames", "hit_frame"))
        if not checks["finite_dynamics"]: reasons.append("nonfinite_dynamics")
        checks["declared_dynamic_limits"] = bool(dyn) and dyn.get("max_abs_joint_velocity_radps", np.inf) <= float(limits["max_abs_joint_velocity_radps"]) and dyn.get("max_abs_joint_acceleration_radps2", np.inf) <= float(limits["max_abs_joint_acceleration_radps2"]) and dyn.get("max_abs_joint_jerk_radps3", np.inf) <= float(limits["max_abs_joint_jerk_radps3"])
        if not checks["declared_dynamic_limits"]: reasons.append("declared_dynamic_limits_failed")
        contract = entry.get("reference_contract", {})
        checks["complete_follow_through_recovery"] = bool(contract.get("post_hit_recovery_frames", 0) >= int(limits["recovery_tail_frames"]) and contract.get("actual_trajectory_as_reference") is False)
        if not checks["complete_follow_through_recovery"]: reasons.append("missing_follow_through_or_recovery")
        offline = entry.get("p5d2_bank", {})
        # The source complete-manifest gates are copied into the NPZ selection;
        # the runtime package itself is not allowed to invent a new target.
        checks["offline_complete_gate"] = bool(entry.get("reference_contract", {}).get("frames", 0) > 0 and entry.get("reference_contract", {}).get("hit_frame", -1) >= 0)
        if not checks["offline_complete_gate"]: reasons.append("offline_complete_gate_missing")
        static_checks = [checks[k] for k in ("canonical_goal_10d_contract", "frame_tcp_contract", "runtime_npz_contract", "finite_dynamics", "declared_dynamic_limits", "complete_follow_through_recovery", "offline_complete_gate")]
        checks["offline_static_admission"] = all(static_checks)
        if not checks["offline_static_admission"]: reasons.append("offline_static_admission_failed")
        static_eligible += int(checks["offline_static_admission"])

        # These are deliberately not inferred from an NPZ.  They need the
        # exact frozen prior/safety implementation and a versioned replay.
        checks["model900_model3396_prior_compatibility"] = False
        checks["safety_projection_replay"] = False
        checks["neighbor_continuation_audit"] = False
        reasons.extend(["prior_compatibility_not_replayed", "safety_projection_not_replayed", "neighbor_continuity_not_audited"])
        status = "TRACKER_TRAINING_ELIGIBLE" if all(checks.values()) else "OFFLINE_REJECTED"
        entry["qualification_status"] = status
        entry["qualification_reasons"] = reasons
        entry["reference_audit"] = {"checks": checks, "dynamics": dyn, "runtime_execution_required": True}
        rows.append({"motion_id": entry.get("motion_id"), "sample_id": offline.get("sample_id"), "seed_motion_id": offline.get("source_seed_motion_id"), "category": offline.get("category"), "status": status, "reasons": reasons, "checks": checks, "dynamics": dyn})
        for reason in reasons: reason_counts[reason] = reason_counts.get(reason, 0) + 1
        published_eligible += int(status == "TRACKER_TRAINING_ELIGIBLE")

    audited["status"] = "offline_static_pass_runtime_evidence_pending"
    audited["audit_contract"] = "p5_reference_admission/v1"
    audited["audit_policy"] = {"missing_evidence_fails_closed": True, "actual_trajectory_as_reference": False, "motion4_excluded": True, "training_started": False}
    audited["audit_summary"] = {"reference_count": len(rows), "offline_static_eligible_count": static_eligible, "tracker_training_eligible_count": published_eligible, "qualified_teacher_count": 0}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p5d2_complete_all_reference_bank_audited.json").write_text(json.dumps(audited, ensure_ascii=False, indent=2) + "\n")
    report = {"schema_version": "p5_reference_admission_audit/v2", "source_bank": str(BANK.resolve()), "reference_count": len(rows), "offline_static_eligible_count": static_eligible, "status_counts": {s: sum(r["status"] == s for r in rows) for s in ("OFFLINE_REJECTED", "TRACKER_TRAINING_ELIGIBLE", "QUALIFIED_TEACHER")}, "reason_counts": reason_counts, "training_started": False, "rows": rows}
    (OUT / "p5d2_complete_reference_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reference_count": len(rows), "offline_static_eligible_count": static_eligible, "status_counts": report["status_counts"], "reason_counts": reason_counts, "report": str(OUT / "p5d2_complete_reference_audit_report.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
