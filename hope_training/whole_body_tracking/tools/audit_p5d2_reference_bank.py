#!/usr/bin/env python3
"""Audit every generated P5D-2 reference against the frozen P5 contract.

This audit is deliberately fail-closed: missing evidence is not interpreted as
passing.  It produces an audited copy of the manifest and a machine-readable
summary; it never starts simulation or training.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "eval_outputs/strike_goal_p5/p5d2_runtime_reference_bank"
OFFLINE = ROOT / "eval_outputs/strike_goal_p5/p5d2_multianchor_offline_repaired_v2/manifest.json"
OUT = ROOT / "eval_outputs/strike_goal_p5/p5d2_reference_audit"


def finite3(value: object) -> bool:
    x = np.asarray(value, dtype=np.float64)
    return x.shape == (3,) and bool(np.isfinite(x).all())


def main() -> None:
    offline = json.loads(OFFLINE.read_text())
    offline_by = {(s["sample_id"], int(a["seed_motion_id"])): a for s in offline["samples"] for a in s["seed_attempts"]}
    OUT.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    # Audit the canonical all-splits bank once.  Split manifests are views of
    # the same entries and must not inflate the audit count.
    for manifest_path in [BANK / "p5d2_all_reference_bank_manifest.json"]:
        payload = json.loads(manifest_path.read_text())
        audited = copy.deepcopy(payload)
        audited["audit_contract"] = "p5_reference_admission/v1"
        audited["qualification_policy"] = {
            "missing_evidence_fails_closed": True,
            "physx_teacher_approval": False,
            "actual_execution_used_as_reference": False,
        }
        rows = []
        for entry in audited.get("motions", []):
            bank = entry.get("p5d2_bank", {})
            sample_id = bank.get("sample_id")
            seed = bank.get("source_seed_motion_id")
            checks: dict[str, bool] = {}
            reasons: list[str] = []
            goal = entry.get("canonical_goal_10d")
            checks["canonical_goal_10d"] = bool(
                isinstance(goal, dict)
                and finite3(goal.get("position_b0_m"))
                and finite3(goal.get("normal_b0"))
                and finite3(goal.get("linear_velocity_b0_mps"))
                and np.isfinite(float(goal.get("time_to_hit_s", np.nan)))
                and float(goal.get("time_to_hit_s", 0.0)) > 0.0
            )
            if not checks["canonical_goal_10d"]: reasons.append("missing_or_invalid_canonical_goal_10d")
            if checks["canonical_goal_10d"] and isinstance(entry.get("strike_target"), dict):
                target = entry["strike_target"]
                goal_pos_w = np.asarray(goal["position_b0_m"], dtype=np.float64) + np.asarray((-0.5, -0.7625, 1.04))
                checks["canonical_target_frame_consistency"] = bool(
                    np.linalg.norm(np.asarray(target.get("racket_position_m", [np.nan] * 3)) - goal_pos_w) <= 2.0e-3
                    and np.linalg.norm(np.asarray(target.get("racket_velocity_mps", [np.nan] * 3)) - np.asarray(goal["linear_velocity_b0_mps"])) <= 2.0e-3
                    and np.degrees(np.arccos(np.clip(np.dot(np.asarray(target.get("racket_normal_w", [np.nan] * 3)), np.asarray(goal["normal_b0"])), -1.0, 1.0))) <= 0.2
                )
            else:
                checks["canonical_target_frame_consistency"] = False
            if not checks["canonical_target_frame_consistency"]: reasons.append("canonical_target_frame_or_tcp_contract_mismatch")

            path = Path(entry.get("motion_npz", ""))
            checks["runtime_npz_contract"] = path.exists()
            if checks["runtime_npz_contract"]:
                z = np.load(path, allow_pickle=False)
                required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
                checks["runtime_npz_contract"] = required.issubset(set(z.files)) and z["joint_pos"].shape[1:] == (31,) and z["body_pos_w"].shape[1:] == (32, 3)
                q = np.asarray(z["joint_pos"], dtype=np.float64)
                qd = np.asarray(z["joint_vel"], dtype=np.float64)
                fps = float(np.asarray(z["fps"]).reshape(-1)[0])
                qdd = np.diff(qd, axis=0) * fps
                jerk = np.diff(qdd, axis=0) * fps
                dynamics = {"max_abs_qd_radps": float(np.max(np.abs(qd))), "max_abs_qdd_radps2": float(np.max(np.abs(qdd))) if qdd.size else 0.0, "max_abs_jerk_radps3": float(np.max(np.abs(jerk))) if jerk.size else 0.0, "fps": fps, "frames": int(q.shape[0])}
            else:
                dynamics = {}
                reasons.append("missing_runtime_npz_contract")
            checks["finite_dynamic_derivatives"] = bool(dynamics) and all(np.isfinite(float(v)) for k, v in dynamics.items() if k != "frames")
            if not checks["finite_dynamic_derivatives"]: reasons.append("nonfinite_or_missing_dynamic_derivatives")

            strike_only = entry.get("strike_only_contract", {})
            checks["complete_follow_through_and_recovery"] = bool(
                strike_only.get("post_hit_frames_kept", 0) > 0
                and strike_only.get("zero_velocity_tail_frames", 0) == 0
            )
            if not checks["complete_follow_through_and_recovery"]:
                reasons.append("strike_only_zero_tail_or_missing_follow_through_recovery")

            evidence = offline_by.get((sample_id, int(seed))) if sample_id is not None and seed is not None else None
            if evidence is not None:
                gates = evidence.get("offline_gates", {})
                checks["offline_geometry_limits_collision"] = bool(gates) and all(bool(v) for v in gates.values())
            else:
                # Anchors have separate repair audits, but the generated bank
                # does not carry a complete per-entry geometry certificate.
                checks["offline_geometry_limits_collision"] = False
            if not checks["offline_geometry_limits_collision"]: reasons.append("missing_or_failed_offline_geometry_certificate")

            # These cannot be inferred from a static NPZ or from a MuJoCo
            # candidate alone.  They require explicit versioned limits and a
            # prior/safety/PhysX replay, so fail closed for admission.
            checks["declared_dynamic_limits_pass"] = False
            checks["prior_residual_range_pass"] = False
            checks["safety_projection_pass"] = False
            checks["continuation_neighborhood_pass"] = False
            reasons.extend(["dynamic_limits_not_declared", "prior_residual_range_not_audited", "safety_projection_not_replayed", "neighbor_continuity_not_audited"])

            status = "TRACKER_TRAINING_ELIGIBLE" if all(checks.values()) else "OFFLINE_REJECTED"
            entry["qualification_status"] = status
            entry["qualification_reasons"] = reasons
            entry["reference_audit"] = {"checks": checks, "dynamics": dynamics}
            rows.append({"motion_id": entry.get("motion_id"), "sample_id": sample_id, "seed_motion_id": seed, "status": status, "reasons": reasons, "checks": checks, "dynamics": dynamics})
        out_path = OUT / manifest_path.name
        out_path.write_text(json.dumps(audited, ensure_ascii=False, indent=2) + "\n")
        summary.extend(rows)

    report = {
        "schema_version": "p5_reference_admission_audit/v1",
        "policy": "fail_closed_missing_evidence",
        "source_bank": str(BANK),
        "reference_count": len(summary),
        "status_counts": {s: sum(x["status"] == s for x in summary) for s in ("OFFLINE_REJECTED", "TRACKER_TRAINING_ELIGIBLE", "QUALIFIED_TEACHER")},
        "reason_counts": {},
        "rows": summary,
    }
    for row in summary:
        for reason in row["reasons"]:
            report["reason_counts"][reason] = report["reason_counts"].get(reason, 0) + 1
    (OUT / "p5d2_reference_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reference_count": report["reference_count"], "status_counts": report["status_counts"], "reason_counts": report["reason_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
