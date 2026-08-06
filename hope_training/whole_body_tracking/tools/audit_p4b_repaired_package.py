#!/usr/bin/env python3
"""Audit internal FK and layered-goal consistency of a P4B package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_upper_momentum_library import UrdfModel, _quat_matrix


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "eval_outputs/strike_goal_p4/p4b_canonical_prior_candidate_v1/manifest.json",
    )
    parser.add_argument(
        "--repair-candidate",
        type=Path,
        default=ROOT / "eval_outputs/strike_goal_p4/p4b_repair_candidates/motion_00/repair_candidate.npz",
    )
    parser.add_argument("--metadata", type=Path, default=ROOT / "docs/a3_articulation_metadata.json")
    parser.add_argument("--urdf", type=Path, default=ROOT / "training/assets/agibot_a3/urdf/model.urdf")
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval_outputs/strike_goal_p4/p4b_package_consistency_audit.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    joint_names = list(metadata["joint_names"])
    body_names = list(metadata["body_names"])
    entries = [entry for entry in manifest["motions"] if int(entry["motion_id"]) == args.motion_id]
    if len(entries) != 1:
        raise ValueError(f"expected one motion {args.motion_id}")
    entry = entries[0]
    motion_path = args.manifest.parent / entry["canonical_motion_npz"]
    with np.load(motion_path, allow_pickle=False) as archive:
        joint_pos = np.asarray(archive["joint_pos"], dtype=np.float64)
        body_pos = np.asarray(archive["body_pos_b0"], dtype=np.float64)
        body_quat = np.asarray(archive["body_quat_b0_wxyz"], dtype=np.float64)
        momentum = np.asarray(archive["upper_momentum_pelvis"], dtype=np.float64)
    with np.load(args.repair_candidate, allow_pickle=False) as repair:
        expected_joint_pos = np.asarray(repair["projected_joint_pos"], dtype=np.float64)

    model = UrdfModel(args.urdf)
    position_errors = []
    orientation_errors_deg = []
    for frame, q in enumerate(joint_pos):
        fk = model.fk(dict(zip(joint_names, q, strict=True)))
        root_position = body_pos[frame, 0]
        root_rotation = _quat_matrix(body_quat[frame, 0], "wxyz")
        for body_index, name in enumerate(body_names):
            expected_position = root_position + root_rotation @ fk[name][:3, 3]
            expected_rotation = root_rotation @ fk[name][:3, :3]
            actual_rotation = _quat_matrix(body_quat[frame, body_index], "wxyz")
            position_errors.append(float(np.linalg.norm(body_pos[frame, body_index] - expected_position)))
            cosine = np.clip((np.trace(expected_rotation.T @ actual_rotation) - 1.0) * 0.5, -1.0, 1.0)
            orientation_errors_deg.append(float(np.degrees(np.arccos(cosine))))

    layers = entry.get("goal_state_layers", {})
    planner = layers.get("canonical_planner_goal_ball_center_impact_v1", {})
    gates = {
        "joint_candidate_exact_linf_le_1e-6_rad": float(np.max(np.abs(joint_pos - expected_joint_pos))) <= 1.0e-6,
        "body_fk_position_max_le_1e-5_m": max(position_errors) <= 1.0e-5,
        "body_fk_orientation_max_le_1e-3_deg": max(orientation_errors_deg) <= 1.0e-3,
        "momentum_shape_and_finite": momentum.shape == (joint_pos.shape[0], 6) and bool(np.isfinite(momentum).all()),
        "four_goal_layers_present": all(
            key in layers
            for key in (
                "canonical_planner_goal_ball_center_impact_v1",
                "canonical_motion_label_b0_before_repair",
                "legacy_calibrated_control_anchor_offset_b_m",
                "adapted_reference_hit_state_b0",
                "actual_execution_hit_state",
            )
        ),
        "offline_timing_not_misrepresented_as_live_tts": (
            planner.get("time_to_strike_s", "missing") is None
            and planner.get("timing_status") == "must_be_filled_from_live_control_clock_at_command_receipt"
        ),
    }
    output = {
        "schema_version": "p4b_package_consistency/v1",
        "training_started": False,
        "ppo_allowed": False,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "motion": str(motion_path.resolve()),
        "motion_sha256": _sha256(motion_path),
        "motion_id": args.motion_id,
        "joint_candidate_linf_rad": float(np.max(np.abs(joint_pos - expected_joint_pos))),
        "maximum_body_fk_position_error_m": max(position_errors),
        "maximum_body_fk_orientation_error_deg": max(orientation_errors_deg),
        "gates": gates,
        "all_package_gates_pass": all(gates.values()),
        "not_qualified_here": [
            "formal P1 PhysX dynamic replay",
            "actual-execution goal layer",
            "self collision in the formal asset",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
