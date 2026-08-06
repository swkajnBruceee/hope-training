#!/usr/bin/env python3
"""Materialize a P4B repair candidate as an internally consistent prior.

The repaired joint trajectory is expanded through the same prepared A3 URDF
used by the project momentum builder.  Body pose/velocity and upper momentum
arrays are regenerated; stale nominal body arrays are never copied onto the
repaired joints.  The output retains the canonical frame contract solely so it
can pass through the existing rigid scene-placement layer.  It remains an
evaluation candidate, not a training-approved library.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from build_upper_momentum_library import (
    UrdfModel,
    _compute_momentum,
    _quat_matrix,
    _rotation_log,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "eval_outputs/strike_goal_p3/canonical_motion_prior_v1/manifest.json"
DEFAULT_REPAIR = ROOT / "eval_outputs/strike_goal_p4/p4b_repair_candidates/motion_00/repair_candidate.npz"
DEFAULT_OUTPUT = ROOT / "eval_outputs/strike_goal_p4/p4b_canonical_prior_candidate_v1"
DEFAULT_METADATA = ROOT / "docs/a3_articulation_metadata.json"
DEFAULT_URDF = ROOT / "training/assets/agibot_a3/urdf/model.urdf"
CANONICAL_CONTRACT = "motion_prior_base_heading_frame0/v1"
MOUNT_OFFSET = np.asarray((0.210211399202899, 0.0320784994676765, 0.0320358706296689), dtype=np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Stable rotation-matrix to normalized wxyz quaternion conversion."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        value = np.array(
            (0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale)
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            value = np.array(((rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                              (rotation[0, 1] + rotation[1, 0]) / scale,
                              (rotation[0, 2] + rotation[2, 0]) / scale))
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            value = np.array(((rotation[0, 2] - rotation[2, 0]) / scale,
                              (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                              (rotation[1, 2] + rotation[2, 1]) / scale))
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            value = np.array(((rotation[1, 0] - rotation[0, 1]) / scale,
                              (rotation[0, 2] + rotation[2, 0]) / scale,
                              (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale))
    value /= np.linalg.norm(value)
    return value


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    output = np.empty_like(values)
    output[0] = (values[1] - values[0]) * fps
    output[-1] = (values[-1] - values[-2]) * fps
    output[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return output


def _regenerate_body_arrays(
    model: UrdfModel,
    joint_names: list[str],
    body_names: list[str],
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    frames = joint_pos.shape[0]
    body_pos = np.empty((frames, len(body_names), 3), dtype=np.float64)
    rotations = np.empty((frames, len(body_names), 3, 3), dtype=np.float64)
    body_quat = np.empty((frames, len(body_names), 4), dtype=np.float64)
    for frame in range(frames):
        fk = model.fk(dict(zip(joint_names, joint_pos[frame], strict=True)))
        root_rotation = _quat_matrix(root_quat[frame], "wxyz")
        for body_index, name in enumerate(body_names):
            local = fk[name]
            body_pos[frame, body_index] = root_pos[frame] + root_rotation @ local[:3, 3]
            rotations[frame, body_index] = root_rotation @ local[:3, :3]
            body_quat[frame, body_index] = _matrix_to_quat_wxyz(rotations[frame, body_index])
    return body_pos, body_quat


def _relative_body_velocity_from_joint_state(
    model: UrdfModel,
    joint_names: list[str],
    body_names: list[str],
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return link velocity relative to the root using q/qd directional FK.

    The repair velocity is applied as a delta to the source body velocity
    field.  This preserves the source time/velocity convention exactly when
    the deformation is zero instead of re-deriving and silently rescaling it.
    """
    epsilon_s = 1.0e-5
    linear = np.zeros((joint_pos.shape[0], len(body_names), 3), dtype=np.float64)
    angular = np.zeros_like(linear)
    for frame, (q, qd) in enumerate(zip(joint_pos, joint_vel, strict=True)):
        plus = model.fk(dict(zip(joint_names, q + epsilon_s * qd, strict=True)))
        minus = model.fk(dict(zip(joint_names, q - epsilon_s * qd, strict=True)))
        root_rotation = _quat_matrix(root_quat[frame], "wxyz")
        for body_index, name in enumerate(body_names):
            linear[frame, body_index] = root_rotation @ (
                (plus[name][:3, 3] - minus[name][:3, 3]) / (2.0 * epsilon_s)
            )
            relative_rotation = minus[name][:3, :3].T @ plus[name][:3, :3]
            omega_local_root = minus[name][:3, :3] @ (
                _rotation_log(relative_rotation) / (2.0 * epsilon_s)
            )
            angular[frame, body_index] = root_rotation @ omega_local_root
    return linear, angular


def _tcp_hit_state(arrays: dict[str, np.ndarray], wrist_index: int, hit: int) -> dict[str, list[float]]:
    wrist_quat = arrays["body_quat_b0_wxyz"][hit, wrist_index]
    rotation = _quat_matrix(wrist_quat, "wxyz")
    offset = rotation @ MOUNT_OFFSET
    position = arrays["body_pos_b0"][hit, wrist_index] + offset
    velocity = arrays["body_lin_vel_b0"][hit, wrist_index] + np.cross(
        arrays["body_ang_vel_b0"][hit, wrist_index], offset
    )
    normal = rotation[:, 1]
    tangent = rotation[:, 0]
    return {
        "racket_position_b0_m": position.tolist(),
        "racket_velocity_b0_mps": velocity.tolist(),
        "racket_normal_b0": normal.tolist(),
        "racket_tangent_b0": tangent.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repair-candidate", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--scene-source-manifest",
        type=Path,
        default=None,
        help=(
            "Existing formal P1 source manifest used only by the subsequent "
            "rigid scene-placement step. Defaults to the canonical manifest's source."
        ),
    )
    args = parser.parse_args()

    source_manifest = json.loads(args.canonical_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("contract_version") != CANONICAL_CONTRACT:
        raise ValueError("input is not a canonical motion-prior manifest")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    joint_names = list(metadata["joint_names"])
    body_names = list(metadata["body_names"])
    wrist_index = body_names.index("right_wrist_yaw_Link")
    model = UrdfModel(args.urdf)
    upper_links = sorted(
        name for name in model.descendants("waist_yaw_Link")
        if name in model.inertials and model.inertials[name].mass > 0.0
    )
    with np.load(args.repair_candidate, allow_pickle=False) as repair:
        repaired_q = np.asarray(repair["projected_joint_pos"] if "projected_joint_pos" in repair else repair["joint_pos"], dtype=np.float64)
        repaired_qd = np.asarray(repair["projected_joint_vel"] if "projected_joint_vel" in repair else repair["joint_vel"], dtype=np.float64)
        candidate_goal = (
            {
                "position": np.asarray(repair["canonical_goal_position_b0_m"], dtype=np.float64),
                "normal": np.asarray(repair["canonical_goal_normal_b0"], dtype=np.float64),
                "velocity": np.asarray(repair["canonical_goal_linear_velocity_b0_mps"], dtype=np.float64),
                "time": float(np.asarray(repair["canonical_goal_time_to_hit_s"]).reshape(-1)[0]),
            }
            if "canonical_goal_position_b0_m" in repair else None
        )
    repair_audit_path = args.repair_candidate.with_name("repair_audit.json")
    repair_audit = (
        json.loads(repair_audit_path.read_text(encoding="utf-8"))
        if repair_audit_path.is_file()
        else None
    )

    output_manifest = copy.deepcopy(source_manifest)
    if args.scene_source_manifest is not None:
        scene_source_manifest = args.scene_source_manifest.expanduser().resolve()
        if not scene_source_manifest.is_file():
            raise FileNotFoundError(f"scene source manifest does not exist: {scene_source_manifest}")
        output_manifest["source_manifest"] = str(scene_source_manifest)
        output_manifest["source_manifest_sha256"] = _sha256(scene_source_manifest)
    output_manifest["qualification"] = "p4b_deterministic_limit_repair_candidate"
    output_manifest["training_approved"] = False
    output_manifest["training_role"] = "formal_dynamic_qualification_only"
    output_manifest["p4b_repair_contract"] = {
        "version": "p4b_canonical_prior_materialization/v1",
        "source_manifest": str(args.canonical_manifest.resolve()),
        "source_manifest_sha256": _sha256(args.canonical_manifest),
        "repair_candidate": str(args.repair_candidate.resolve()),
        "repair_candidate_sha256": _sha256(args.repair_candidate),
        "motion_id": args.motion_id,
        "body_state_regenerated": True,
        "upper_momentum_regenerated": True,
        "ppo_allowed": False,
    }
    output_motion_dir = args.output_dir / "motion_npz"
    output_motion_dir.mkdir(parents=True, exist_ok=True)

    output_entries = []
    for entry in source_manifest["motions"]:
        output_entry = copy.deepcopy(entry)
        source_path = args.canonical_manifest.parent / entry["canonical_motion_npz"]
        with np.load(source_path, allow_pickle=False) as source:
            arrays = {name: np.asarray(source[name]).copy() for name in source.files}
        if int(entry["motion_id"]) == args.motion_id:
            if repaired_q.shape != arrays["joint_pos"].shape:
                raise ValueError("repair candidate joint shape differs from canonical motion")
            fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
            root_pos = np.asarray(arrays["body_pos_b0"][:, 0], dtype=np.float64)
            root_quat = np.asarray(arrays["body_quat_b0_wxyz"][:, 0], dtype=np.float64)
            body_pos, body_quat = _regenerate_body_arrays(
                model, joint_names, body_names, repaired_q, root_pos, root_quat, fps
            )
            source_joint_pos = np.asarray(arrays["joint_pos"], dtype=np.float64)
            source_joint_vel = np.asarray(arrays["joint_vel"], dtype=np.float64)
            source_relative_lin, source_relative_ang = _relative_body_velocity_from_joint_state(
                model, joint_names, body_names, source_joint_pos, source_joint_vel, root_quat
            )
            repaired_relative_lin, repaired_relative_ang = _relative_body_velocity_from_joint_state(
                model, joint_names, body_names, repaired_q, repaired_qd, root_quat
            )
            body_lin = (
                np.asarray(arrays["body_lin_vel_b0"], dtype=np.float64)
                + repaired_relative_lin
                - source_relative_lin
            )
            body_ang = (
                np.asarray(arrays["body_ang_vel_b0"], dtype=np.float64)
                + repaired_relative_ang
                - source_relative_ang
            )
            momentum, upper_mass, upper_length, _ = _compute_momentum(
                model, repaired_q, joint_names, upper_links, 1.0 / fps
            )
            arrays.update(
                joint_pos=repaired_q.astype(np.float32),
                joint_vel=repaired_qd.astype(np.float32),
                body_pos_b0=body_pos.astype(np.float32),
                body_quat_b0_wxyz=body_quat.astype(np.float32),
                body_lin_vel_b0=body_lin.astype(np.float32),
                body_ang_vel_b0=body_ang.astype(np.float32),
                upper_momentum_pelvis=momentum.astype(np.float32),
                upper_mass_kg=np.asarray((upper_mass,), dtype=np.float32),
                upper_length_scale_m=np.asarray((upper_length,), dtype=np.float32),
            )
            hit = int(entry["hit_frame"])
            adapted_reference = _tcp_hit_state(arrays, wrist_index, hit)
            original_label = copy.deepcopy(entry["strike_target_b0"])
            if candidate_goal is not None:
                old_position = np.asarray(original_label["racket_position_b0_m"], dtype=np.float64)
                position_delta = candidate_goal["position"] - old_position
                original_label["racket_position_b0_m"] = candidate_goal["position"].tolist()
                original_label["racket_normal_b0"] = (candidate_goal["normal"] / np.linalg.norm(candidate_goal["normal"])).tolist()
                original_label["racket_velocity_b0_mps"] = candidate_goal["velocity"].tolist()
                if "ball_position_b0_m" in original_label:
                    original_label["ball_position_b0_m"] = (np.asarray(original_label["ball_position_b0_m"], dtype=np.float64) + position_delta).tolist()
            output_entry["goal_state_layers"] = {
                "canonical_planner_goal_ball_center_impact_v1": {
                    "position_b0_m": original_label["ball_position_b0_m"],
                    "normal_b0": original_label["racket_normal_b0"],
                    "linear_velocity_b0_mps": original_label["racket_velocity_b0_mps"],
                    "time_to_strike_s": None,
                    "timing_status": "must_be_filled_from_live_control_clock_at_command_receipt",
                    "nominal_motion_time_from_frame0_s": hit / fps,
                    "note": "The legacy READY prelude is not part of motion frame time; 0.6 s must not be sent as live time-to-strike.",
                },
                "canonical_motion_label_b0_before_repair": original_label,
                "legacy_calibrated_control_anchor_offset_b_m": [-0.0343194, 0.0407395, -0.0581275],
                "adapted_reference_hit_state_b0": adapted_reference,
                "actual_execution_hit_state": None,
            }
            # ``strike_target_b0`` is the canonical P1 task contract, not an
            # execution measurement.  The repaired TCP state is deliberately
            # stored in its own layer above: overwriting the task target with
            # it would relabel a tracking error as a new strike goal.
            output_entry["strike_target_b0"] = original_label
            output_entry["p4b_repair"] = {
                "version": "p4b_canonical_prior_materialization/v1",
                "source_motion_sha256": _sha256(source_path),
                "repair_candidate": str(args.repair_candidate.resolve()),
                "repair_candidate_sha256": _sha256(args.repair_candidate),
                "body_state_regenerated": True,
                "upper_momentum_regenerated": True,
                "training_approved": False,
            }
            # Keep the transformation auditable without relying on a
            # directory name or implicit hit-frame equivalence.  This travels
            # through scene placement into the formal PhysX package.
            output_entry["repair_provenance"] = {
                "repair_version": (
                    repair_audit.get("schema_version")
                    if repair_audit is not None
                    else "p4b_deterministic_prior_repair/v1"
                ),
                "original_motion_npz": str(source_path.resolve()),
                "original_motion_sha256": _sha256(source_path),
                "repaired_motion_npz": None,
                "repaired_motion_sha256": None,
                "original_hit_frame": hit,
                "repaired_hit_frame": hit,
                "coordinate_frame": CANONICAL_CONTRACT,
                "tcp_definition": {
                    "body": "right_wrist_yaw_Link",
                    "mount_offset_local_m": MOUNT_OFFSET.tolist(),
                    "normal_axis": "+Y",
                },
                "repair_constraints": (
                    repair_audit.get("repair") if repair_audit is not None else None
                ),
            }

        destination = output_motion_dir / f"motion_{int(entry['motion_id']):02d}_{entry['episode_id']}.npz"
        np.savez_compressed(destination, **arrays)
        output_entry["canonical_motion_npz"] = str(destination.relative_to(args.output_dir))
        output_entry["canonical_motion_sha256"] = _sha256(destination)
        if int(entry["motion_id"]) == args.motion_id:
            output_entry["repair_provenance"]["repaired_motion_npz"] = str(destination.resolve())
            output_entry["repair_provenance"]["repaired_motion_sha256"] = _sha256(destination)
        output_entries.append(output_entry)

    output_manifest["motions"] = output_entries
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "manifest.json"
    output_path.write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
