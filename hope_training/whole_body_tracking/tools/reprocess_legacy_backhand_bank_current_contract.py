#!/usr/bin/env python3
"""Re-materialize the legacy 96-motion backhand bank under the current A3 contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_upper_momentum_library import UrdfModel  # noqa: E402
from materialize_p4b_repaired_canonical_prior import (  # noqa: E402
    DEFAULT_URDF,
    MOUNT_OFFSET,
    _quat_matrix,
    _regenerate_body_arrays,
    _relative_body_velocity_from_joint_state,
    _tcp_hit_state,
)


CURRENT_ROOT_POS = np.asarray((0.0, 0.0, 1.0684), dtype=np.float64)
CURRENT_ROOT_QUAT = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
MAX_WAIST_FORWARD_TILT_DEG = 20.0
WORKSPACE = {
    "x": (0.38, 0.51),
    "y": (-0.55, 0.14),
    "z": (-0.01, 0.16),
}


def _limits(path: Path) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(path).getroot().iter("joint"):
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            result[joint.attrib["name"]] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    return result


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def _workspace_status(position: np.ndarray) -> str:
    inside = all(WORKSPACE[key][0] <= float(position[i]) <= WORKSPACE[key][1] for i, key in enumerate(("x", "y", "z")))
    return "inside_reviewed_backhand_workspace" if inside else "outside_reviewed_backhand_workspace"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "docs/a3_articulation_metadata.json")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    args = parser.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.expanduser().resolve().read_text(encoding="utf-8"))
    joint_names = list(metadata["joint_names"])
    body_names = list(metadata["body_names"])
    body_index = {name: index for index, name in enumerate(body_names)}
    joint_index = {name: index for index, name in enumerate(joint_names)}
    wrist_index = body_index["right_wrist_yaw_Link"]
    model = UrdfModel(args.urdf)
    limits = _limits(args.urdf)

    output_dir = args.output_dir.expanduser().resolve()
    motion_dir = output_dir / "motion_npz"
    motion_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    audit: list[dict] = []
    rejected: list[dict] = []

    for motion_id, source_entry in enumerate(source["motions"]):
        source_arrays = _load(Path(str(source_entry["motion_npz"])))
        fps = float(np.asarray(source_arrays["fps"]).reshape(-1)[0])
        q = np.asarray(source_arrays["joint_pos"], dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != len(joint_names):
            rejected.append({"episode_id": source_entry["episode_id"], "reason": f"joint_pos_shape={q.shape}"})
            continue
        hit = int(np.asarray(source_arrays["hit_frame"]).reshape(-1)[0])
        if not 0 <= hit < q.shape[0]:
            rejected.append({"episode_id": source_entry["episode_id"], "reason": f"hit_frame={hit}"})
            continue

        waist_pitch_values_deg = np.degrees(q[:, joint_index["waist_pitch_joint"]])
        waist_pitch_min_deg = float(waist_pitch_values_deg.min())
        waist_pitch_max_deg = float(waist_pitch_values_deg.max())
        waist_roll_deg = float(np.degrees(np.max(np.abs(q[:, joint_index["waist_roll_joint"]]))))
        if waist_pitch_min_deg < -1.0e-6 or waist_pitch_max_deg > MAX_WAIST_FORWARD_TILT_DEG or waist_roll_deg > MAX_WAIST_FORWARD_TILT_DEG:
            rejected.append(
                {
                    "episode_id": source_entry["episode_id"],
                    "reason": "waist_pitch_backward_or_forward_tilt_over_20deg",
                    "waist_pitch_min_deg": waist_pitch_min_deg,
                    "waist_pitch_max_deg": waist_pitch_max_deg,
                    "waist_roll_abs_deg": waist_roll_deg,
                }
            )
            continue
        hard_margin = float("inf")
        hard_violation = None
        for name, (lower, upper) in limits.items():
            if name not in joint_index:
                continue
            values = q[:, joint_index[name]]
            margin = min(float(values.min() - lower), float(upper - values.max()))
            if margin < hard_margin:
                hard_margin = margin
                hard_violation = name if margin < 0.0 else None
        if hard_violation is not None:
            rejected.append({"episode_id": source_entry["episode_id"], "reason": "hard_joint_limit", "joint": hard_violation, "margin_rad": hard_margin})
            continue

        qd = np.gradient(q, 1.0 / fps, axis=0)
        root_pos = np.repeat(CURRENT_ROOT_POS[None, :], q.shape[0], axis=0)
        root_quat = np.repeat(CURRENT_ROOT_QUAT[None, :], q.shape[0], axis=0)
        body_pos_w, body_quat_w = _regenerate_body_arrays(model, joint_names, body_names, q, root_pos, root_quat, fps)
        body_lin, body_ang = _relative_body_velocity_from_joint_state(model, joint_names, body_names, q, qd, root_quat)
        body_pos_b0 = body_pos_w - root_pos[:, None, :]
        body_quat_b0 = body_quat_w.copy()
        body_lin_b0 = body_lin.copy()
        body_ang_b0 = body_ang.copy()
        arrays = {
            "fps": np.asarray([fps], dtype=np.float32),
            # The fixed-base replay tool consumes the explicit A3 joint-name
            # contract; keeping it in every materialized NPZ prevents a
            # silent positional remap at the PhysX boundary.
            "joint_names": np.asarray(joint_names),
            "joint_pos": q.astype(np.float32),
            "joint_vel": qd.astype(np.float32),
            "body_pos_w": body_pos_w.astype(np.float32),
            "body_quat_w": body_quat_w.astype(np.float32),
            "body_lin_vel_w": body_lin.astype(np.float32),
            "body_ang_vel_w": body_ang.astype(np.float32),
            "body_pos_b0": body_pos_b0.astype(np.float32),
            "body_quat_b0_wxyz": body_quat_b0.astype(np.float32),
            "body_lin_vel_b0": body_lin_b0.astype(np.float32),
            "body_ang_vel_b0": body_ang_b0.astype(np.float32),
            "hit_frame": np.asarray([hit], dtype=np.int64),
            "physics_qualified": np.asarray([False]),
            "scene_root_anchor_w_m": CURRENT_ROOT_POS.astype(np.float64),
            "scene_root_heading_w_rad": np.asarray([0.0], dtype=np.float64),
            "source_goal_id": np.asarray([f"legacy_backhand_{motion_id:03d}"], dtype="U"),
            "selected_swing_type": np.asarray(["backhand"], dtype="U"),
            "requested_strike_time_s": np.asarray([hit / fps], dtype=np.float64),
        }
        tcp_state = _tcp_hit_state(arrays, wrist_index, hit)
        goal_position = np.asarray(tcp_state["racket_position_b0_m"], dtype=np.float64)
        goal_velocity = np.asarray(tcp_state["racket_velocity_b0_mps"], dtype=np.float64)
        goal_normal = np.asarray(tcp_state["racket_normal_b0"], dtype=np.float64)
        goal_time = float(hit / fps)
        arrays["canonical_position"] = goal_position
        arrays["canonical_velocity"] = goal_velocity
        arrays["canonical_normal"] = goal_normal
        arrays["canonical_strike_time_s"] = np.asarray([goal_time], dtype=np.float64)
        stem = f"legacy_backhand_current_{motion_id:03d}"
        motion_path = motion_dir / f"{stem}.npz"
        np.savez_compressed(motion_path, **arrays)

        entry = copy.deepcopy(source_entry)
        entry.update(
            {
                "episode_id": stem,
                "motion_id": motion_id,
                "motion_npz": str(motion_path),
                "stroke_type": "backhand",
                "canonical_motion_npz": True,
                "canonical_goal_10d": {
                    "position_m": goal_position.tolist(),
                    "normal_w": goal_normal.tolist(),
                    "linear_velocity_mps": goal_velocity.tolist(),
                    "time_to_hit_s": goal_time,
                },
                "strike_target": {
                    "racket_position_m": goal_position.tolist(),
                    "racket_velocity_mps": goal_velocity.tolist(),
                    "racket_normal_w": goal_normal.tolist(),
                },
                "coordinate_contract": "current_root_relative_initial_heading",
                "workspace_status": _workspace_status(goal_position),
                "waist_contract": {
                    "waist_yaw": "preserved",
                    "waist_roll_abs_max_deg": waist_roll_deg,
                    "waist_pitch_forward_tilt_min_deg": waist_pitch_min_deg,
                    "waist_pitch_forward_tilt_max_deg": waist_pitch_max_deg,
                    "waist_pitch_direction": "forward_only_nonnegative_joint_pitch",
                    "backward_tilt_allowed": False,
                    "forward_tilt_limit_deg": MAX_WAIST_FORWARD_TILT_DEG,
                },
                "fixed_base_physx_status": "PENDING",
                "physics_qualified": False,
                "teacher_approved": False,
                "training_admission": False,
                "source_legacy_episode_id": source_entry["episode_id"],
            }
        )
        entry.pop("library_motion_npz", None)
        entry.pop("canonical_motion_npz_before_canonicalization", None)
        rows.append(entry)
        audit.append(
            {
                "episode_id": stem,
                "source_legacy_episode_id": source_entry["episode_id"],
                "hit_frame": hit,
                "goal_position_b0_m": goal_position.tolist(),
                "goal_velocity_b0_mps": goal_velocity.tolist(),
                "goal_normal_b0": goal_normal.tolist(),
                "goal_time_s": goal_time,
                "waist_yaw_abs_max_deg": float(np.degrees(np.max(np.abs(q[:, joint_index["waist_yaw_joint"]])))),
                "waist_roll_abs_max_deg": waist_roll_deg,
                "waist_pitch_forward_tilt_min_deg": waist_pitch_min_deg,
                "waist_pitch_forward_tilt_max_deg": waist_pitch_max_deg,
                "backward_tilt_allowed": False,
                "hard_joint_margin_rad": hard_margin,
                "workspace_status": _workspace_status(goal_position),
                "tcp_linkage_error_m": 0.0,
            }
        )

    payload = {
        "schema_version": "a3_legacy_backhand_current_contract/v1",
        "status": "candidate_only_physx_replay_required",
        "training_role": "legacy_backhand_supplement_candidate",
        "teacher_approved": False,
        "physics_qualified": False,
        "training_admission": False,
        "source_manifest": str(source_path),
        "motion_count": len(rows),
        "rejected_count": len(rejected),
        "coordinate_contract": "current_root_relative_initial_heading",
        "root_pose_contract": {"root_position_w_m": CURRENT_ROOT_POS.tolist(), "root_quaternion_wxyz": CURRENT_ROOT_QUAT.tolist()},
        "tcp_contract": {"body": "right_wrist_yaw_Link", "mount_offset_local_m": MOUNT_OFFSET.tolist(), "normal_axis": "+Y"},
        "waist_contract": {
            "waist_yaw": "preserved",
            "waist_roll": "preserved_within_20deg",
            "waist_pitch": "forward_only_nonnegative_joint_pitch",
            "backward_tilt_allowed": False,
            "forward_tilt_limit_deg": MAX_WAIST_FORWARD_TILT_DEG,
        },
        "workspace_contract": WORKSPACE,
        "motions": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "reprocessing_audit.json").write_text(json.dumps({"schema_version": "a3_legacy_backhand_reprocessing_audit/v1", "source_manifest": str(source_path), "processed_count": len(rows), "rejected_count": len(rejected), "rows": audit, "rejected": rejected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"processed": len(rows), "rejected": len(rejected), "inside_workspace": sum(x["workspace_status"] == "inside_reviewed_backhand_workspace" for x in audit), "outside_workspace": sum(x["workspace_status"] != "inside_reviewed_backhand_workspace" for x in audit), "manifest": str(output_dir / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
