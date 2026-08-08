#!/usr/bin/env python3
"""Audit generated A3 candidate TCPs against their canonical strike targets.

The offline IK goal is stored in the immutable initial-base-heading frame,
which is the A3 root-relative frame for this fixed-base bank.  This audit
reconstructs the racket-center TCP from the stored ``right_wrist_yaw_Link``
pose and the validated A3 mount offset, then compares it after transforming
the world point back into the root frame.  It also reports the deliberately
wrong world-vs-relative comparison so a frame mistake cannot hide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MOUNT_OFFSET = np.asarray(
    (0.210211399202899, 0.0320784994676765, 0.0320358706296689), dtype=np.float64
)
ROOT_BODY_INDEX = 0
RIGHT_WRIST_YAW_BODY_INDEX = 31


def _quat_apply(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    qv = q[1:]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q[0] * v)


def _quat_conjugate_apply(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return _quat_apply(np.asarray((q[0], -q[1], -q[2], -q[3])), vector)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means all motions.")
    parser.add_argument("--keep-rows", action="store_true", help="Include every row in the JSON report.")
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(manifest.get("motions", []))
    if args.max_candidates:
        entries = entries[: args.max_candidates]
    if not entries:
        raise ValueError("manifest has no motions")

    rows: list[dict] = []
    errors: list[dict] = []
    source_mismatch: list[dict] = []
    relative_errors: list[float] = []
    world_errors: list[float] = []
    root_quat_errors: list[float] = []
    root_positions: list[np.ndarray] = []
    for entry in entries:
        episode_id = str(entry["episode_id"])
        motion_path = Path(str(entry["motion_npz"]))
        if not motion_path.is_absolute():
            motion_path = manifest_dir / motion_path
        try:
            with np.load(motion_path, allow_pickle=False) as data:
                required = {"joint_pos", "hit_frame"}
                missing = sorted(required - set(data.files))
                if missing:
                    raise ValueError(f"missing keys {missing}")
                hit = int(np.asarray(data["hit_frame"]).reshape(-1)[0])
                if {"body_pos_b0", "body_quat_b0_wxyz", "canonical_goal_position_b0_m"}.issubset(data.files):
                    body_pos_key = "body_pos_b0"
                    body_quat_key = "body_quat_b0_wxyz"
                    target_key = "canonical_goal_position_b0_m"
                    coordinate_source = "stored canonical base-heading b0 arrays"
                elif {"body_pos_w", "body_quat_w", "canonical_position"}.issubset(data.files):
                    body_pos_key = "body_pos_w"
                    body_quat_key = "body_quat_w"
                    target_key = "canonical_position"
                    coordinate_source = "stored world arrays transformed to root frame"
                else:
                    raise ValueError("NPZ has neither canonical b0 nor candidate world TCP contract")
                body_pos = np.asarray(data[body_pos_key], dtype=np.float64)
                body_quat = np.asarray(data[body_quat_key], dtype=np.float64)
                if body_pos.ndim != 3 or body_pos.shape[1:] != (32, 3):
                    raise ValueError(f"body_pos_w shape={body_pos.shape}")
                if body_quat.shape != (body_pos.shape[0], 32, 4):
                    raise ValueError(f"body_quat_w shape={body_quat.shape}")
                if not 0 <= hit < body_pos.shape[0]:
                    raise ValueError(f"hit_frame={hit} outside frames={body_pos.shape[0]}")
                root_pos = body_pos[hit, ROOT_BODY_INDEX]
                root_quat = body_quat[hit, ROOT_BODY_INDEX]
                wrist_pos = body_pos[hit, RIGHT_WRIST_YAW_BODY_INDEX]
                wrist_quat = body_quat[hit, RIGHT_WRIST_YAW_BODY_INDEX]
                tcp_world = wrist_pos + _quat_apply(wrist_quat, MOUNT_OFFSET)
                tcp_root = _quat_conjugate_apply(root_quat, tcp_world - root_pos)
                if target_key == "canonical_goal_position_b0_m":
                    target = np.asarray(data[target_key], dtype=np.float64).reshape(3)
                    manifest_target = np.asarray(entry["canonical_goal_10d"]["position_b0_m"], dtype=np.float64).reshape(3)
                else:
                    target = np.asarray(entry["strike_target"]["racket_position_m"], dtype=np.float64).reshape(3)
                    manifest_target = np.asarray(data[target_key], dtype=np.float64).reshape(3)
                relative_delta = tcp_root - target
                world_comparison_valid = target_key != "canonical_goal_position_b0_m"
                world_delta = tcp_world - target if world_comparison_valid else None
                source_target = None
                source_path_value = entry.get("source_npz")
                if source_path_value:
                    source_path = Path(str(source_path_value)).expanduser()
                    if source_path.is_file():
                        with np.load(source_path, allow_pickle=False) as source:
                            source_name = "canonical_position" if "canonical_position" in source.files else "canonical_goal_position_b0_m"
                            source_target = np.asarray(source[source_name], dtype=np.float64).reshape(3)
                        if np.linalg.norm(source_target - target) > 1.0e-6:
                            source_mismatch.append(
                                {
                                    "episode_id": episode_id,
                                    "manifest_target_m": target.tolist(),
                                    "source_target_m": source_target.tolist(),
                                    "norm_m": float(np.linalg.norm(source_target - target)),
                                }
                            )
                root_quat_error = abs(float(np.linalg.norm(root_quat)) - 1.0)
                relative_norm = float(np.linalg.norm(relative_delta))
                world_norm = float(np.linalg.norm(world_delta)) if world_delta is not None else None
                relative_errors.append(relative_norm)
                if world_norm is not None:
                    world_errors.append(world_norm)
                root_quat_errors.append(root_quat_error)
                root_positions.append(root_pos)
                row = {
                    "episode_id": episode_id,
                    "stroke_type": entry.get("stroke_type"),
                    "dataset_role": entry.get("dataset_role"),
                    "hit_frame": hit,
                    "root_position_w_m": root_pos.tolist(),
                    "root_quaternion_wxyz": root_quat.tolist(),
                    "tcp_world_m": tcp_world.tolist(),
                    "tcp_root_initial_heading_m": tcp_root.tolist(),
                    "target_root_initial_heading_m": target.tolist(),
                    "relative_error_xyz_m": relative_delta.tolist(),
                    "relative_error_norm_m": relative_norm,
                    "wrong_world_vs_target_error_norm_m": world_norm,
                    "source_target_m": source_target.tolist() if source_target is not None else None,
                    "npz_target_m": manifest_target.tolist(),
                    "coordinate_source": coordinate_source,
                    "root_quaternion_norm_error": root_quat_error,
                }
                if args.keep_rows:
                    rows.append(row)
                elif relative_norm > 0.30 or (world_norm is not None and world_norm < 0.30):
                    # Keep suspicious rows even in compact mode.  A small
                    # world error would indicate a different target-frame
                    # interpretation and deserves direct inspection.
                    rows.append(row)
        except Exception as exc:
            errors.append({"episode_id": episode_id, "error": f"{type(exc).__name__}: {exc}"})

    rel = np.asarray(relative_errors, dtype=np.float64)
    world = np.asarray(world_errors, dtype=np.float64)
    summary = {
        "schema_version": "a3_candidate_tcp_alignment_audit/v1",
        "status": "completed" if not errors else "completed_with_errors",
        "manifest": str(manifest_path),
        "evaluated_count": len(relative_errors),
        "error_count": len(errors),
        "source_target_mismatch_count": len(source_mismatch),
        "coordinate_contract": {
            "target_frame": "initial_base_heading/root-relative (or stored canonical b0 equivalent)",
            "root_body_index": ROOT_BODY_INDEX,
            "root_body_name": "pelvis_link",
            "tcp_body_index": RIGHT_WRIST_YAW_BODY_INDEX,
            "tcp_body_name": "right_wrist_yaw_Link",
            "tcp_definition": "right_wrist_yaw_Link pose plus A3_MOUNT_OFFSET",
            "mount_offset_m": MOUNT_OFFSET.tolist(),
            "quaternion_order": "wxyz",
            "comparison": "inverse(root_quat) * (tcp_world - root_position) vs target",
        },
        "relative_root_error_m": {
            "max": float(rel.max()) if rel.size else float("nan"),
            "mean": float(rel.mean()) if rel.size else float("nan"),
            "p50": _percentile(relative_errors, 50),
            "p95": _percentile(relative_errors, 95),
            "p99": _percentile(relative_errors, 99),
            "count_gt_0.03": int(np.sum(rel > 0.03)),
            "count_gt_0.30": int(np.sum(rel > 0.30)),
            "count_gt_0.60": int(np.sum(rel > 0.60)),
        },
        "wrong_world_frame_error_m": {
            "applicable": bool(world.size),
            "max": float(world.max()) if world.size else None,
            "mean": float(world.mean()) if world.size else None,
            "p50": _percentile(world_errors, 50) if world.size else None,
            "p95": _percentile(world_errors, 95) if world.size else None,
            "note": "Only applicable when the NPZ stores world arrays and the target is root-relative.",
        },
        "root_position_range_w_m": {
            "min": np.min(np.stack(root_positions), axis=0).tolist() if root_positions else None,
            "max": np.max(np.stack(root_positions), axis=0).tolist() if root_positions else None,
        },
        "root_quaternion_norm_error_max": max(root_quat_errors) if root_quat_errors else None,
        "errors": errors,
        "source_target_mismatches": source_mismatch,
        "suspicious_rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "evaluated_count", "error_count", "source_target_mismatch_count", "relative_root_error_m", "wrong_world_frame_error_m", "root_quaternion_norm_error_max")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
