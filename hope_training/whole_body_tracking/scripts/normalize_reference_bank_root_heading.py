#!/usr/bin/env python3
"""Normalize a materialized A3 reference bank to one explicit root heading.

The old mixed forehand/backhand bank contains forehand clips with a 180 degree
root yaw and backhand clips with identity yaw.  The tracker resets every
episode to one fixed READY root pose, so mixed root headings are invalid.  This
tool rotates each payload into the requested identity-root scene frame and
updates the per-motion target fields by the same rigid transform.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np


ANCHOR = np.asarray([-0.5, -0.7625, 1.04], dtype=np.float64)
MOUNT_OFFSET = np.asarray((0.210211399202899, 0.0320784994676765, 0.0320358706296689), dtype=np.float64)
IDENTITY_WXYZ = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
PI_YAW_WXYZ = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
R_PI = np.asarray(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def _qmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _normalize_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-12)
    # Quaternion sign is immaterial; choose the first numerically nonzero
    # component positive for deterministic audits (this also handles pure
    # pi-yaw quaternions whose w component is zero).
    flat = q.reshape(-1, 4)
    for row in flat:
        for value in row:
            if abs(float(value)) > 1.0e-8:
                if value < 0.0:
                    row *= -1.0
                break
    return q


def _rotate(v: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", R_PI, np.asarray(v, dtype=np.float64))


def _rotate_by(rotation: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", rotation, np.asarray(v, dtype=np.float64))


def _world_point(p: np.ndarray) -> np.ndarray:
    return ANCHOR + _rotate(np.asarray(p, dtype=np.float64) - ANCHOR)


def _world_quat_xyzw(q_xyzw: list[float]) -> list[float]:
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64)
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    q_new = _normalize_q(_qmul(PI_YAW_WXYZ, q_wxyz))
    return q_new[[1, 2, 3, 0]].tolist()


def _transform_payload(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}

    body_pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(arrays["body_quat_w"], dtype=np.float64)
    root_q = _normalize_q(body_quat[0, 0])
    if min(np.linalg.norm(root_q - IDENTITY_WXYZ), np.linalg.norm(root_q + IDENTITY_WXYZ)) < 1.0e-5:
        applied = "identity"
        q_delta = IDENTITY_WXYZ
        rotation = np.eye(3, dtype=np.float64)
    elif min(np.linalg.norm(root_q - PI_YAW_WXYZ), np.linalg.norm(root_q + PI_YAW_WXYZ)) < 1.0e-5:
        applied = "root_yaw_pi"
        q_delta = PI_YAW_WXYZ
        rotation = R_PI
    else:
        raise ValueError(f"{path}: unsupported root quaternion {root_q.tolist()}")

    root = body_pos[:, 0].copy()
    body_pos = root[:, None, :] + _rotate_by(rotation, body_pos - root[:, None, :])
    body_quat = _normalize_q(_qmul(q_delta, body_quat))
    arrays["body_pos_w"] = body_pos.astype(np.float32)
    arrays["body_quat_w"] = body_quat.astype(np.float32)
    arrays["body_lin_vel_w"] = _rotate_by(rotation, arrays["body_lin_vel_w"]).astype(np.float32)
    arrays["body_ang_vel_w"] = _rotate_by(rotation, arrays["body_ang_vel_w"]).astype(np.float32)
    arrays["body_pos_b0"] = (body_pos - root[:, None, :]).astype(np.float32)
    arrays["body_quat_b0_wxyz"] = body_quat.astype(np.float32)
    arrays["body_lin_vel_b0"] = arrays["body_lin_vel_w"].copy()
    arrays["body_ang_vel_b0"] = arrays["body_ang_vel_w"].copy()

    if applied == "root_yaw_pi":
        if "canonical_goal_position_b0_m" in arrays:
            arrays["canonical_goal_position_b0_m"] = _rotate(arrays["canonical_goal_position_b0_m"]).astype(np.float64)
        for key in ("canonical_goal_normal_b0", "canonical_goal_linear_velocity_b0_mps"):
            if key in arrays:
                arrays[key] = _rotate(arrays[key]).astype(np.float64)
    arrays["scene_root_anchor_w_m"] = ANCHOR.copy()
    arrays["scene_root_heading_w_rad"] = np.asarray([0.0], dtype=np.float64)

    audit = {
        "source_root_quaternion_wxyz": root_q.tolist(),
        "applied_transform": applied,
        "output_root_quaternion_wxyz": _normalize_q(arrays["body_quat_w"][0, 0]).tolist(),
        "rotation_matrix": rotation.tolist(),
    }
    return arrays, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_manifest = args.source_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_dir = output_dir / "motion_npz"
    motion_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    rows = []
    audits = []
    max_fk_error = 0.0

    for entry in source["motions"]:
        source_path = Path(str(entry["motion_npz"])).expanduser().resolve()
        arrays, transform_audit = _transform_payload(source_path)
        stem = str(entry["episode_id"])
        output_path = motion_dir / f"{stem}.npz"
        np.savez_compressed(output_path, **arrays)

        row = copy.deepcopy(entry)
        row["motion_npz"] = str(output_path)
        target = copy.deepcopy(row.get("strike_target", {}))
        if transform_audit["applied_transform"] == "root_yaw_pi":
            for key in ("racket_position_m", "ball_position_m"):
                if key in target:
                    target[key] = _world_point(target[key]).tolist()
            for key in (
                "racket_velocity_mps",
                "racket_normal_w",
                "racket_tangent_w",
                "racket_velocity_direction_w",
                "ball_in_velocity_mps",
                "ball_out_velocity_mps",
            ):
                if key in target:
                    target[key] = _rotate(target[key]).tolist()
            if "racket_quat_xyzw" in target:
                target["racket_quat_xyzw"] = _world_quat_xyzw(target["racket_quat_xyzw"])
        row["strike_target"] = target
        row["scene_root_contract"] = {
            "root_position_w_m": ANCHOR.tolist(),
            "root_quaternion_wxyz": IDENTITY_WXYZ.tolist(),
            "root_heading_w_rad": 0.0,
            "normalization": "all motions rigidly normalized to one identity-root frame",
        }
        rows.append(row)

        hit = int(row.get("hit_event", {}).get("motion_hit_frame", 0))
        wrist = np.asarray(arrays["body_pos_w"])[hit, 31].astype(np.float64)
        q = np.asarray(arrays["body_quat_w"])[hit, 31].astype(np.float64)
        w, x, y, z = q
        qv = np.asarray([x, y, z])
        offset = 2.0 * np.dot(qv, MOUNT_OFFSET) * qv + (w * w - np.dot(qv, qv)) * MOUNT_OFFSET + 2.0 * w * np.cross(qv, MOUNT_OFFSET)
        fk_tcp = wrist + offset
        canonical_target = ANCHOR + np.asarray(arrays["canonical_goal_position_b0_m"], dtype=np.float64)
        fk_error = float(np.linalg.norm(fk_tcp - canonical_target))
        max_fk_error = max(max_fk_error, fk_error)
        audits.append({
            "motion_id": int(row.get("motion_id", len(audits))),
            "episode_id": stem,
            "stroke_type": row.get("stroke_type"),
            "source_root_quaternion_wxyz": transform_audit["source_root_quaternion_wxyz"],
            "output_root_quaternion_wxyz": transform_audit["output_root_quaternion_wxyz"],
            "hit_frame": hit,
            "fk_target_error_m": fk_error,
        })

    output = copy.deepcopy(source)
    output["schema_version"] = "upright_forehand_backhand_scene_placed_v4_identity_root"
    output["source_manifest"] = str(source_manifest)
    output["motions"] = rows
    output["root_frame_contract"] = {
        "root_position_w_m": ANCHOR.tolist(),
        "root_quaternion_wxyz": IDENTITY_WXYZ.tolist(),
        "root_heading_w_rad": 0.0,
        "all_motion_root_quaternions_normalized": True,
    }
    output["dataset_status"] = "candidate_identity_root_normalized_requires_physx_replay"
    (output_dir / "tracking_motion_manifest.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "manifest": str(output_dir / "tracking_motion_manifest.json"),
        "motion_count": len(audits),
        "forehand_count": sum(x["stroke_type"] == "forehand" for x in audits),
        "backhand_count": sum(x["stroke_type"] == "backhand" for x in audits),
        "max_fk_target_error_m": max_fk_error,
        "max_root_quaternion_error": max(
            float(np.linalg.norm(np.asarray(x["output_root_quaternion_wxyz"]) - IDENTITY_WXYZ))
            for x in audits
        ),
        "rows": audits,
    }
    (output_dir / "root_heading_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: audit[k] for k in ("motion_count", "forehand_count", "backhand_count", "max_fk_target_error_m", "max_root_quaternion_error")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
