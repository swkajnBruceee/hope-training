#!/usr/bin/env python3
"""Build the audited 10-D goal index used by V1.3B teacher rollouts.

This is intentionally an offline index builder.  It never treats fixed-base
reference joint trajectories as teacher actions.  The teacher action label is
created later by a PhysX rollout of the frozen model_3396/model_900/model_5000
composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


GOAL_KEYS = ("position_m", "linear_velocity_mps", "normal_w", "time_to_hit_s")


def _resolve(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("motions"), list):
        raise ValueError(f"invalid motion manifest: {path}")
    return data


def _validate_entry(entry: dict[str, Any], manifest_path: Path, index: int) -> dict[str, Any]:
    motion_path_value = entry.get("motion_npz") or entry.get("library_motion_npz")
    if not motion_path_value:
        raise ValueError(f"{manifest_path}: entry {index} has no motion_npz")
    motion_path = _resolve(str(motion_path_value), manifest_path)
    if not motion_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: missing motion file {motion_path}")

    goal = entry.get("canonical_goal_10d")
    if not isinstance(goal, dict):
        target = entry.get("strike_target", {})
        event = entry.get("hit_event", {})
        goal = {
            "position_m": target.get("racket_position_m"),
            "linear_velocity_mps": target.get("racket_velocity_mps"),
            "normal_w": target.get("racket_normal_w"),
            "time_to_hit_s": event.get("strike_time_s"),
        }
    missing = [key for key in GOAL_KEYS if key not in goal]
    if missing:
        raise ValueError(f"{manifest_path}: entry {index} missing goal fields {missing}")

    position = np.asarray(goal["position_m"], dtype=np.float32)
    velocity = np.asarray(goal["linear_velocity_mps"], dtype=np.float32)
    normal = np.asarray(goal["normal_w"], dtype=np.float32)
    time_to_hit = float(goal["time_to_hit_s"])
    if position.shape != (3,) or velocity.shape != (3,) or normal.shape != (3,):
        raise ValueError(f"{manifest_path}: entry {index} goal vectors must be 3-D")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all() or not np.isfinite(normal).all():
        raise ValueError(f"{manifest_path}: entry {index} goal contains non-finite values")
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(time_to_hit) or time_to_hit <= 0.0 or abs(normal_norm - 1.0) > 2.0e-3:
        raise ValueError(
            f"{manifest_path}: entry {index} invalid goal norm/time "
            f"(||normal||={normal_norm}, time={time_to_hit})"
        )

    with np.load(motion_path, allow_pickle=False) as payload:
        required = ("fps", "joint_pos", "body_pos_w")
        missing_payload = [key for key in required if key not in payload.files]
        if missing_payload:
            raise ValueError(f"{motion_path}: missing payload fields {missing_payload}")
        fps = int(np.asarray(payload["fps"]).reshape(-1)[0])
        joint_shape = list(np.asarray(payload["joint_pos"]).shape)
        body_shape = list(np.asarray(payload["body_pos_w"]).shape)
    if fps != 50:
        raise ValueError(f"{motion_path}: fps={fps}, expected 50")
    if len(joint_shape) != 2 or joint_shape[-1] != 31:
        raise ValueError(f"{motion_path}: joint_pos shape={joint_shape}, expected [T,31]")
    if len(body_shape) != 3 or body_shape[-2:] != [32, 3]:
        raise ValueError(f"{motion_path}: body_pos_w shape={body_shape}, expected [T,32,3]")

    return {
        "episode_id": str(entry.get("episode_id", entry.get("motion_id", index))),
        "motion_id": str(entry.get("motion_id", entry.get("episode_id", index))),
        "stroke_type": str(entry.get("stroke_type", "unknown")).lower(),
        "motion_npz": str(motion_path),
        "fps": fps,
        "joint_pos_shape": joint_shape,
        "body_pos_w_shape": body_shape,
        "hit_event": dict(entry.get("hit_event", {})),
        "goal_10d": {
            "target_position_m": position.tolist(),
            "target_linear_velocity_mps": velocity.tolist(),
            "target_normal_w": (normal / normal_norm).tolist(),
            "signed_time_to_hit_s": time_to_hit,
        },
        "source_goal_id": entry.get("source_goal_id"),
        "dataset_role": entry.get("dataset_role", "unknown"),
        "sample_weight": float(entry.get("sample_weight", 1.0)),
        "split_group_id": entry.get("split_group_id"),
        "teacher_rollout_status": "pending_physx_rollout",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = (("training", args.training_manifest), ("validation", args.validation_manifest))
    splits: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for split_name, path_value in manifests:
        path = path_value.expanduser().resolve()
        manifest = _load_manifest(path)
        rows = [_validate_entry(entry, path, index) for index, entry in enumerate(manifest["motions"])]
        splits[split_name] = rows
        counts[split_name] = {
            "motions": len(rows),
            "forehand": sum(row["stroke_type"] == "forehand" for row in rows),
            "backhand": sum(row["stroke_type"] == "backhand" for row in rows),
        }

    all_rows = splits["training"] + splits["validation"]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v13b_teacher_distillation_goal_index/v1",
        "status": "goals_audited_teacher_rollout_pending",
        "teacher_contract": {
            "lower_checkpoint": "checkpoints/frozen_priors/model_3396.pt",
            "upper_checkpoint": "checkpoints/frozen_priors/model_900.pt",
            "student_checkpoint": "<model_5000_complete_priors>",
            "lower_alpha": 1.0,
            "upper_alpha": 0.9,
            "student_observation_contract": "98D_public_state_plus_goal_10d",
            "student_action_contract": "26D",
        },
        "source_manifests": {name: str(path.expanduser().resolve()) for name, path in manifests},
        "counts": counts,
        "goal_contract": [
            "target_position_m",
            "target_linear_velocity_mps",
            "target_normal_w",
            "signed_time_to_hit_s",
        ],
        "splits": splits,
        "index_sha256": _sha256_json(all_rows),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "counts": counts, "status": payload["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
