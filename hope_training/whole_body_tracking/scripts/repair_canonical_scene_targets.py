#!/usr/bin/env python3
"""Rewrite scene-placed manifest strike targets from canonical b0 payloads.

Some generated entries retained a source-scene ``strike_target`` after their
body trajectory was rigidly placed into the P5 scene.  The NPZ canonical goal
is authoritative: transform it from the immutable initial-base-heading frame
to world using the stored scene anchor, then update the manifest metadata.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


def _rotate_z(vector: np.ndarray, heading_rad: float) -> np.ndarray:
    c, s = float(np.cos(heading_rad)), float(np.sin(heading_rad))
    x, y, z = [float(v) for v in np.asarray(vector).reshape(3)]
    return np.asarray([c * x - s * y, s * x + c * y, z], dtype=np.float64)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    output_path = args.output_manifest.expanduser().resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    repaired = copy.deepcopy(payload)
    repaired_rows = []
    alignment_rows = []
    repaired_count = 0
    max_reference_fk_error = 0.0
    mount_offset = np.asarray((0.210211399202899, 0.0320784994676765, 0.0320358706296689))

    for global_motion_id, entry in enumerate(payload.get("motions", [])):
        row = copy.deepcopy(entry)
        # The two source banks each used a local 0..98 id.  Runtime sampling
        # is list-index based, so make the public manifest id unambiguously
        # global as well.
        row["motion_id"] = global_motion_id
        motion_path = Path(str(entry["motion_npz"])).expanduser()
        if not motion_path.is_absolute():
            motion_path = (source_path.parent / motion_path).resolve()
        with np.load(motion_path, allow_pickle=False) as data:
            required = {
                "canonical_goal_position_b0_m",
                "canonical_goal_linear_velocity_b0_mps",
                "canonical_goal_normal_b0",
                "scene_root_anchor_w_m",
            }
            if not required.issubset(set(data.files)):
                repaired_rows.append(row)
                continue
            anchor = np.asarray(data["scene_root_anchor_w_m"], dtype=np.float64).reshape(3)
            heading = float(np.asarray(data.get("scene_root_heading_w_rad", [0.0])).reshape(-1)[0])
            position = anchor + _rotate_z(data["canonical_goal_position_b0_m"], heading)
            velocity = _rotate_z(data["canonical_goal_linear_velocity_b0_mps"], heading)
            normal = _rotate_z(data["canonical_goal_normal_b0"], heading)
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)

            target = copy.deepcopy(row.get("strike_target", {}))
            old_position = np.asarray(target.get("racket_position_m", position), dtype=np.float64)
            old_ball = target.get("ball_position_m")
            if old_ball is not None:
                ball_offset = np.asarray(old_ball, dtype=np.float64) - old_position
                target["ball_position_m"] = (position + ball_offset).tolist()
            target["racket_position_m"] = position.tolist()
            target["racket_velocity_mps"] = velocity.tolist()
            target["racket_normal_w"] = normal.tolist()
            speed = float(np.linalg.norm(velocity))
            if speed > 1.0e-12:
                target["racket_velocity_direction_w"] = (velocity / speed).tolist()
            target["target_frame_contract"] = "scene_world_from_initial_base_heading_canonical_v1"
            row["strike_target"] = target
            row["strike_target_repair"] = {
                "source": "motion_npz.canonical_goal_*_b0",
                "scene_anchor_w_m": anchor.tolist(),
                "scene_heading_w_rad": heading,
                "old_source_scene_racket_position_w_m": old_position.tolist(),
                "new_scene_racket_position_w_m": position.tolist(),
            }
            # Independent FK sanity check at the hit frame.  Body 31 is the
            # right wrist-yaw link; this is the same fixed mount used by the
            # runtime racket FK fallback.
            hit = int(row.get("hit_event", {}).get("motion_hit_frame", 0))
            body_pos = np.asarray(data["body_pos_w"])[hit, 31]
            body_quat = np.asarray(data["body_quat_w"])[hit, 31]
            # Quaternion is stored wxyz in the motion payload.
            w, x, y, z = [float(v) for v in body_quat]
            qv = np.asarray([x, y, z])
            tcp = body_pos + (2.0 * np.dot(qv, mount_offset) * qv + (w * w - np.dot(qv, qv)) * mount_offset + 2.0 * w * np.cross(qv, mount_offset))
            max_reference_fk_error = max(max_reference_fk_error, float(np.linalg.norm(tcp - position)))
            alignment_rows.append({
                "motion_id": global_motion_id,
                "episode_id": row.get("episode_id"),
                "stroke_type": row.get("stroke_type"),
                "hit_frame": hit,
                "scene_anchor_w_m": anchor.tolist(),
                "canonical_target_position_w_m": position.tolist(),
                "manifest_target_position_w_m": row["strike_target"]["racket_position_m"],
                "fk_target_error_m": float(np.linalg.norm(tcp - position)),
                "target_self_error_m": 0.0,
            })
            repaired_count += 1
        repaired_rows.append(row)

    repaired["motions"] = repaired_rows
    repaired["schema_version"] = "upright_forehand_backhand_scene_placed_reference_bank/v2"
    repaired["canonical_target_contract"] = "motion_npz_canonical_b0_plus_scene_anchor"
    repaired["canonical_target_repaired_count"] = repaired_count
    repaired["canonical_target_max_reference_fk_error_m"] = max_reference_fk_error
    repaired["motion_id_contract"] = "global_manifest_list_index_v1"
    repaired["alignment_audit"] = "alignment_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(repaired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_path.parent / "alignment_audit.json").write_text(
        json.dumps({
            "manifest": str(output_path),
            "motion_count": len(alignment_rows),
            "forehand_count": sum(x["stroke_type"] == "forehand" for x in alignment_rows),
            "backhand_count": sum(x["stroke_type"] == "backhand" for x in alignment_rows),
            "max_fk_target_error_m": max_reference_fk_error,
            "max_manifest_target_self_error_m": 0.0,
            "rows": alignment_rows,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_manifest": str(output_path),
        "motion_count": len(repaired_rows),
        "repaired_count": repaired_count,
        "max_reference_fk_error_m": max_reference_fk_error,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
