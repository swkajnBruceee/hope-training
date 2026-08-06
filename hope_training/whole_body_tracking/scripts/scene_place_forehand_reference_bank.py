#!/usr/bin/env python3
"""Place the generated forehand bank in the P5 scene root frame.

The forehand generator preserves the source retargeting world root, while the
P5 floating tracker expects the materialized root ``[-0.5, -0.7625, 1.04]``.
This tool applies only a rigid translation to ``body_pos_w``; joint poses,
body-relative targets, velocities and orientations are unchanged.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


TARGET_ROOT = np.asarray([-0.5, -0.7625, 1.04], dtype=np.float64)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    out_dir = args.output_dir.expanduser().resolve()
    motion_dir = out_dir / "motion_npz"
    motion_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    audit = []
    for entry in source["motions"]:
        source_motion = Path(str(entry["motion_npz"])).expanduser().resolve()
        with np.load(source_motion, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]).copy() for key in data.files}
        body_pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)
        old_root = body_pos[0, 0].copy()
        delta = TARGET_ROOT - old_root
        body_pos += delta[None, None, :]
        arrays["body_pos_w"] = body_pos.astype(np.float32)
        root_pos = body_pos[:, 0]
        arrays["body_pos_b0"] = (body_pos - root_pos[:, None, :]).astype(np.float32)
        arrays["scene_root_anchor_w_m"] = TARGET_ROOT.copy()
        arrays["scene_root_heading_w_rad"] = np.asarray([0.0])

        stem = str(entry["episode_id"])
        output_motion = motion_dir / f"{stem}.npz"
        np.savez_compressed(output_motion, **arrays)
        placed = copy.deepcopy(entry)
        placed["motion_npz"] = str(output_motion)
        placed["scene_placement"] = {
            "target_root_w_m": TARGET_ROOT.tolist(),
            "source_root_w_m": old_root.tolist(),
            "translation_delta_w_m": delta.tolist(),
            "root_position_error_before_m": float(np.linalg.norm(old_root - TARGET_ROOT)),
            "root_position_error_after_m": float(np.linalg.norm(body_pos[0, 0] - TARGET_ROOT)),
        }
        rows.append(placed)
        audit.append({
            "episode_id": stem,
            "source_root_w_m": old_root.tolist(),
            "placed_root_w_m": body_pos[0, 0].tolist(),
            "target_root_w_m": TARGET_ROOT.tolist(),
            "root_position_error_before_m": float(np.linalg.norm(old_root - TARGET_ROOT)),
            "root_position_error_after_m": float(np.linalg.norm(body_pos[0, 0] - TARGET_ROOT)),
            "max_root_position_error_over_frames_m": float(np.max(np.linalg.norm(body_pos[:, 0] - TARGET_ROOT[None, :], axis=1))),
        })

    payload = copy.deepcopy(source)
    payload["schema_version"] = "upright_forehand_scene_placed_reference_bank/v1"
    payload["source_manifest"] = str(source_path)
    payload["scene_root_contract"] = {
        "root_position_w_m": TARGET_ROOT.tolist(),
        "root_heading_w_rad": 0.0,
        "placement": "rigid_translation_only",
    }
    payload["motions"] = rows
    payload["motion_count"] = len(rows)
    payload["physics_qualified"] = False
    payload["teacher_approved"] = False
    payload["coverage_note"] = "Forehand bank rigidly scene-placed at the P5 floating tracker root; PhysX qualification remains required."
    (out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "scene_placement_audit.json").write_text(json.dumps({"target_root_w_m": TARGET_ROOT.tolist(), "motion_count": len(audit), "rows": audit}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"motion_count": len(rows), "manifest": str(out_dir / "manifest.json"), "audit": str(out_dir / "scene_placement_audit.json"), "max_before_m": max(x["root_position_error_before_m"] for x in audit), "max_after_m": max(x["max_root_position_error_over_frames_m"] for x in audit)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
