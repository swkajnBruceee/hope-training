#!/usr/bin/env python3
"""Build a held-out manifest mixing fixed and prepositioned strike modes.

The source manifests are never modified. Fixed-mode entries receive explicit
metadata derived from their NPZ root pose and calibrated strike target so the
same contract can validate both modes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np


def _yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_inverse_z(vector: np.ndarray, yaw: float) -> list[float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [float(c * vector[0] + s * vector[1]), float(-s * vector[0] + c * vector[1]), float(vector[2])]


def _resolve(path: str | Path, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    # Existing manifests use both repository-relative paths and paths relative
    # to the manifest directory. Prefer the repository-relative path when this
    # tool is invoked from the workspace, then fall back to the legacy layout.
    repo_root = Path(__file__).resolve().parents[3]
    for root in (Path.cwd(), repo_root):
        root_candidate = (root / candidate).resolve()
        if root_candidate.exists():
            return root_candidate
    return (base / candidate).resolve()


def _fixed_metadata(entry: dict, manifest_path: Path) -> dict:
    motion_path = _resolve(entry["motion_npz"], manifest_path.parent)
    data = np.load(motion_path)
    hit_frame = int(entry["hit_event"]["motion_hit_frame"])
    root = np.asarray(data["body_pos_w"])[hit_frame, 0].astype(np.float64)
    root_quat = np.asarray(data["body_quat_w"])[hit_frame, 0].astype(np.float64)
    yaw = _yaw_from_wxyz(root_quat)
    hit = np.asarray(entry["strike_target"]["racket_position_m"], dtype=np.float64)
    target_base = _rotate_inverse_z(hit - root, yaw)
    position = [float(x) for x in root]
    return {
        "stance_mode": "fixed",
        "original_hit_position_w_m": [float(x) for x in hit],
        "base_pose_before_w": {"position_m": position, "yaw_rad": yaw},
        "base_pose_target_w": {"position_m": position, "yaw_rad": yaw},
        "stance_offset_xy_w_m": [0.0, 0.0],
        "strike_target_base_m": target_base,
        "joint_limit_source": "native_manifest_gate",
    }


def _normalise_motion_path(entry: dict, manifest_path: Path) -> dict:
    """Make copied entries self-contained for the Isaac manifest loader."""
    result = copy.deepcopy(entry)
    if result.get("motion_npz"):
        result["motion_npz"] = str(_resolve(result["motion_npz"], manifest_path.parent))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forehand-manifest", type=Path, required=True)
    parser.add_argument("--forehand-ids", help="Optional comma-separated forehand episode IDs")
    parser.add_argument("--backhand-manifest", type=Path, required=True)
    parser.add_argument("--backhand-ids", required=True, help="Comma-separated backhand episode IDs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fh_manifest = json.loads(args.forehand_manifest.read_text(encoding="utf-8"))
    bh_manifest = json.loads(args.backhand_manifest.read_text(encoding="utf-8"))
    selected_ids = [x.strip() for x in args.backhand_ids.split(",") if x.strip()]
    by_id = {str(e["episode_id"]): e for e in bh_manifest.get("motions", [])}
    missing = [episode_id for episode_id in selected_ids if episode_id not in by_id]
    if missing:
        raise ValueError(f"backhand IDs not found: {missing}")

    forehand_entries = fh_manifest.get("motions", [])
    if args.forehand_ids:
        forehand_ids = [x.strip() for x in args.forehand_ids.split(",") if x.strip()]
        by_forehand = {str(e["episode_id"]): e for e in forehand_entries}
        missing_forehand = [episode_id for episode_id in forehand_ids if episode_id not in by_forehand]
        if missing_forehand:
            raise ValueError(f"forehand IDs not found: {missing_forehand}")
        forehand_entries = [by_forehand[episode_id] for episode_id in forehand_ids]
    motions = [_normalise_motion_path(e, args.forehand_manifest.resolve()) for e in forehand_entries]
    for episode_id in selected_ids:
        entry = _normalise_motion_path(by_id[episode_id], args.backhand_manifest.resolve())
        entry["stance_metadata"] = _fixed_metadata(entry, args.backhand_manifest)
        entry["selection_note"] = "independent fixed-base heldout in mixed stance contract"
        motions.append(entry)

    output = {
        "manifest_name": "p2_mixed_heldout_stance_fixed_20260716",
        "status": "heldout_evidence_only",
        "training_role": "heldout_not_training",
        "source_manifests": [str(args.forehand_manifest), str(args.backhand_manifest)],
        "replay_ready_count": len(motions),
        "stroke_counts": {
            "forehand": sum(str(e.get("stroke_type", "")).lower() == "forehand" for e in motions),
            "backhand": sum(str(e.get("stroke_type", "")).lower() == "backhand" for e in motions),
        },
        "stance_contract": {
            "mode": "mixed",
            "allowed_modes": ["fixed", "prepositioned"],
            "walking_enabled": False,
        },
        "motions": motions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote mixed manifest with {len(motions)} motions: {args.output}")


if __name__ == "__main__":
    main()
