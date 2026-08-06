"""Repair and statically validate motion/strike-target pairs.

The training manifest must pair a retargeted motion with the target generated
for that exact episode and hit window.  This tool never edits the source
manifest or NPZ files.  It replaces stale target references from authoritative
retarget manifests, then validates the exported NPZ FK against the target.

Example:

    python repair_manifest_target_pairs.py \
        source/manifest.json output/manifest.json \
        --authoritative-manifest retarget_v2/tracking_motion_manifest_backhand.json \
        --authoritative-manifest retarget_v3/tracking_motion_manifest_backhand.json \
        --authoritative-manifest retarget_v4/tracking_motion_manifest_backhand.json \
        --target-override EPISODE=/path/to/target.npz=/path/to/spec.json \
        --report output/report.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MOUNT_OFFSET = np.asarray(
    (0.210211399202899, 0.0320784994676765, 0.0320358706296689), dtype=np.float64
)
WRIST_BODY_INDEX = 31
MOTION_FPS = 50.0
MAX_EXPORTED_FK_ERROR_M = 0.010
MAX_BALL_RACKET_CENTER_DISTANCE_M = 0.150


def _resolve(path_text: str, *bases: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    qv = np.asarray((x, y, z), dtype=np.float64)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_override(value: str) -> tuple[str, Path, Path | None]:
    parts = value.split("=", 2)
    if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
        raise ValueError(
            "--target-override must be EPISODE=target.npz or "
            "EPISODE=target.npz=target_spec.json"
        )
    target = Path(parts[1]).expanduser().resolve()
    spec = Path(parts[2]).expanduser().resolve() if len(parts) == 3 and parts[2] else None
    return parts[0], target, spec


def _target_position(target: dict[str, Any]) -> np.ndarray:
    value = target.get("racket_position_m")
    if value is None:
        raise ValueError("strike_target.racket_position_m is missing")
    position = np.asarray(value, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"invalid strike target position: {value!r}")
    return position


def _entry_target_from_npz(target_path: Path) -> tuple[np.ndarray, np.ndarray, float, int]:
    data = np.load(target_path, allow_pickle=False)
    required = ("racket_pose_at_hit", "hit_pos", "hit_index")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{target_path}: missing target fields {missing}")
    pose = np.asarray(data["racket_pose_at_hit"], dtype=np.float64)
    ball = np.asarray(data["hit_pos"], dtype=np.float64)
    if pose.shape != (7,) or ball.shape != (3,):
        raise ValueError(f"{target_path}: invalid target shapes pose={pose.shape}, ball={ball.shape}")
    hit_index = int(np.asarray(data["hit_index"]).reshape(-1)[0])
    distance = float(np.linalg.norm(ball - pose[:3]))
    return pose[:3], ball, distance, hit_index


def _validate_pair(entry: dict[str, Any], manifest_path: Path, old_target_path: Path | None) -> dict[str, Any]:
    motion_path = _resolve(str(entry["motion_npz"]), manifest_path.parent, Path.cwd())
    target_path = _resolve(str(entry["target_npz"]), manifest_path.parent, Path.cwd())
    if not motion_path.exists():
        raise ValueError(f"{entry['episode_id']}: motion NPZ does not exist: {motion_path}")
    if not target_path.exists():
        raise ValueError(f"{entry['episode_id']}: target NPZ does not exist: {target_path}")

    motion = np.load(motion_path, allow_pickle=False)
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if key not in motion:
            raise ValueError(f"{entry['episode_id']}: motion NPZ missing {key}")
    hit = entry.get("hit_event", {})
    frame = int(hit.get("motion_hit_frame", -1))
    if frame < 0 or frame >= motion["body_pos_w"].shape[0]:
        raise ValueError(f"{entry['episode_id']}: hit frame {frame} outside motion")
    source_hit = int(hit.get("source_hit_index", -1))
    source_fps = float(hit.get("source_fps", 0.0))
    expected_frame = int(round(source_hit / source_fps * MOTION_FPS)) if source_fps > 0 else -1
    if expected_frame != frame:
        raise ValueError(
            f"{entry['episode_id']}: frame mapping mismatch source={source_hit}/{source_fps} "
            f"expects motion={expected_frame}, manifest has {frame}"
        )

    wrist_pos = np.asarray(motion["body_pos_w"][frame, WRIST_BODY_INDEX], dtype=np.float64)
    wrist_quat = np.asarray(motion["body_quat_w"][frame, WRIST_BODY_INDEX], dtype=np.float64)
    exported_racket = wrist_pos + _quat_apply_wxyz(wrist_quat, MOUNT_OFFSET)
    manifest_target = _target_position(entry["strike_target"])
    npz_target, ball, ball_distance, target_hit_index = _entry_target_from_npz(target_path)
    fk_error = float(np.linalg.norm(exported_racket - manifest_target))
    target_copy_error = float(np.linalg.norm(manifest_target - npz_target))
    if fk_error > MAX_EXPORTED_FK_ERROR_M:
        raise ValueError(f"{entry['episode_id']}: exported FK error {fk_error:.6f} m exceeds threshold")
    if target_copy_error > 0.001:
        raise ValueError(f"{entry['episode_id']}: manifest/target NPZ mismatch {target_copy_error:.6f} m")
    if target_hit_index != source_hit:
        raise ValueError(
            f"{entry['episode_id']}: target hit index {target_hit_index} != source hit index {source_hit}"
        )
    if ball_distance > MAX_BALL_RACKET_CENTER_DISTANCE_M:
        raise ValueError(f"{entry['episode_id']}: ball/racket center distance {ball_distance:.6f} m exceeds threshold")

    return {
        "episode_id": entry["episode_id"],
        "motion_npz": str(motion_path),
        "target_npz": str(target_path),
        "hit_frame": frame,
        "source_hit_index": source_hit,
        "exported_racket_position_m": exported_racket.tolist(),
        "manifest_target_position_m": manifest_target.tolist(),
        "target_npz_position_m": npz_target.tolist(),
        "exported_fk_error_m": fk_error,
        "manifest_target_npz_error_m": target_copy_error,
        "ball_position_m": ball.tolist(),
        "ball_to_racket_center_distance_m": ball_distance,
        "old_target_npz": str(old_target_path) if old_target_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--authoritative-manifest", action="append", required=True)
    parser.add_argument("--target-override", action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    output_path = args.output_manifest.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    source = _load_json(source_path)
    source_entries = {str(item["episode_id"]): item for item in source.get("motions", [])}
    if not source_entries:
        raise ValueError(f"no motions in {source_path}")

    authoritative: dict[str, tuple[dict[str, Any], Path]] = {}
    for manifest_text in args.authoritative_manifest:
        manifest_path = Path(manifest_text).expanduser().resolve()
        data = _load_json(manifest_path)
        for item in data.get("motions", []):
            episode_id = str(item["episode_id"])
            if episode_id in authoritative:
                raise ValueError(f"duplicate authoritative episode: {episode_id}")
            authoritative[episode_id] = (item, manifest_path)

    overrides: dict[str, tuple[Path, Path | None]] = {}
    for value in args.target_override:
        episode_id, target_path, spec_path = _parse_override(value)
        overrides[episode_id] = (target_path, spec_path)

    output = copy.deepcopy(source)
    output["manifest_name"] = f"{source.get('manifest_name', output_path.stem)}_v2_repaired_target_pairs"
    # The repository training contract currently accepts only this status.
    # Detailed static-gate evidence remains in repair_contract and the report.
    output["dataset_status"] = "active_training_candidate"
    output["repair_contract"] = {
        "method": "authoritative_same_episode_target_pairing",
        "wrist_body_index": WRIST_BODY_INDEX,
        "mount_offset_m": MOUNT_OFFSET.tolist(),
        "motion_fps": MOTION_FPS,
        "max_exported_fk_error_m": MAX_EXPORTED_FK_ERROR_M,
        "max_ball_racket_center_distance_m": MAX_BALL_RACKET_CENTER_DISTANCE_M,
        "source_manifest": str(source_path),
        "authoritative_manifests": [str(Path(p).expanduser().resolve()) for p in args.authoritative_manifest],
        "target_overrides": sorted(overrides),
        "note": "Static gates passed; native zero-residual rollout is still required before PPO training.",
    }

    rows: list[dict[str, Any]] = []
    for index, old_entry in enumerate(source.get("motions", [])):
        episode_id = str(old_entry["episode_id"])
        if episode_id not in authoritative:
            raise ValueError(f"no authoritative entry for {episode_id}")
        authoritative_entry, authoritative_manifest = authoritative[episode_id]
        old_motion = _resolve(str(old_entry["motion_npz"]), source_path.parent, Path.cwd())
        new_motion = _resolve(str(authoritative_entry["motion_npz"]), authoritative_manifest.parent, Path.cwd())
        if old_motion != new_motion:
            raise ValueError(f"{episode_id}: motion pair changed unexpectedly: {old_motion} != {new_motion}")

        repaired = copy.deepcopy(old_entry)
        for key in ("motion_npz", "optimized_csv", "hit_event", "strike_target", "fps", "joint_pos_shape", "body_pos_w_shape"):
            if key in authoritative_entry:
                repaired[key] = copy.deepcopy(authoritative_entry[key])
        repaired["motion_npz"] = str(new_motion)
        repaired["optimized_csv"] = str(
            _resolve(str(authoritative_entry["optimized_csv"]), authoritative_manifest.parent, Path.cwd())
        )
        old_target = _resolve(str(old_entry["target_npz"]), source_path.parent, Path.cwd())
        if episode_id in overrides:
            target_path, spec_path = overrides[episode_id]
            target_spec = spec_path
        else:
            target_path = _resolve(str(authoritative_entry["target_npz"]), authoritative_manifest.parent, Path.cwd())
            target_spec = (
                _resolve(str(authoritative_entry["target_spec_json"]), authoritative_manifest.parent, Path.cwd())
                if authoritative_entry.get("target_spec_json")
                else None
            )
        repaired["target_npz"] = str(target_path)
        if target_spec is not None:
            if not target_spec.exists():
                raise ValueError(f"{episode_id}: target spec does not exist: {target_spec}")
            repaired["target_spec_json"] = str(target_spec)
        repaired["target_pair_repair"] = {
            "authoritative_manifest": str(authoritative_manifest),
            "old_target_npz": str(old_target),
            "new_target_npz": str(target_path),
        }
        output["motions"][index] = repaired

        row = _validate_pair(repaired, output_path, old_target)
        row["authoritative_manifest"] = str(authoritative_manifest)
        rows.append(row)

    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    output["repair_contract"]["validated_rows_sha256"] = hashlib.sha256(canonical).hexdigest()
    output["repaired_target_pair_sha256"] = output["repair_contract"]["validated_rows_sha256"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "status": "pass",
        "source_manifest": str(source_path),
        "output_manifest": str(output_path),
        "motion_count": len(rows),
        "rows": rows,
        "max_exported_fk_error_m": max(row["exported_fk_error_m"] for row in rows),
        "max_manifest_target_npz_error_m": max(row["manifest_target_npz_error_m"] for row in rows),
        "max_ball_to_racket_center_distance_m": max(row["ball_to_racket_center_distance_m"] for row in rows),
        "validated_rows_sha256": output["repair_contract"]["validated_rows_sha256"],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
