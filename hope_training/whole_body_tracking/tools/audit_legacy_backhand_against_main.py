#!/usr/bin/env python3
"""Audit the 96-motion legacy backhand supplement against the 14,830 main bank."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _bounds(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "span": (values.max(axis=0) - values.min(axis=0)).tolist(),
    }


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-csv", type=Path, required=True)
    parser.add_argument("--supplement-manifest", type=Path, required=True)
    parser.add_argument("--tcp-audit", type=Path, required=True)
    parser.add_argument("--physx-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    main_rows = [row for row in csv.DictReader(args.main_csv.expanduser().resolve().open()) if row["stroke"] == "backhand"]
    main_goals = []
    main_missing = []
    main_csv_goal_errors = []
    main_frames = set()
    for row in main_rows:
        goal_path = Path(row["normalized_goal_json"]).expanduser().resolve()
        if not goal_path.exists():
            main_missing.append(str(goal_path))
            continue
        goal = json.loads(goal_path.read_text(encoding="utf-8"))
        main_frames.add(goal.get("frame"))
        position = np.asarray(goal["position_m"], dtype=np.float64)
        velocity = np.asarray(goal["linear_velocity_mps"], dtype=np.float64)
        normal = np.asarray(goal["racket_normal"], dtype=np.float64)
        if not np.allclose(position, [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], atol=1e-7):
            main_csv_goal_errors.append(row["goal_id"])
        main_goals.append({"position": position, "velocity": velocity, "normal": normal, "time": float(goal["time_to_strike_s"])})

    supplement_path = args.supplement_manifest.expanduser().resolve()
    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    motions = supplement["motions"]
    payload_errors: list[dict] = []
    positions = []
    velocities = []
    normals = []
    times = []
    waist = []
    for entry in motions:
        path = Path(entry["motion_npz"]).expanduser().resolve()
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {"joint_names", "joint_pos", "joint_vel", "hit_frame", "canonical_position", "canonical_velocity", "canonical_normal", "canonical_strike_time_s"}
                missing = sorted(required - set(data.files))
                if missing:
                    payload_errors.append({"episode_id": entry["episode_id"], "error": "missing_fields", "fields": missing})
                    continue
                q = np.asarray(data["joint_pos"], dtype=np.float64)
                names = np.asarray(data["joint_names"]).reshape(-1)
                pos = np.asarray(data["canonical_position"], dtype=np.float64).reshape(3)
                vel = np.asarray(data["canonical_velocity"], dtype=np.float64).reshape(3)
                normal = np.asarray(data["canonical_normal"], dtype=np.float64).reshape(3)
                t = float(np.asarray(data["canonical_strike_time_s"]).reshape(-1)[0])
                hit = int(np.asarray(data["hit_frame"]).reshape(-1)[0])
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                checks = {
                    "joint_name_count": len(names) == 31,
                    "joint_pos_shape": q.shape == (81, 31),
                    "finite": bool(np.isfinite(q).all()),
                    "position_manifest_error_m": _max_abs(pos, np.asarray(entry["canonical_goal_10d"]["position_m"], dtype=np.float64)),
                    "velocity_manifest_error_mps": _max_abs(vel, np.asarray(entry["canonical_goal_10d"]["linear_velocity_mps"], dtype=np.float64)),
                    "normal_manifest_error": _max_abs(normal, np.asarray(entry["canonical_goal_10d"]["normal_w"], dtype=np.float64)),
                    "time_manifest_error_s": abs(t - float(entry["canonical_goal_10d"]["time_to_hit_s"])),
                    "time_hit_frame_error_s": abs(t - hit / fps),
                    "normal_unit_error": abs(float(np.linalg.norm(normal)) - 1.0),
                }
                if not all(checks.values() if isinstance(v, bool) else True for v in checks.values()):
                    payload_errors.append({"episode_id": entry["episode_id"], "error": "payload_contract", "checks": checks})
                positions.append(pos)
                velocities.append(vel)
                normals.append(normal)
                times.append(t)
                waist.append(entry.get("waist_contract", {}))
        except Exception as exc:
            payload_errors.append({"episode_id": entry["episode_id"], "error": repr(exc)})

    main_position = np.asarray([x["position"] for x in main_goals], dtype=np.float64)
    main_velocity = np.asarray([x["velocity"] for x in main_goals], dtype=np.float64)
    main_normal = np.asarray([x["normal"] for x in main_goals], dtype=np.float64)
    main_time = np.asarray([x["time"] for x in main_goals], dtype=np.float64)
    supp_position = np.asarray(positions, dtype=np.float64)
    supp_velocity = np.asarray(velocities, dtype=np.float64)
    supp_normal = np.asarray(normals, dtype=np.float64)
    supp_time = np.asarray(times, dtype=np.float64)
    distances = np.linalg.norm(supp_position[:, None, :] - main_position[None, :, :], axis=2).min(axis=1)
    outside_position = ((supp_position < main_position.min(axis=0)) | (supp_position > main_position.max(axis=0))).sum(axis=0).tolist()

    tcp = json.loads(args.tcp_audit.expanduser().resolve().read_text(encoding="utf-8"))
    physx = json.loads(args.physx_audit.expanduser().resolve().read_text(encoding="utf-8"))
    physx_statuses = {}
    for row in physx.get("rows", []):
        physx_statuses[row["motion_file"]] = row.get("status")

    report = {
        "schema_version": "a3_legacy_backhand_against_main_audit/v1",
        "main_bank": {
            "path": str(args.main_csv.expanduser().resolve()),
            "count": len(main_goals),
            "frame_values": sorted(main_frames),
            "normalized_goal_missing_count": len(main_missing),
            "csv_goal_mismatch_count": len(main_csv_goal_errors),
            "position": _bounds(main_position),
            "velocity": _bounds(main_velocity),
            "normal": _bounds(main_normal),
            "strike_time_s": {"min": float(main_time.min()), "max": float(main_time.max()), "unique": sorted(set(main_time.tolist()))},
        },
        "supplement_bank": {
            "path": str(supplement_path),
            "count": len(motions),
            "payload_count_checked": len(positions),
            "payload_error_count": len(payload_errors),
            "position": _bounds(supp_position),
            "velocity": _bounds(supp_velocity),
            "normal": _bounds(supp_normal),
            "strike_time_s": {"min": float(supp_time.min()), "max": float(supp_time.max()), "unique": sorted(set(supp_time.tolist()))},
            "waist_roll_abs_max_deg": max(float(x.get("waist_roll_abs_max_deg", 0.0)) for x in waist),
            "waist_pitch_abs_max_deg": max(float(x.get("waist_pitch_abs_max_deg", 0.0)) for x in waist),
        },
        "alignment": {
            "coordinate_contract": supplement.get("coordinate_contract"),
            "root_pose_contract": supplement.get("root_pose_contract"),
            "tcp_contract": supplement.get("tcp_contract"),
            "tcp_audit_status": tcp.get("status"),
            "tcp_evaluated_count": tcp.get("evaluated_count"),
            "tcp_source_target_mismatch_count": tcp.get("source_target_mismatch_count"),
            "tcp_relative_root_error_m": tcp.get("relative_root_error_m"),
            "physx_status_counts": {status: list(physx_statuses.values()).count(status) for status in sorted(set(physx_statuses.values()))},
            "supplement_position_outside_main_bounds_per_axis": {"x": outside_position[0], "y": outside_position[1], "z": outside_position[2]},
            "supplement_position_nearest_main_m": {"min": float(distances.min()), "mean": float(distances.mean()), "p50": float(np.percentile(distances, 50)), "max": float(distances.max()), "le_0p01_count": int((distances <= 0.01).sum()), "le_0p02_count": int((distances <= 0.02).sum()), "le_0p05_count": int((distances <= 0.05).sum())},
        },
        "payload_errors": payload_errors,
        "status": "pass" if len(motions) == 96 and len(main_goals) == 14830 and not payload_errors and tcp.get("error_count", 1) == 0 and tcp.get("source_target_mismatch_count", 1) == 0 and all(v == "FIXED_BASE_PHYSX_REPLAY_PASS" for v in physx_statuses.values()) else "review_required",
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "main_count": len(main_goals), "supplement_count": len(motions), "payload_errors": len(payload_errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
