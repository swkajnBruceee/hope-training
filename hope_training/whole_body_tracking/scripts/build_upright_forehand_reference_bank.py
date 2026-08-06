#!/usr/bin/env python3
"""Build a smooth, broad upright forehand reference bank.

This materializer is deliberately offline and candidate-only.  It takes the
existing forehand retarget seeds, smooths the arm trajectory with a gentle
time-domain Gaussian filter, keeps waist yaw but removes waist roll/pitch,
then creates exactly 99 phase-aligned convex combinations.  The output is not
PhysX-qualified and must still pass the P0/P1 replay gates before training.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from build_upper_momentum_library import UrdfModel  # noqa: E402
from materialize_p4b_repaired_canonical_prior import (  # noqa: E402
    DEFAULT_URDF,
    MOUNT_OFFSET,
    _quat_matrix,
    _regenerate_body_arrays,
    _relative_body_velocity_from_joint_state,
)
from build_upright_backhand_reference_bank import _arm_indices, _smoothness  # noqa: E402


def _resolve_motion_path(raw: str, manifest_path: Path) -> Path:
    """Resolve both current relative paths and old /home/bruce absolute paths."""
    value = Path(str(raw)).expanduser()
    candidates = []
    if value.is_absolute():
        candidates.append(value)
        text = str(value)
        marker = "/data/analysis/"
        if marker in text:
            candidates.append(DATASET_ROOT / "data" / "analysis" / text.split(marker, 1)[1])
    else:
        candidates.extend((DATASET_ROOT / value, manifest_path.parent / value, ROOT / value))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve motion_npz={raw!r}; checked {[str(x) for x in candidates]}")


def _load_motion(entry: dict, manifest_path: Path) -> dict[str, np.ndarray]:
    path = _resolve_motion_path(entry["motion_npz"], manifest_path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def _smooth_arm(q: np.ndarray, arm: list[int], sigma: float) -> np.ndarray:
    result = np.asarray(q, dtype=np.float64).copy()
    if sigma > 0.0:
        result[:, arm] = gaussian_filter1d(result[:, arm], sigma=float(sigma), axis=0, mode="nearest")
    return result


def _blend(a: dict[str, np.ndarray], b: dict[str, np.ndarray], alpha: float,
           names: list[str], body_names: list[str], model: UrdfModel,
           arm: list[int], waist_roll: int, waist_pitch: int,
           smoothing_sigma: float) -> dict[str, np.ndarray]:
    q_a = np.asarray(a["joint_pos"], dtype=np.float64)
    q_b = np.asarray(b["joint_pos"], dtype=np.float64)
    q = (1.0 - alpha) * q_a + alpha * q_b
    q = _smooth_arm(q, arm, smoothing_sigma)
    q[:, [waist_roll, waist_pitch]] = 0.0
    fps = float(np.asarray(a["fps"]).reshape(-1)[0])
    qd = np.gradient(q, 1.0 / fps, axis=0)
    root_pos_a = np.asarray(a["body_pos_w"], dtype=np.float64)[:, 0]
    root_pos_b = np.asarray(b["body_pos_w"], dtype=np.float64)[:, 0]
    root_pos = (1.0 - alpha) * root_pos_a + alpha * root_pos_b
    root_quat_a = np.asarray(a["body_quat_w"], dtype=np.float64)[:, 0]
    root_quat_b = np.asarray(b["body_quat_w"], dtype=np.float64)[:, 0]
    root_quat = (1.0 - alpha) * root_quat_a + alpha * root_quat_b
    root_quat /= np.maximum(np.linalg.norm(root_quat, axis=-1, keepdims=True), 1.0e-12)
    body_pos_w, body_quat_w = _regenerate_body_arrays(model, names, body_names, q, root_pos, root_quat, fps)
    body_lin, body_ang = _relative_body_velocity_from_joint_state(model, names, body_names, q, qd, root_quat)
    body_pos_b0 = body_pos_w - root_pos[:, None, :]
    return {
        "fps": np.asarray(a["fps"]).copy(),
        "joint_pos": q.astype(np.float32),
        "joint_vel": qd.astype(np.float32),
        "body_pos_w": body_pos_w.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_lin_vel_w": body_lin.astype(np.float32),
        "body_ang_vel_w": body_ang.astype(np.float32),
        "body_pos_b0": body_pos_b0.astype(np.float32),
        "body_quat_b0_wxyz": body_quat_w.astype(np.float32),
        "body_lin_vel_b0": body_lin.astype(np.float32),
        "body_ang_vel_b0": body_ang.astype(np.float32),
        "hit_frame": np.asarray(a.get("hit_frame", [30])).copy(),
        "physics_qualified": np.asarray([False]),
        "scene_root_anchor_w_m": np.asarray(a.get("scene_root_anchor_w_m", [-0.5, -0.7625, 1.04])).copy(),
        "scene_root_heading_w_rad": np.asarray(a.get("scene_root_heading_w_rad", [0.0])).copy(),
    }


def _goal_from_fk(arrays: dict[str, np.ndarray], hit: int, body_names: list[str]) -> dict:
    wrist = body_names.index("right_wrist_yaw_Link")
    root_pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)
    root_quat = np.asarray(arrays["body_quat_w"], dtype=np.float64)
    wrist_pos = root_pos[hit, wrist]
    wrist_rot = _quat_matrix(np.asarray(arrays["body_quat_w"])[hit, wrist], "wxyz")
    offset_w = wrist_rot @ MOUNT_OFFSET
    position_b0 = (wrist_pos + offset_w) - root_pos[hit, 0]
    normal_b0 = wrist_rot[:, 1]
    normal_b0 /= max(float(np.linalg.norm(normal_b0)), 1.0e-12)
    lin = np.asarray(arrays["body_lin_vel_w"], dtype=np.float64)[hit, wrist]
    ang = np.asarray(arrays["body_ang_vel_w"], dtype=np.float64)[hit, wrist]
    tcp_velocity = lin + np.cross(ang, offset_w)
    root_linear = np.asarray(arrays["body_lin_vel_w"], dtype=np.float64)[hit, 0]
    return {
        "position_b0_m": position_b0.tolist(),
        "normal_b0": normal_b0.tolist(),
        "linear_velocity_b0_mps": (tcp_velocity - root_linear).tolist(),
        "time_to_hit_s": float(hit / float(np.asarray(arrays["fps"]).reshape(-1)[0])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=99)
    parser.add_argument("--smoothing-sigma", type=float, default=3.5)
    parser.add_argument("--max-arm-speed", type=float, default=3.25)
    parser.add_argument("--max-arm-acceleration", type=float, default=40.0)
    parser.add_argument("--max-post-hit-drop", type=float, default=0.30)
    parser.add_argument("--metadata", type=Path, default=ROOT / "docs/a3_articulation_metadata.json")
    args = parser.parse_args()
    if args.count != 99:
        raise ValueError("the requested same-size bank is fixed at exactly 99 motions")

    source_path = args.source_manifest.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    names = list(metadata["joint_names"]); body_names = list(metadata["body_names"])
    arm = _arm_indices(names); waist_roll = names.index("waist_roll_joint"); waist_pitch = names.index("waist_pitch_joint")
    model = UrdfModel(DEFAULT_URDF)

    safe = []
    for entry in source["motions"]:
        arrays = _load_motion(entry, source_path)
        q = _smooth_arm(np.asarray(arrays["joint_pos"], dtype=np.float64), arm, args.smoothing_sigma)
        q[:, [waist_roll, waist_pitch]] = 0.0
        arrays["joint_pos"] = q
        fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
        stats = _smoothness(q, fps, arm)
        if stats[0] <= args.max_arm_speed and stats[1] <= args.max_arm_acceleration and stats[2] <= args.max_post_hit_drop:
            safe.append((entry, arrays, stats))
    if len(safe) < 4:
        raise RuntimeError(f"only {len(safe)} smooth forehand seeds passed the offline gates")

    # The eight available forehand seeds cover two target families and four
    # lateral offsets.  Endpoints + 3-point pair blends + seven quarter-edge
    # variants give exactly 99 deterministic references.
    recipes = [(i, i, 0.0) for i in range(len(safe))]
    for i in range(len(safe)):
        for j in range(i + 1, len(safe)):
            for alpha in (0.25, 0.50, 0.75):
                recipes.append((i, j, alpha))
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (4, 5)]:
        if i < len(safe) and j < len(safe):
            recipes.append((i, j, 0.125))
    recipes = recipes[:args.count]
    if len(recipes) != args.count:
        raise RuntimeError(f"source has {len(safe)} seeds, generated {len(recipes)} instead of {args.count}")

    out_dir = args.output_dir.expanduser().resolve(); motion_dir = out_dir / "motion_npz"; motion_dir.mkdir(parents=True, exist_ok=True)
    rows = []; audit = []
    for motion_id, (ia, ib, alpha) in enumerate(recipes):
        entry_a, arrays_a, _ = safe[ia]; entry_b, arrays_b, _ = safe[ib]
        arrays = _blend(arrays_a, arrays_b, alpha, names, body_names, model, arm, waist_roll, waist_pitch, args.smoothing_sigma)
        q = np.asarray(arrays["joint_pos"], dtype=np.float64); fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
        hit = int(entry_a.get("hit_event", {}).get("motion_hit_frame", np.asarray(arrays.get("hit_frame", [30])).reshape(-1)[0]))
        hit = max(0, min(hit, q.shape[0] - 1)); arrays["hit_frame"] = np.asarray([hit])
        goal = _goal_from_fk(arrays, hit, body_names)
        arrays["canonical_goal_position_b0_m"] = np.asarray(goal["position_b0_m"], dtype=np.float64)
        arrays["canonical_goal_normal_b0"] = np.asarray(goal["normal_b0"], dtype=np.float64)
        arrays["canonical_goal_linear_velocity_b0_mps"] = np.asarray(goal["linear_velocity_b0_mps"], dtype=np.float64)
        arrays["canonical_goal_time_to_hit_s"] = np.asarray([goal["time_to_hit_s"]], dtype=np.float64)
        stats = _smoothness(q, fps, arm)
        passed = bool(stats[0] <= args.max_arm_speed and stats[1] <= args.max_arm_acceleration and stats[2] <= args.max_post_hit_drop)
        if not passed:
            raise RuntimeError(f"generated motion {motion_id} failed smoothness gates: {stats}")
        stem = f"upright_forehand_ref_{motion_id:03d}_a{alpha:.3f}"
        path = motion_dir / f"{stem}.npz"; np.savez_compressed(path, **arrays)
        out = copy.deepcopy(entry_a); out.update({"episode_id": stem, "motion_id": motion_id, "motion_npz": str(path), "stroke_type": "forehand", "canonical_goal_10d": goal})
        out.pop("library_motion_npz", None); out.pop("canonical_motion_npz", None)
        out["reference_contract"] = {"schema": "p5_complete_reference/v1", "hit_frame": hit, "frames": int(q.shape[0]), "upright_torso": True, "allow_forward_tilt": False}
        out["upright_torso_reference_bank"] = True; out["augmentation"] = {"method": "smooth_forehand_convex_phase_aligned", "source_a": entry_a["episode_id"], "source_b": entry_b["episode_id"], "alpha": float(alpha), "smoothing_sigma_frames": float(args.smoothing_sigma)}
        out["teacher_approved"] = False; out["physics_qualified"] = False; rows.append(out)
        audit.append({"motion_id": motion_id, "source_a": entry_a["episode_id"], "source_b": entry_b["episode_id"], "alpha": float(alpha), "arm_speed_max_radps": stats[0], "arm_acceleration_max_radps2": stats[1], "post_hit_speed_drop_radps_per_frame": stats[2], "waist_roll_abs_max_rad": float(np.max(np.abs(q[:, waist_roll]))), "waist_pitch_abs_max_rad": float(np.max(np.abs(q[:, waist_pitch]))), "passed_offline_gates": passed})

    payload = {"schema_version": "upright_forehand_reference_bank/v1", "status": "candidate_only_physx_replay_required", "training_role": "reference_candidates_not_training_approved", "teacher_approved": False, "physics_qualified": False, "source_manifest": str(source_path), "motion_count": len(rows), "upright_torso_contract": {"waist_yaw": "preserved", "waist_roll": 0.0, "waist_pitch": 0.0, "allow_forward_tilt": False}, "smoothness_gates": {"smoothing_sigma_frames": args.smoothing_sigma, "max_arm_speed_radps": args.max_arm_speed, "max_arm_acceleration_radps2": args.max_arm_acceleration, "max_post_hit_drop_radps_per_frame": args.max_post_hit_drop, "hit_frame": 30, "fps": 50}, "coverage_note": "99 smooth convex references spanning the available forehand target families and four lateral offsets; PhysX replay remains mandatory.", "motions": rows}
    (out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "offline_audit.json").write_text(json.dumps({"safe_seed_count": len(safe), "safe_seed_ids": [x[0]["episode_id"] for x in safe], "generated_count": len(rows), "rows": audit}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"safe_seeds": len(safe), "generated": len(rows), "manifest": str(out_dir / "manifest.json"), "audit": str(out_dir / "offline_audit.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
