#!/usr/bin/env python3
"""Build a large, smooth upright backhand reference bank.

The source bank contains several retargeting seeds for the same backhand
workspace.  Some seeds have an unnecessarily aggressive arm trajectory.  This
builder keeps only the low-speed/low-acceleration seeds, interpolates their
phase-aligned joint trajectories, and removes waist roll/pitch while retaining
waist yaw.  It is a reference-candidate generator, not a PhysX qualification
tool: every output is explicitly marked ``physics_qualified=false``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_upper_momentum_library import UrdfModel  # noqa: E402
from materialize_p4b_repaired_canonical_prior import (  # noqa: E402
    DEFAULT_URDF,
    _regenerate_body_arrays,
    _relative_body_velocity_from_joint_state,
)


def _normalise_quaternion(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-12)


def _arm_indices(names: list[str]) -> list[int]:
    return [i for i, name in enumerate(names) if any(k in name for k in ("shoulder", "elbow", "wrist"))]


def _smoothness(q: np.ndarray, fps: float, arm: list[int]) -> tuple[float, float, float]:
    qd = np.gradient(q, 1.0 / fps, axis=0)
    qdd = np.gradient(qd, 1.0 / fps, axis=0)
    speed = np.linalg.norm(qd[:, arm], axis=1)
    # A negative difference after the hit is the recovery deceleration.  The
    # limit catches the previous "急停" trajectories without rejecting a
    # gradual, natural recovery.
    post_hit_drop = float(max(0.0, -np.min(np.diff(speed[30:46]))))
    return float(np.max(speed)), float(np.max(np.linalg.norm(qdd[:, arm], axis=1))), post_hit_drop


def _load_motion(entry: dict) -> dict[str, np.ndarray]:
    path = Path(str(entry["motion_npz"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def _blend(a: dict[str, np.ndarray], b: dict[str, np.ndarray], alpha: float, names: list[str], model: UrdfModel) -> dict[str, np.ndarray]:
    q = (1.0 - alpha) * np.asarray(a["joint_pos"], dtype=np.float64) + alpha * np.asarray(b["joint_pos"], dtype=np.float64)
    qd = np.gradient(q, 1.0 / float(np.asarray(a["fps"]).reshape(-1)[0]), axis=0)
    root_pos_a = np.asarray(a["body_pos_w"], dtype=np.float64)[:, 0]
    root_pos_b = np.asarray(b["body_pos_w"], dtype=np.float64)[:, 0]
    root_pos = (1.0 - alpha) * root_pos_a + alpha * root_pos_b
    root_quat_a = np.asarray(a["body_quat_w"], dtype=np.float64)[:, 0]
    root_quat_b = np.asarray(b["body_quat_w"], dtype=np.float64)[:, 0]
    root_quat = _normalise_quaternion((1.0 - alpha) * root_quat_a + alpha * root_quat_b)
    body_names = list(json.loads((ROOT / "docs/a3_articulation_metadata.json").read_text())["body_names"])
    body_pos_w, body_quat_w = _regenerate_body_arrays(model, names, body_names, q, root_pos, root_quat, float(np.asarray(a["fps"]).reshape(-1)[0]))
    body_lin, body_ang = _relative_body_velocity_from_joint_state(model, names, body_names, q, qd, root_quat)
    # The source bank is a fixed-base reference (world root is constant), so
    # relative FK velocities are also the world link velocities.
    return {
        "fps": np.asarray(a["fps"]).copy(),
        "joint_pos": q.astype(np.float32),
        "joint_vel": qd.astype(np.float32),
        "body_pos_w": body_pos_w.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_lin_vel_w": body_lin.astype(np.float32),
        "body_ang_vel_w": body_ang.astype(np.float32),
        "body_pos_b0": (body_pos_w - root_pos[:, None, :]).astype(np.float32),
        "body_quat_b0_wxyz": body_quat_w.astype(np.float32),
        "body_lin_vel_b0": body_lin.astype(np.float32),
        "body_ang_vel_b0": body_ang.astype(np.float32),
        "hit_frame": np.asarray(a.get("hit_frame", [30])).copy(),
        "physics_qualified": np.asarray([False]),
        "scene_root_anchor_w_m": np.asarray(a.get("scene_root_anchor_w_m", [-0.5, -0.7625, 1.04])).copy(),
        "scene_root_heading_w_rad": np.asarray(a.get("scene_root_heading_w_rad", [0.0])).copy(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--count", type=int, default=96)
    ap.add_argument("--max-arm-speed", type=float, default=3.25)
    ap.add_argument("--max-arm-acceleration", type=float, default=40.0)
    ap.add_argument("--max-post-hit-drop", type=float, default=0.30)
    ap.add_argument("--metadata", type=Path, default=ROOT / "docs/a3_articulation_metadata.json")
    args = ap.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    names = list(metadata["joint_names"])
    arm = _arm_indices(names)
    waist_roll = names.index("waist_roll_joint")
    waist_pitch = names.index("waist_pitch_joint")
    model = UrdfModel(DEFAULT_URDF)

    entries = source["motions"]
    loaded: list[tuple[int, dict, dict[str, np.ndarray], tuple[float, float, float]]] = []
    for i, entry in enumerate(entries):
        arrays = _load_motion(entry)
        q = np.asarray(arrays["joint_pos"], dtype=np.float64).copy()
        q[:, [waist_roll, waist_pitch]] = 0.0
        fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
        stats = _smoothness(q, fps, arm)
        if stats[0] <= args.max_arm_speed and stats[1] <= args.max_arm_acceleration and stats[2] <= args.max_post_hit_drop:
            arrays["joint_pos"] = q
            loaded.append((i, entry, arrays, stats))
    if len(loaded) < 4:
        raise RuntimeError(f"only {len(loaded)} safe seeds survived the smoothness gates")

    # Prefer spatially distinct seeds, then retain the remaining safe seeds as
    # phase/style variants.  The first pass is deterministic and reproducible.
    selected: list[tuple[int, dict, dict[str, np.ndarray], tuple[float, float, float]]] = []
    seen_goal: set[tuple[float, ...]] = set()
    for item in loaded:
        goal = item[1].get("canonical_goal_10d", {})
        key = tuple(round(float(x), 5) for x in goal.get("position_b0_m", []))
        if key not in seen_goal:
            selected.append(item)
            seen_goal.add(key)
    for item in loaded:
        if item not in selected:
            selected.append(item)

    # Build endpoints plus pairwise midpoints and quarter points.  Pairwise
    # blends cover the convex hull of the accepted backhand workspace without
    # inventing a high-speed new stroke.
    recipes: list[tuple[int, int, float]] = [(i, i, 0.0) for i in range(len(selected))]
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            recipes.append((i, j, 0.5))
    for i, j in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (0, min(4, len(selected) - 1))]:
        if i < len(selected) and j < len(selected) and i != j:
            recipes.extend([(i, j, 0.25), (i, j, 0.75)])
    recipes = recipes[: max(args.count, len(selected))]

    out_dir = args.output_dir.expanduser().resolve()
    motion_dir = out_dir / "motion_npz"
    motion_dir.mkdir(parents=True, exist_ok=True)
    out_entries: list[dict] = []
    audit: list[dict] = []
    for out_i, (ia, ib, alpha) in enumerate(recipes):
        source_a = selected[ia]; source_b = selected[ib]
        a_entry, a_arrays = source_a[1], source_a[2]
        b_entry, b_arrays = source_b[1], source_b[2]
        arrays = _blend(a_arrays, b_arrays, alpha, names, model)
        q = np.asarray(arrays["joint_pos"], dtype=np.float64)
        fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
        stats = _smoothness(q, fps, arm)
        # Interpolate the canonical 10-D reference contract.
        ga = a_entry["canonical_goal_10d"]; gb = b_entry["canonical_goal_10d"]
        normal = _normalise_quaternion(np.asarray((1 - alpha) * np.asarray(ga["normal_b0"]) + alpha * np.asarray(gb["normal_b0"]), dtype=np.float64))
        goal = {
            "position_b0_m": ((1 - alpha) * np.asarray(ga["position_b0_m"]) + alpha * np.asarray(gb["position_b0_m"])).tolist(),
            "normal_b0": normal.tolist(),
            "linear_velocity_b0_mps": ((1 - alpha) * np.asarray(ga["linear_velocity_b0_mps"]) + alpha * np.asarray(gb["linear_velocity_b0_mps"])).tolist(),
            "time_to_hit_s": float((1 - alpha) * float(ga["time_to_hit_s"]) + alpha * float(gb["time_to_hit_s"])),
        }
        # Keep the canonical target in the NPZ as well as in the manifest;
        # different replay/training entry points consume different sides of
        # this reference contract.
        arrays["canonical_goal_position_b0_m"] = np.asarray(goal["position_b0_m"], dtype=np.float64)
        arrays["canonical_goal_normal_b0"] = np.asarray(goal["normal_b0"], dtype=np.float64)
        arrays["canonical_goal_linear_velocity_b0_mps"] = np.asarray(goal["linear_velocity_b0_mps"], dtype=np.float64)
        arrays["canonical_goal_time_to_hit_s"] = np.asarray([goal["time_to_hit_s"]], dtype=np.float64)
        stem = f"upright_backhand_ref_{out_i:03d}_a{alpha:.3f}"
        motion_path = motion_dir / f"{stem}.npz"
        np.savez_compressed(motion_path, **arrays)
        entry = copy.deepcopy(a_entry)
        entry["episode_id"] = stem
        entry["motion_id"] = out_i
        entry["motion_npz"] = str(motion_path)
        # Do not leave historical source aliases in a generated candidate:
        # Isaac's loader gives those aliases precedence and would silently
        # replay the old aggressive clip instead of this new NPZ.
        entry.pop("library_motion_npz", None)
        entry.pop("canonical_motion_npz", None)
        entry["canonical_goal_10d"] = goal
        entry["reference_contract"] = {"schema": "p5_complete_reference/v1", "hit_frame": 30, "frames": int(q.shape[0]), "upright_torso": True}
        entry["upright_torso_reference_bank"] = True
        entry["augmentation"] = {"method": "safe_seed_convex_phase_aligned", "source_a": source_a[1]["episode_id"], "source_b": source_b[1]["episode_id"], "alpha": float(alpha)}
        entry["teacher_approved"] = False
        entry["physics_qualified"] = False
        out_entries.append(entry)
        audit.append({"motion_id": out_i, "source_a": source_a[1]["episode_id"], "source_b": source_b[1]["episode_id"], "alpha": float(alpha), "arm_speed_max_radps": stats[0], "arm_acceleration_max_radps2": stats[1], "post_hit_speed_drop_radps_per_frame": stats[2], "waist_roll_abs_max_rad": float(np.max(np.abs(q[:, waist_roll]))), "waist_pitch_abs_max_rad": float(np.max(np.abs(q[:, waist_pitch]))), "passed_offline_gates": bool(stats[0] <= args.max_arm_speed and stats[1] <= args.max_arm_acceleration and stats[2] <= args.max_post_hit_drop)})

    payload = {
        "schema_version": "upright_backhand_reference_bank/v1",
        "status": "candidate_only_physx_replay_required",
        "training_role": "reference_candidates_not_training_approved",
        "teacher_approved": False,
        "physics_qualified": False,
        "source_manifest": str(source_path),
        "motion_count": len(out_entries),
        "upright_torso_contract": {"waist_yaw": "preserved", "waist_roll": 0.0, "waist_pitch": 0.0, "allow_forward_tilt": False},
        "smoothness_gates": {"max_arm_speed_radps": args.max_arm_speed, "max_arm_acceleration_radps2": args.max_arm_acceleration, "max_post_hit_drop_radps_per_frame": args.max_post_hit_drop, "hit_frame": 30, "fps": 50},
        "coverage_note": "Convex hull of low-speed accepted seeds spanning the available backhand workspace; PhysX replay remains mandatory.",
        "motions": out_entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "offline_audit.json").write_text(json.dumps({"safe_seed_count": len(selected), "safe_seed_ids": [x[1]["episode_id"] for x in selected], "generated_count": len(out_entries), "rows": audit}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"safe_seeds": len(selected), "generated": len(out_entries), "manifest": str(out_dir / "manifest.json"), "audit": str(out_dir / "offline_audit.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
