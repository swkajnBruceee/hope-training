#!/usr/bin/env python3
"""Diagnose frame/scale/root/init issues for one refinement spec."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    del _ROOT

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from analysis.mocap_cleaning.a3_refinement_solver import _compute_racket_series, load_retarget_csv
from analysis.mocap_cleaning.bvh_motion import joint_global_transform, load_bvh
from analysis.mocap_cleaning.refinement_spec import resolve_existing_path


def _metadata_for_spec(spec: dict[str, Any]) -> dict[str, Any]:
    sample_npz = resolve_existing_path(spec["inputs"]["source_sample_npz"])
    metadata_path = sample_npz.parent.parent / "metadata" / f"{spec['episode_id']}.json"
    return json.loads(resolve_existing_path(metadata_path).read_text())


def _raw_bvh_path(spec: dict[str, Any], metadata: dict[str, Any]) -> Path:
    source_csv_rel = metadata["source"]["source_csv"]
    m = re.search(r"Skeleton(\d+)$", spec["episode_id"])
    if not m:
        raise ValueError(f"cannot parse skeleton from {spec['episode_id']}")
    skeleton_num = int(m.group(1))
    return Path("DATA260703") / source_csv_rel.replace("Csv/", "Bvh/").replace(".csv", f"_Skeleton {skeleton_num:03d}.bvh")


def _vector_report(name: str, value: np.ndarray) -> str:
    return f"{name}: [{value[0]: .4f}, {value[1]: .4f}, {value[2]: .4f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--search-radius", type=int, default=5)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    metadata = _metadata_for_spec(spec)
    motion = load_bvh(_raw_bvh_path(spec, metadata))
    generic_csv = load_retarget_csv(spec["artifacts"]["generic_retarget_csv"]) if Path(spec["artifacts"]["generic_retarget_csv"]).exists() else None
    hit_index = int(spec["hit_target"]["hit_index"])
    hit_time = float(metadata["source"]["hit_metadata"]["hit_time"])
    target_times = hit_time + np.load(resolve_existing_path(spec["inputs"]["source_sample_npz"]), allow_pickle=False)["time_rel"]
    bvh_frame_hit = int(np.argmin(np.abs(motion.time - hit_time)))

    hips_pos, _ = joint_global_transform(motion, "Hips", bvh_frame_hit)
    rhand_pos, _ = joint_global_transform(motion, "RightHand", bvh_frame_hit)
    rshoulder_pos, _ = joint_global_transform(motion, "RightShoulder", bvh_frame_hit)
    rarm_pos, _ = joint_global_transform(motion, "RightArm", bvh_frame_hit)
    rforearm_pos, _ = joint_global_transform(motion, "RightForeArm", bvh_frame_hit)

    print(f"spec_version: {spec['spec_version']}")
    print(f"contract_version: {spec['contract_version']}")
    print(f"episode_id: {spec['episode_id']}")
    print(f"quat_order: {spec['coordinate_contract']['quat_order']}")
    print(f"position_frame: {spec['coordinate_contract']['position_frame']}")
    print(f"fps/dt: {spec['coordinate_contract']['fps']} / {spec['coordinate_contract']['dt']}")
    print(f"hit_index: {hit_index}")
    print(f"hit_time_raw_s: {hit_time:.6f}")
    print(f"bvh_frame_hit: {bvh_frame_hit}")
    print(_vector_report("target.racket_position_m", np.asarray(spec["hit_target"]["racket_position_m"], dtype=np.float64)))
    print(_vector_report("bvh.hips_position_raw_units", hips_pos))
    print(_vector_report("bvh.right_hand_position_raw_units", rhand_pos))
    print(_vector_report("bvh.right_shoulder_position_raw_units", rshoulder_pos))
    print(f"bvh.upper_arm_length_raw: {np.linalg.norm(rarm_pos - rshoulder_pos):.4f}")
    print(f"bvh.forearm_length_raw: {np.linalg.norm(rhand_pos - rforearm_pos):.4f}")
    print(f"bvh.shoulder_to_hand_raw: {np.linalg.norm(rhand_pos - rshoulder_pos):.4f}")

    if generic_csv is not None:
        base_pos = generic_csv[hit_index, :3]
        print(_vector_report("generic_csv.base_position_at_hit", base_pos))
        racket_pos, racket_normal, _ = _compute_racket_series(generic_csv, spec)
        err = racket_pos[hit_index] - np.asarray(spec["hit_target"]["racket_position_m"], dtype=np.float64)
        print(_vector_report("a3_fk.racket_position_at_hit", racket_pos[hit_index]))
        print(_vector_report("a3_fk.minus_target", err))
        best = None
        radius = int(args.search_radius)
        for offset in range(-radius, radius + 1):
            idx = int(np.clip(hit_index + offset, 0, generic_csv.shape[0] - 1))
            pos_err = float(np.linalg.norm(racket_pos[idx] - np.asarray(spec["hit_target"]["racket_position_m"], dtype=np.float64)))
            if best is None or pos_err < best[1]:
                best = (offset, pos_err, idx)
        print(f"best_pos_err_within_hit±{radius}: offset={best[0]}, frame={best[2]}, err={best[1]:.4f}")


if __name__ == "__main__":
    main()
