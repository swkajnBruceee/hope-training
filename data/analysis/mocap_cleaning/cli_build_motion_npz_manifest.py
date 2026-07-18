#!/usr/bin/env python3
"""Build a manifest/summary for motion NPZ outputs generated from csv_to_npz jobs."""

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
from pathlib import Path

import numpy as np


def _hit_metadata(job: dict, fps: int) -> dict:
    spec_path = Path(str(job.get("target_spec_json", "")))
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    contract = spec.get("coordinate_contract", {})
    hit_target = spec.get("hit_target", {})
    source_fps = float(contract.get("fps", job.get("input_fps", 0)) or 0.0)
    hit_index = int(
        contract.get(
            "hit_index",
            hit_target.get("hit_index", spec.get("candidate_hit_index", spec.get("hit_index", 0))),
        )
        or 0
    )
    target_npz = Path(str(job.get("target_npz", "")))
    if target_npz.exists():
        with np.load(target_npz, allow_pickle=False) as target:
            hit_index = int(target["hit_index"]) if "hit_index" in target else hit_index
            source_fps = float(target["source_fps"]) if "source_fps" in target else source_fps
            if not hit_target and 0 <= hit_index < len(target["racket_pos"]):
                hit_target = {
                    "racket_position_m": target["racket_pos"][hit_index].tolist(),
                    "racket_velocity_mps": target["racket_vel"][hit_index].tolist(),
                    "racket_normal_w": (
                        target["racket_normal_w"] if "racket_normal_w" in target else target["racket_normal_base"]
                    )[hit_index].tolist(),
                    "racket_tangent_w": (
                        target["racket_tangent_w"] if "racket_tangent_w" in target else target["racket_tangent_base"]
                    )[hit_index].tolist(),
                    "racket_velocity_direction_w": (
                        target["racket_vel"][hit_index]
                        / max(float(np.linalg.norm(target["racket_vel"][hit_index])), 1e-9)
                    ).tolist(),
                }
    hit_time_from_start_s = float(hit_index / source_fps) if source_fps > 0 else None
    hit_frame_float = float(hit_time_from_start_s * fps) if hit_time_from_start_s is not None else None
    hit_frame = int(np.floor(hit_frame_float)) if hit_frame_float is not None else None
    hit_subframe_alpha = float(hit_frame_float - hit_frame) if hit_frame_float is not None and hit_frame is not None else None
    return {
        "target_spec_json": str(spec_path),
        "hit_event": {
            "source_hit_index": hit_index,
            "source_fps": source_fps,
            "hit_time_from_start_s": hit_time_from_start_s,
            "motion_fps": int(fps),
            "motion_hit_frame": hit_frame,
            "motion_hit_subframe_alpha": hit_subframe_alpha,
        },
        "strike_target": {
            key: hit_target[key]
            for key in (
                "racket_position_m",
                "racket_quat_xyzw",
                "racket_normal_w",
                "racket_tangent_w",
                "racket_velocity_mps",
                "racket_velocity_direction_w",
                "ball_position_m",
                "ball_in_velocity_mps",
                "ball_out_velocity_mps",
                "ball_to_racket_center_distance_m",
            )
            if key in hit_target
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    jobs = json.loads(args.jobs_json.read_text(encoding="utf-8"))["jobs"]
    entries = []
    for job in jobs:
        p = Path(job["output_file"])
        z = np.load(p)
        fps = int(z["fps"][0])
        metadata = _hit_metadata(job, fps)
        entries.append(
            {
                "episode_id": str(job["output_name"]),
                "motion_npz": str(p.resolve()),
                "fps": fps,
                "joint_pos_shape": list(z["joint_pos"].shape),
                "joint_vel_shape": list(z["joint_vel"].shape),
                "body_pos_w_shape": list(z["body_pos_w"].shape),
                "body_quat_w_shape": list(z["body_quat_w"].shape),
                **metadata,
            }
        )

    manifest = {
        "stage": "optimized_motion_npz",
        "count": len(entries),
        "all_joint_pos_shape": sorted({tuple(x["joint_pos_shape"]) for x in entries}),
        "all_body_pos_w_shape": sorted({tuple(x["body_pos_w_shape"]) for x in entries}),
        "all_fps": sorted({x["fps"] for x in entries}),
        "entries": entries,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Optimized Motion NPZ Summary",
        "",
        f"- count: `{len(entries)}`",
        f"- fps: `{manifest['all_fps']}`",
        f"- joint_pos shape: `{manifest['all_joint_pos_shape']}`",
        f"- body_pos_w shape: `{manifest['all_body_pos_w_shape']}`",
        "",
        "## Output Directory",
        "",
        f"- `manifest`: `{args.output_manifest}`",
        "",
        "## Episodes",
        "",
    ]
    for entry in entries:
        lines.append(f"- `{entry['episode_id']}`")
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_manifest}")
    print(f"Wrote {args.output_summary}")


if __name__ == "__main__":
    main()
