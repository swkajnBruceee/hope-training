#!/usr/bin/env python3
"""Construct explicitly non-ground-truth TTMD6 paddle orientation candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize_rows(values: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=1)
    valid = norms > eps
    out = np.zeros_like(values)
    out[valid] = values[valid] / norms[valid, None]
    return out, valid


def make_frame(handle: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal, valid_n = normalize_rows(np.cross(handle, reference))
    tangent, valid_t = normalize_rows(np.cross(normal, handle))
    valid = valid_n & valid_t
    return normal, tangent, valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    records = []
    for record in source["records"]:
        data = np.load(record["normalized_npz"])
        human = data["human_local_raw"]
        paddle = data["paddle_local_raw"]
        # Point 7 is the distal right-forearm proxy from the source-order
        # hypothesis. It is not a measured paddle handle marker.
        handle, handle_valid = normalize_rows(paddle - human[:, 7, :])
        velocity = np.gradient(paddle, 1.0 / float(record["fps"]), axis=0, edge_order=1)
        velocity_dir, velocity_valid = normalize_rows(velocity)
        up = np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), paddle.shape)

        variants = {}
        for name, reference in (
            ("velocity_plane_pos", velocity_dir),
            ("upright_plane_pos", up),
        ):
            normal, tangent, valid = make_frame(handle, reference)
            variants[name] = {
                "normal_local": normal,
                "tangent_local": tangent,
                "valid": valid,
            }
            variants[name.replace("_pos", "_neg")] = {
                "normal_local": -normal,
                "tangent_local": -tangent,
                "valid": valid,
            }

        candidate_frame = int(record["paddle_speed_peak_frame_candidate_only"])
        lo = max(0, candidate_frame - 5)
        hi = min(len(paddle) - 1, candidate_frame + 5)
        variant_summary = {}
        for name, variant in variants.items():
            valid = variant["valid"]
            variant_summary[name] = {
                "valid_frame_count": int(np.sum(valid)),
                "candidate_window_valid": bool(np.all(valid[lo : hi + 1])),
                "ground_truth": False,
                "construction_status": "constructed_heuristic",
            }

        stem = record["source_id"]
        out_path = args.output.parent / "clips" / f"{stem}_orientation_candidates.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            handle_proxy_local=handle,
            handle_proxy_valid=handle_valid,
            paddle_velocity_local=velocity,
            paddle_velocity_valid=velocity_valid,
            **{
                f"{name}_normal_local": variant["normal_local"]
                for name, variant in variants.items()
            },
            **{
                f"{name}_tangent_local": variant["tangent_local"]
                for name, variant in variants.items()
            },
            **{f"{name}_valid": variant["valid"] for name, variant in variants.items()},
        )
        records.append(
            {
                "source_id": stem,
                "source_normalized_npz": record["normalized_npz"],
                "orientation_candidates_npz": str(out_path),
                "candidate_event_frame_only": candidate_frame,
                "candidate_event_window_only": [lo, hi],
                "handle_proxy": "paddle_cog - source point 7 (distal right-forearm hypothesis)",
                "variants": variant_summary,
                "ground_truth_orientation": False,
                "hit_frame_status": "unassigned",
                "training_eligible": False,
                "retarget_status": "orientation_candidates_only",
            }
        )

    output = {
        "dataset": "TTMD6",
        "stage": "constructed_orientation_candidates_v0",
        "source_manifest": str(args.source_manifest),
        "record_count": len(records),
        "construction_is_ground_truth": False,
        "hit_frame_status": "unassigned",
        "training_eligible": False,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"constructed orientation candidates for {len(records)} clips")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
