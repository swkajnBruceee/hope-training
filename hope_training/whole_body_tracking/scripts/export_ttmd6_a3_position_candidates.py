#!/usr/bin/env python3
"""Export TTMD6 paddle-position candidates in explicitly named A3-base hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ORIENTATION_NAMES = (
    "velocity_plane_pos",
    "velocity_plane_neg",
    "upright_plane_pos",
    "upright_plane_neg",
)

LATERAL_MAPPINGS = {
    "minus_y": (-1.0, "source_right_to_a3_minus_y"),
    "plus_y": (1.0, "source_right_to_a3_plus_y"),
}


def map_local_to_a3(values: np.ndarray, lateral_sign: float, scale: float) -> np.ndarray:
    # Local source coordinates are [lateral, forward, up]. A3-base candidates
    # are [forward, lateral, up]. The sign is part of the locked coordinate
    # contract, not an orientation-quality label.
    return np.stack(
        [values[..., 1], lateral_sign * values[..., 0], values[..., 2]], axis=-1
    ) * scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("orientation_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-m-per-raw-unit", type=float, default=0.001)
    parser.add_argument(
        "--lateral-mapping",
        choices=("minus_y", "plus_y", "both"),
        default="minus_y",
        help="TTMD6 lateral mapping; minus_y is the locked production default.",
    )
    parser.add_argument(
        "--orientation-variant",
        choices=(*ORIENTATION_NAMES, "all"),
        default="velocity_plane_neg",
        help="Paddle orientation construction; velocity_plane_neg is the locked production default.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.orientation_manifest.read_text(encoding="utf-8"))
    output_root = args.output.parent / "a3_position_candidates"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.lateral_mapping == "both":
        lateral_mappings = tuple(LATERAL_MAPPINGS.values())
    else:
        lateral_mappings = (LATERAL_MAPPINGS[args.lateral_mapping],)
    orientation_names = ORIENTATION_NAMES if args.orientation_variant == "all" else (args.orientation_variant,)

    records = []
    for record in manifest["records"]:
        source = np.load(record["source_normalized_npz"])
        orientation = np.load(record["orientation_candidates_npz"])
        variants = []
        for lateral_sign, sign_name in lateral_mappings:
            for orientation_name in orientation_names:
                stem = f"{record['source_id']}__{sign_name}__{orientation_name}"
                path = output_root / f"{stem}.npz"
                np.savez_compressed(
                    path,
                    racket_target_pos_b=map_local_to_a3(
                        source["paddle_local_raw"], lateral_sign, args.scale_m_per_raw_unit
                    ),
                    racket_target_vel_b=map_local_to_a3(
                        source["paddle_velocity_local_mps_hypothesis"] / args.scale_m_per_raw_unit,
                        lateral_sign,
                        args.scale_m_per_raw_unit,
                    ),
                    racket_target_normal_b=map_local_to_a3(
                        orientation[f"{orientation_name}_normal_local"], lateral_sign, 1.0
                    ),
                    racket_target_tangent_b=map_local_to_a3(
                        orientation[f"{orientation_name}_tangent_local"], lateral_sign, 1.0
                    ),
                    candidate_frame_only=np.asarray(record["candidate_event_frame_only"], dtype=np.int64),
                )
                variants.append(
                    {
                        "variant_id": stem,
                        "path": str(path),
                        "lateral_mapping": sign_name,
                        "orientation_variant": orientation_name,
                        "coordinate_status": "hypothesis_only",
                        "unit_status": "hypothesis_only",
                        "hit_frame_status": "unassigned",
                        "ground_truth_orientation": False,
                        "training_eligible": False,
                        "a3_retarget_status": "not_started",
                    }
                )
        records.append(
            {
                "source_id": record["source_id"],
                "candidate_count": len(variants),
                "variants": variants,
            }
        )

    output = {
        "dataset": "TTMD6",
        "stage": "a3_base_position_candidates_v0",
        "input_manifest": str(args.orientation_manifest),
        "record_count": len(records),
        "candidate_count": sum(item["candidate_count"] for item in records),
        "lateral_mapping_policy": (
            "locked_source_right_to_a3_minus_y"
            if args.lateral_mapping == "minus_y"
            else f"explicit_{args.lateral_mapping}"
        ),
        "lateral_mapping_mode": args.lateral_mapping,
        "orientation_policy": (
            "locked_velocity_plane_neg"
            if args.orientation_variant == "velocity_plane_neg"
            else f"explicit_{args.orientation_variant}"
        ),
        "orientation_mode": args.orientation_variant,
        "coordinate_status": "hypothesis_only",
        "unit_status": "hypothesis_only",
        "hit_frame_status": "unassigned",
        "training_eligible": False,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"exported {output['candidate_count']} A3-base candidate variants")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
