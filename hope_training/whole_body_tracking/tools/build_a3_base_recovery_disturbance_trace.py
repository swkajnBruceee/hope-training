#!/usr/bin/env python3
"""Build a deterministic, sign-balanced Recovery-A disturbance trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PROFILES = (
    (0, "clean", 0.0, 0.0),
    (1, "candidate", 0.035, 0.20),
    (2, "medium", 0.050, 0.30),
    (3, "upper", 0.075, 0.45),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _balanced_signed_samples(
    rng: np.random.Generator,
    count: int,
    abs_limit: float,
    *,
    phase: int,
) -> np.ndarray:
    """Return 2-D samples with exactly balanced sign quadrants."""
    if abs_limit == 0.0:
        return np.zeros((count, 2), dtype=np.float32)
    quadrants = np.asarray(((1, 1), (1, -1), (-1, 1), (-1, -1)), dtype=np.float32)
    signs = np.tile(quadrants, (int(np.ceil(count / 4)), 1))[:count]
    signs = np.roll(signs, shift=phase % count, axis=0)
    magnitudes = rng.uniform(0.15 * abs_limit, abs_limit, size=(count, 2)).astype(np.float32)
    samples = signs * magnitudes
    # Shuffle complete samples, preserving exact quadrant counts.
    return samples[rng.permutation(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples-per-profile", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    if args.samples_per_profile < 4 or args.samples_per_profile % 4:
        parser.error("--samples-per-profile must be a positive multiple of four")

    rng = np.random.default_rng(args.seed)
    profile_ids = []
    roll_pitch = []
    angular_velocity = []
    for profile_id, _name, pose_limit, angular_limit in PROFILES:
        profile_ids.append(np.full(args.samples_per_profile, profile_id, dtype=np.int8))
        roll_pitch.append(
            _balanced_signed_samples(
                rng, args.samples_per_profile, pose_limit, phase=profile_id
            )
        )
        angular_velocity.append(
            _balanced_signed_samples(
                rng, args.samples_per_profile, angular_limit, phase=profile_id + 1
            )
        )

    profile_id = np.concatenate(profile_ids)
    roll_pitch_rad = np.concatenate(roll_pitch)
    angular_velocity_rad_s = np.concatenate(angular_velocity)
    trace_index = np.arange(profile_id.size, dtype=np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray([1], dtype=np.int32),
        generator_seed=np.asarray([args.seed], dtype=np.int64),
        trace_index=trace_index,
        profile_id=profile_id,
        roll_pitch_rad=roll_pitch_rad,
        angular_velocity_rad_s=angular_velocity_rad_s,
    )

    profile_manifest = []
    for profile_id_value, name, pose_limit, angular_limit in PROFILES:
        mask = profile_id == profile_id_value
        pose = roll_pitch_rad[mask]
        angular = angular_velocity_rad_s[mask]
        profile_manifest.append(
            {
                "profile_id": profile_id_value,
                "name": name,
                "count": int(mask.sum()),
                "roll_pitch_abs_limit_rad": pose_limit,
                "angular_velocity_abs_limit_rad_s": angular_limit,
                "roll_pitch_sign_quadrants": {
                    "++": int(((pose[:, 0] > 0) & (pose[:, 1] > 0)).sum()),
                    "+-": int(((pose[:, 0] > 0) & (pose[:, 1] < 0)).sum()),
                    "-+": int(((pose[:, 0] < 0) & (pose[:, 1] > 0)).sum()),
                    "--": int(((pose[:, 0] < 0) & (pose[:, 1] < 0)).sum()),
                },
                "angular_velocity_sign_quadrants": {
                    "++": int(((angular[:, 0] > 0) & (angular[:, 1] > 0)).sum()),
                    "+-": int(((angular[:, 0] > 0) & (angular[:, 1] < 0)).sum()),
                    "-+": int(((angular[:, 0] < 0) & (angular[:, 1] > 0)).sum()),
                    "--": int(((angular[:, 0] < 0) & (angular[:, 1] < 0)).sum()),
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "trace_id": "a3_base_recovery_disturbance_trace_v1",
        "generator_seed": args.seed,
        "samples_per_profile": args.samples_per_profile,
        "profile_order": [profile[1] for profile in PROFILES],
        "profiles": profile_manifest,
        "npz_path": str(args.output),
        "npz_sha256": _sha256(args.output),
        "paired_evaluation_contract": {
            "passive_and_policy_must_use_identical_trace_index": True,
            "seed_alone_is_not_sufficient_identity": True,
            "upper_profile_is_diagnostic_only_until_separately_approved": True,
        },
        "training_distribution_approved": False,
        "deployment_approved": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
