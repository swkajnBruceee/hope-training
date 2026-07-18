#!/usr/bin/env python3
"""Create one immutable, base-frame A3 strike target specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from a3_strike_contract import normalized_target_payload, target_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON object with source_dataset, source_episode_id, stroke_type, hit_time_s, base-frame target fields, and racket_mount_contract_id.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("target input must be a JSON object")
    # The creator is the single permitted normalization point.  Subsequent
    # validation rejects non-unit normals, so evaluators cannot silently alter
    # a target while loading it.
    if "racket_normal_b" in source:
        normal = np.asarray(source["racket_normal_b"], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if normal.shape != (3,) or not np.isfinite(norm) or norm <= 1.0e-9:
            raise ValueError("racket_normal_b must be a finite non-zero 3-vector")
        source["racket_normal_b"] = [float(x) for x in normal / norm]
    payload = normalized_target_payload(source)
    # Preserve world/source representations as evidence, but never substitute
    # them into the immutable base-frame command target.
    for optional in ("racket_position_w_m", "racket_velocity_w_mps", "racket_normal_w", "source_frame", "normal_semantics", "hit_time_interpolation"):
        if optional in source:
            payload[optional] = source[optional]
    payload["source_target_sha256"] = target_sha256(payload)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
