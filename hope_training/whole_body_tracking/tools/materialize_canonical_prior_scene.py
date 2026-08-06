#!/usr/bin/env python3
"""Materialize a canonical prior at a versioned scene root pose."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.utils.motion_prior_scene_placement import (  # noqa: E402
    write_scene_placed_motion_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root-anchor-w-m", type=float, nargs=3, required=True)
    parser.add_argument("--root-heading-w-rad", type=float, required=True)
    parser.add_argument("--scene-frame-version", required=True)
    arguments = parser.parse_args()
    result = write_scene_placed_motion_package(
        arguments.canonical_manifest,
        arguments.output_dir,
        root_anchor_w_m=arguments.root_anchor_w_m,
        root_heading_w_rad=arguments.root_heading_w_rad,
        scene_frame_version=arguments.scene_frame_version,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
