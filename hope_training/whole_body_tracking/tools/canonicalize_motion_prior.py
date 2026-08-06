#!/usr/bin/env python3
"""Build a base-heading-local motion-prior package from a source manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.utils.motion_prior_canonical import write_canonical_motion_package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = write_canonical_motion_package(arguments.manifest, arguments.output_dir)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
