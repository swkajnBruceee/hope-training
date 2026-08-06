#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/extracted/mc [output-dir]" >&2
  exit 2
fi

MC_ROOT="$(realpath "$1")"
OUT_DIR="${2:-$(pwd)/dual_stroke_example_output}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"

cmake -S "$ROOT" -B "$BUILD" -DMC_ROOT="$MC_ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j

python3 "$ROOT/scripts/generate_dual_stroke_dataset.py" \
  --binary "$BUILD/a3_generate_strike_reference" \
  --manifest "$ROOT/examples/dual_stroke_manifest.json" \
  --ready-forehand "$ROOT/examples/ready_forehand_template.yaml" \
  --ready-backhand "$ROOT/examples/ready_backhand_phase0.yaml" \
  --planner-config "$MC_ROOT/arm/hit_ik_point.yaml" \
  --robot-xml "$MC_ROOT/models/hit/kinematics/a3_t2d5.xml" \
  --output-root "$OUT_DIR" \
  --control-hz 100 \
  --csv-to-npz "$ROOT/scripts/csv_to_npz.py" \
  --continue-on-error
