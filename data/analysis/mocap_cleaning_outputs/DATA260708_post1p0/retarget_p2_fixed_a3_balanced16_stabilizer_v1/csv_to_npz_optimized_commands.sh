#!/usr/bin/env bash
set -euo pipefail
python hope_training/whole_body_tracking/scripts/csv_to_npz.py --batch_jobs_json data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_balanced16_stabilizer_v1/csv_to_npz_optimized_jobs.json --input_fps 200 --output_fps 50 --robot agibot_a3 --headless
