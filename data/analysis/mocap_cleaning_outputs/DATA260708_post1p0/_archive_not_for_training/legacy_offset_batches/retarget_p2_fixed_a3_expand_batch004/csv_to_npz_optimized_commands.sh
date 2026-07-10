#!/usr/bin/env bash
set -euo pipefail

source hope_training/whole_body_tracking/setup_train_env.sh
hope_isaac_py hope_training/whole_body_tracking/scripts/csv_to_npz.py --batch_jobs_json data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_expand_batch004/csv_to_npz_optimized_jobs.json --input_fps 200 --output_fps 50 --robot agibot_a3 --headless
