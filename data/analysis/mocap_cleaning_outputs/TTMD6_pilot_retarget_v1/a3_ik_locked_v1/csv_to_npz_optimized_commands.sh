#!/usr/bin/env bash
set -euo pipefail

source hope_training/whole_body_tracking/setup_train_env.sh
hope_isaac_py hope_training/whole_body_tracking/scripts/csv_to_npz.py --batch_jobs_json data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v1/a3_ik_locked_v1/csv_to_npz_optimized_jobs.json --input_fps 120 --output_fps 120 --robot agibot_a3 --headless
