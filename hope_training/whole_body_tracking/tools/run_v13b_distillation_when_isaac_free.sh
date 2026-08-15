#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source setup_train_env.sh >/dev/null 2>&1

while pgrep -f '[s]cripts/train.py' >/dev/null || \
      pgrep -f '[m]onitor_v13b_precision_rescue_upper_gate.py' >/dev/null; do
  echo "[$(date '+%F %T')] waiting for active Isaac training/monitor process" >&2
  sleep 60
done

hope_isaac_py tools/generate_v13b_teacher_distillation_rollouts.py \
  --goal-index eval_outputs/v13b_teacher_distillation/v13b_teacher_distillation_goal_index.json \
  --checkpoint /home/bistu/桌面/HOPETableTennis/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_target_conditioned_reference_free_v13b_complete_priors_rightfront_v1/2026-08-09_18-10-06_v13b_resetfixed_model18900_clean_23118_rightfront_16384x50000_resume_from2300_exact/model_5000.pt \
  --motion-manifest /home/bistu/桌面/HOPETableTennis/a3_ik_point_offline_wrapper_v2/training_reference_bank_merged_20260807/training_manifest.json \
  --output-dir eval_outputs/v13b_teacher_distillation/rollouts_model5000 \
  --start 0 --count 23118 --batch-envs 128 --max-steps 600 \
  --progress 0.1000020000400008 --device cuda:1
