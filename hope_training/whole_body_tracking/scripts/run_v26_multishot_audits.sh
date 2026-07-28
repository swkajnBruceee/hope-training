#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source setup_train_env.sh

checkpoint="logs/rsl_rl/agibot_a3_joint_coordinator_v22_wide_deep_stability_20260727/2026-07-27_20-39-41_v22_2d_support_from_zero_left004_wide004_knee042_256x1500/model_1499.pt"
test -f "${checkpoint}" || {
  echo "checkpoint not found: ${checkpoint}" >&2
  exit 1
}

video="${V26_VIDEO:-false}"
common=(
  task=HOPEA3JointCoordinatorV26MultiShotRearm
  algo=ppo_joint_coordinator
  headless=true
  "video=${video}"
  num_envs=1
  seed=0
  "checkpoint=${checkpoint}"
)

run_sequence() {
  local name="$1"
  local sequence="$2"
  local steps="$3"
  hope_isaac_py scripts/play.py \
    "${common[@]}" \
    "multi_shot_sequence=\"${sequence}\"" \
    "multi_shot_max_steps=${steps}" \
    "multi_shot_report=eval_outputs/joint_coordinator_v26/${name}.json" \
    "video_name=${name}"
}

run_sequence "same_0_0" "0,0" 800
run_sequence "different_0_4" "0,4" 850
run_sequence "mixed_0_4_2_5_1" "0,4,2,5,1" 2200
