#!/usr/bin/env bash
# Zero-adapter regression: V28 must reproduce the frozen V25/V27 single-shot
# behavior before any PPO update is permitted.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source setup_train_env.sh

CKPT="logs/rsl_rl/agibot_a3_joint_coordinator_v22_wide_deep_stability_20260727/2026-07-27_20-39-41_v22_2d_support_from_zero_left004_wide004_knee042_256x1500/model_1499.pt"
test -f "$CKPT" || { echo "missing V25/V27 baseline: $CKPT" >&2; exit 1; }

hope_isaac_py scripts/train.py \
  task=HOPEA3JointCoordinatorV28BentReadyRecenter \
  algo=ppo_joint_coordinator \
  headless=true \
  logger=tensorboard \
  device=cuda:0 \
  seed=0 \
  num_envs=6 \
  max_iterations=1 \
  checkpoint="$CKPT" \
  +warm_start_support_actor_only=true \
  +warm_start_append_zero_policy_obs=true \
  +audit_policy_action=true \
  +audit_full_episode=true \
  +audit_output=eval_outputs/v28_bent_ready_recenter/model0_zero_adapter_single_shot.json \
  run_name=v28_bent_ready_recenter_zero_adapter_smoke
