#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if pgrep -af '[s]cripts/train.py' >/dev/null; then
  echo "[v23] another training process is already running:"
  pgrep -af '[s]cripts/train.py'
  exit 1
fi

CHECKPOINT="logs/rsl_rl/agibot_a3_joint_coordinator_v22_wide_deep_stability_20260727/2026-07-27_20-39-41_v22_2d_support_from_zero_left004_wide004_knee042_256x1500/model_1499.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[v23] checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT_DIR}/setup_train_env.sh"

hope_isaac_py scripts/train.py \
  task=HOPEA3JointCoordinatorV23CaptureRecovery \
  algo=ppo_joint_coordinator_v23_capture_recovery \
  headless=true \
  logger=tensorboard \
  device=cuda:0 \
  seed=0 \
  num_envs=256 \
  max_iterations=1000 \
  checkpoint="${CHECKPOINT}" \
  +warm_start_support_actor_only=true \
  +warm_start_append_zero_policy_obs=true \
  run_name=v23_state_conditioned_recovery_warm_v22_model1499_256x1000
