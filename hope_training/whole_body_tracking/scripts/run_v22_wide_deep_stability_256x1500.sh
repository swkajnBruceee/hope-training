#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if pgrep -af '[s]cripts/train.py' >/dev/null; then
  echo "[v22] another training process is already running:"
  pgrep -af '[s]cripts/train.py'
  exit 1
fi

CHECKPOINT="logs/rsl_rl/agibot_a3_joint_coordinator_v21_stagger_support_20260727/2026-07-27_19-38-43_v21_stagger004_com_capture_256x1500/model_0.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[v22] checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT_DIR}/setup_train_env.sh"

hope_isaac_py scripts/train.py \
  task=HOPEA3JointCoordinatorV22WideDeepStability \
  algo=ppo_joint_coordinator_v22_wide_deep_stability \
  headless=true \
  logger=tensorboard \
  device=cuda:0 \
  seed=0 \
  num_envs=256 \
  max_iterations=1500 \
  checkpoint="${CHECKPOINT}" \
  +warm_start_actor_only=true \
  +warm_start_append_zero_policy_obs=true \
  run_name=v22_2d_support_from_zero_left004_wide004_knee042_256x1500
