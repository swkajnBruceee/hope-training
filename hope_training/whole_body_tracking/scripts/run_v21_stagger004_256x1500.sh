#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if pgrep -af '[s]cripts/train.py' >/dev/null; then
  echo "[v21] another training process is already running:"
  pgrep -af '[s]cripts/train.py'
  exit 1
fi

CHECKPOINT="/workspace/hopetmp/whole_body_tracking_logs/rsl_rl/agibot_a3_joint_coordinator_v12_recovery_curriculum_20260727/2026-07-27_13-00-45_joint_coordinator_v12_real_return100_warm_v2_model900_256x1500/model_0.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[v21] checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT_DIR}/setup_train_env.sh"

hope_isaac_py scripts/train.py \
  task=HOPEA3JointCoordinatorV21StaggerSupport \
  algo=ppo_joint_coordinator_v21_stagger_support \
  headless=true \
  logger=tensorboard \
  device=cuda:0 \
  seed=0 \
  num_envs=256 \
  max_iterations=1500 \
  checkpoint="${CHECKPOINT}" \
  +warm_start_actor_only=true \
  +warm_start_append_zero_policy_obs=true \
  run_name=v21_stagger004_com_capture_256x1500
