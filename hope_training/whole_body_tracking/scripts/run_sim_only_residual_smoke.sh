#!/usr/bin/env bash
set -euo pipefail

# This experiment is intentionally not a deployment candidate.  The official
# A3 MOTION stack has not exposed a waist fusion point yet, so residual waist
# behavior is currently validated only against Isaac's fixed-base executor.

cd "$(dirname "$0")/.."
source setup_train_env.sh

MANIFEST="${MANIFEST:-sample_motions/p2_fixed_balanced_k8_torso_control_v2_nativecal_20260716/manifest.json}"
RUN_NAME="${RUN_NAME:-sim_only_residual_smoke_s005}"
NUM_ENVS="${NUM_ENVS:-512}"
MAX_ITERATIONS="${MAX_ITERATIONS:-300}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.05}"
SEED="${SEED:-7}"

echo "[sim-only] manifest=${MANIFEST}"
echo "[sim-only] residual_scale=${RESIDUAL_SCALE} num_envs=${NUM_ENVS} iterations=${MAX_ITERATIONS} seed=${SEED}"
echo "[sim-only] official waist fusion is unresolved; do not promote this checkpoint."

hope_isaac_py scripts/train.py \
  task=HOPEA3NativeStrikeManifest \
  algo=ppo \
  headless=true \
  logger=tensorboard \
  num_envs="${NUM_ENVS}" \
  max_iterations="${MAX_ITERATIONS}" \
  seed="${SEED}" \
  manifest_subset_size=8 \
  manifest_frame_z_offset=0.76 \
  motion_manifest="${MANIFEST}" \
  task.actions.native_residual_scale="${RESIDUAL_SCALE}" \
  task.actions.raw_clip=0.25 \
  run_name="${RUN_NAME}"
