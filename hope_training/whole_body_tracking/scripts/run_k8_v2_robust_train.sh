#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source setup_train_env.sh

MANIFEST="${MANIFEST:-sample_motions/p2_fixed_balanced_k8_torso_control_v2_nativecal_20260716/manifest.json}"
RUN_NAME="${RUN_NAME:-k8_torso_control_v2_robust_mild_20260716}"
NUM_ENVS="${NUM_ENVS:-512}"
MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
SEED="${SEED:-7}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.05}"
WAIST_SCALE_MULTIPLIER="${WAIST_SCALE_MULTIPLIER:-1.0}"
NATIVE_ACTUATOR_PROFILE="${NATIVE_ACTUATOR_PROFILE:-official_pd}"

echo "[robust-train] manifest=${MANIFEST}"
echo "[robust-train] residual_scale=${RESIDUAL_SCALE} envs=${NUM_ENVS} iterations=${MAX_ITERATIONS} seed=${SEED}"
echo "[robust-train] waist residual authority multiplier=${WAIST_SCALE_MULTIPLIER}"
echo "[robust-train] native actuator profile=${NATIVE_ACTUATOR_PROFILE}"
echo "[robust-train] reset perturbation: mild pose/velocity/joint noise"

extra_overrides=()
if [[ "${WAIST_SCALE_MULTIPLIER}" != "1.0" ]]; then
  extra_overrides+=(
    "+task.actions.native_joint_scale_multipliers={waist_yaw_joint: ${WAIST_SCALE_MULTIPLIER}, waist_roll_joint: ${WAIST_SCALE_MULTIPLIER}, waist_pitch_joint: ${WAIST_SCALE_MULTIPLIER}}"
  )
fi

hope_isaac_py scripts/train.py \
  task=HOPEA3NativeStrikeManifest \
  algo=ppo \
  headless=true \
  logger=tensorboard \
  num_envs="${NUM_ENVS}" \
  max_iterations="${MAX_ITERATIONS}" \
  seed="${SEED}" \
  manifest_subset_size=8 \
  motion_manifest="${MANIFEST}" \
  manifest_frame_z_offset=0.76 \
  task.actions.native_residual_scale="${RESIDUAL_SCALE}" \
  task.actions.raw_clip=0.25 \
  +task.native_actuator_profile="${NATIVE_ACTUATOR_PROFILE}" \
  "${extra_overrides[@]}" \
  +task.motion.pose_range='{roll: [-0.03, 0.03], pitch: [-0.03, 0.03], yaw: [-0.05, 0.05]}' \
  +task.motion.velocity_range='{x: [-0.05, 0.05], y: [-0.05, 0.05], z: [-0.02, 0.02], roll: [-0.1, 0.1], pitch: [-0.1, 0.1], yaw: [-0.1, 0.1]}' \
  +task.motion.joint_position_range='[-0.02, 0.02]' \
  run_name="${RUN_NAME}"
