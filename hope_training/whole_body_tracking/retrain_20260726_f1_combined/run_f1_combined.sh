#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source setup_train_env.sh

NUM_ENVS="${NUM_ENVS:-256}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
HEADLESS="${HEADLESS:-true}"
WARM_START="${WARM_START:-/workspace/hopetmp/whole_body_tracking_logs/rsl_rl/agibot_a3_retrain_stage_a_20260725/2026-07-25_16-01-09_retrain_20260725_fresh_20260725_160103/model_1999.pt}"
META_ROOT="${META_ROOT:-$ROOT/retrain_20260726_f1_combined/runs}"

if [[ ! -f "$WARM_START" ]]; then
  echo "missing current-contract Stage-A warm start: $WARM_START" >&2
  exit 1
fi

UPPER_CHECKPOINT="/workspace/hopetmp/whole_body_tracking_logs/rsl_rl/agibot_a3_native_strike_manifest/2026-07-24_22-00-32_backhand_strike_only_v1_shoulders_lead12_res025_clip050_3000it/model_900.pt"
if [[ ! -f "$UPPER_CHECKPOINT" ]]; then
  echo "missing frozen model_900 checkpoint: $UPPER_CHECKPOINT" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
META_DIR="$META_ROOT/${STAMP}_f1_combined"
mkdir -p "$META_DIR"

CMD=(hope_isaac_py scripts/train.py
  task=HOPEA3F1Combined
  algo=ppo_f1_combined
  "headless=$HEADLESS"
  "device=$DEVICE"
  logger=tensorboard
  "num_envs=$NUM_ENVS"
  "max_iterations=$MAX_ITERATIONS"
  "seed=$SEED"
  "run_name=f1_combined_${STAMP}"
  resume=true
  "checkpoint=$(realpath "$WARM_START")")

printf '%q ' "${CMD[@]}" > "$META_DIR/command.sh"
printf '\n' >> "$META_DIR/command.sh"
git status --short --untracked-files=all > "$META_DIR/git_status.txt"
git diff --binary > "$META_DIR/git_diff.patch"
sha256sum \
  cfg/task/HOPEA3F1Combined.yaml \
  cfg/algo/ppo_f1_combined.yaml \
  scripts/train.py \
  training/tasks/base_locomotion/mdp/actions.py \
  training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py \
  sample_motions/p2_data260708_backhand_strike_only_v1/manifest.json \
  "$UPPER_CHECKPOINT" \
  "$WARM_START" \
  > "$META_DIR/input_hashes.sha256"

echo "[f1-combined] envs=$NUM_ENVS iterations=$MAX_ITERATIONS warm_start=$WARM_START"
echo "[f1-combined] metadata=$META_DIR"
"${CMD[@]}"
