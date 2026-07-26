#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source setup_train_env.sh

STAGE="${1:-}"
CHECKPOINT="${2:-}"
NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
HEADLESS="${HEADLESS:-true}"
META_ROOT="${META_ROOT:-$ROOT/retrain_20260725/runs}"
FROZEN_HASHES="$ROOT/retrain_20260725/frozen_input_hashes.sha256"

if [[ -z "$STAGE" ]]; then
  echo "usage: $0 {fresh|return|robust_b|f1} [checkpoint-for-continuation]" >&2
  exit 2
fi

if [[ ! -f "$FROZEN_HASHES" ]]; then
  echo "missing frozen input hash file: $FROZEN_HASHES" >&2
  exit 1
fi
if ! sha256sum -c "$FROZEN_HASHES" >/dev/null; then
  echo "frozen input verification failed; do not start training" >&2
  exit 1
fi

case "$STAGE" in
  fresh)
    TASK="HOPEA3RetrainStageA"
    ALGO="ppo_retrain_stage_a"
    MAX_ITERATIONS=2000
    SAVE_INTERVAL=25
    RESUME=false
    ;;
  return)
    TASK="HOPEA3RetrainReturn"
    ALGO="ppo_retrain_return"
    MAX_ITERATIONS=600
    SAVE_INTERVAL=25
    RESUME=true
    ;;
  robust_b)
    TASK="HOPEA3RetrainRobustB"
    ALGO="ppo_retrain_robust_b"
    MAX_ITERATIONS=300
    SAVE_INTERVAL=50
    RESUME=true
    ;;
  f1)
    TASK="HOPEA3RetrainF1"
    ALGO="ppo_retrain_f1"
    MAX_ITERATIONS=3000
    SAVE_INTERVAL=100
    RESUME=true
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

if [[ "$RESUME" == true ]]; then
  if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "continuation requires an existing checkpoint path: $CHECKPOINT" >&2
    exit 2
  fi
  CHECKPOINT="$(realpath "$CHECKPOINT")"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
META_DIR="$META_ROOT/${STAMP}_${STAGE}"
if [[ -e "$META_DIR" ]]; then
  echo "refusing to overwrite metadata directory: $META_DIR" >&2
  exit 1
fi
mkdir -p "$META_DIR"

CMD=(hope_isaac_py scripts/train.py
  "task=$TASK"
  "algo=$ALGO"
  "headless=$HEADLESS"
  "device=$DEVICE"
  "logger=tensorboard"
  "num_envs=$NUM_ENVS"
  "max_iterations=$MAX_ITERATIONS"
  "seed=$SEED"
  "run_name=retrain_20260725_${STAGE}_${STAMP}")
if [[ "$RESUME" == true ]]; then
  CMD+=(resume=true "checkpoint=$CHECKPOINT")
else
  CMD+=(resume=false)
fi

printf '%q ' "${CMD[@]}" > "$META_DIR/command.sh"
printf '\n' >> "$META_DIR/command.sh"
git status --short --untracked-files=all > "$META_DIR/git_status.txt"
git diff --binary > "$META_DIR/git_diff.patch"
sha256sum \
  cfg/task/HOPEA3RetrainStageA.yaml \
  cfg/task/HOPEA3RetrainReturn.yaml \
  cfg/task/HOPEA3RetrainRobustB.yaml \
  cfg/task/HOPEA3RetrainF1.yaml \
  cfg/algo/ppo_retrain_stage_a.yaml \
  cfg/algo/ppo_retrain_return.yaml \
  cfg/algo/ppo_retrain_robust_b.yaml \
  cfg/algo/ppo_retrain_f1.yaml \
  sample_motions/p2_data260708_backhand_strike_only_v1/manifest.json \
  > "$META_DIR/input_hashes.sha256"
if [[ "$RESUME" == true ]]; then
  sha256sum "$CHECKPOINT" >> "$META_DIR/input_hashes.sha256"
fi

echo "[retrain] stage=$STAGE envs=$NUM_ENVS iterations=$MAX_ITERATIONS"
echo "[retrain] metadata=$META_DIR"
echo "[retrain] command=$(<"$META_DIR/command.sh")"
"${CMD[@]}"
