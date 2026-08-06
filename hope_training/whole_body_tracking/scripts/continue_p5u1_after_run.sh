#!/usr/bin/env bash
set -u

RUN="/home/bistu/桌面/HOPETableTennis/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_p5u1_reward_r2/2026-08-04_21-24-54_p5u1_adaptive_fall_assist_1500r_4096x2000_v1"
INITIAL_PID="462971"
ROOT="/home/bistu/桌面/HOPETableTennis/hope_training/whole_body_tracking"
LOG="/home/bistu/桌面/HOPETableTennis/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_p5u1_reward_r2/continuation_launcher.log"

printf '[%s] waiting for initial training PID %s to finish\n' "$(date -Is)" "$INITIAL_PID" >> "$LOG"
while kill -0 "$INITIAL_PID" 2>/dev/null; do
    sleep 500
done

printf '[%s] initial training ended; waiting for model_1999.pt\n' "$(date -Is)" >> "$LOG"
for _ in $(seq 1 120); do
    if [ -s "$RUN/model_1999.pt" ]; then
        break
    fi
    sleep 5
done
if [ ! -s "$RUN/model_1999.pt" ]; then
    printf '[%s] ERROR: model_1999.pt was not found; continuation was not started\n' "$(date -Is)" >> "$LOG"
    exit 2
fi

printf '[%s] starting 2000 additional iterations from model_1999.pt\n' "$(date -Is)" >> "$LOG"
cd "$ROOT"
source ./setup_train_env.sh >> "$LOG" 2>&1
hope_isaac_py scripts/train.py \
    task=HOPEA3FloatingUnifiedUpperReferenceTrackerR2 \
    algo=ppo \
    headless=true \
    device=cuda:0 \
    num_envs=4096 \
    max_iterations=2000 \
    seed=7 \
    motion_manifest=eval_outputs/strike_goal_p5/p5d2_dataset_v1/p5d2_train_manifest.json \
    logger=tensorboard \
    run_name=p5u1_adaptive_fall_assist_continuation_2000r \
    resume=true \
    checkpoint=logs/rsl_rl/agibot_a3_p5u1_reward_r2/2026-08-04_21-24-54_p5u1_adaptive_fall_assist_1500r_4096x2000_v1/model_1999.pt \
    legacy_fall_strategy=true \
    task.env.episode_length_s=10.0 >> "$LOG" 2>&1
