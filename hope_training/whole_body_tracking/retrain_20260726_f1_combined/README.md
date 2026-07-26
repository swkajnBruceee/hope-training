# F1 Combined Floating-Base Training

## Objective

Train only the twelve leg residual channels while physically executing all six
backhand swings.  The frozen `model_900` owns the waist and right-arm strike;
the leg actor learns in-place support that reduces floating-base drift and
preserves exact-hit racket accuracy.  This is not a stepping or target-chasing
task.

## Frozen Control Contract

```text
motions: six-motion backhand manifest, sampled uniformly
root: floating; corrected current frame and validated flexed ready pose
prelude: 50 control steps
upper: model_900, frozen
upper lookahead: shoulder pitch/yaw +12 frames
prelude guard: no lead/residual during prelude, linear release over frames 0..12
leg policy: Base14 public action; only 12 leg channels are enabled
waist channels: masked in the trainable leg actor
post-hit tail: disabled
target-directed base translation and stepping: not enabled
random reset/PD randomization: disabled
```

The prelude guard prevents the old initialization failure: `model_900` cannot
apply its hit-oriented lookahead before the validated flexed ready pose has
been reached.  It does **not** remove the 12-frame shoulder lead at impact.

## Initial Checkpoint

Warm start from the current-coordinate Stage-A checkpoint:

```text
/workspace/hopetmp/whole_body_tracking_logs/rsl_rl/agibot_a3_retrain_stage_a_20260725/2026-07-25_16-01-09_retrain_20260725_fresh_20260725_160103/model_1999.pt
```

Do not use historical `model_3396.pt`: its observation and root-frame contract
is incompatible with this environment.

## Training Plan

One F1 run is fixed at `256` environments and `3000` iterations.  It uses the
reviewed F1 PPO and reward values unchanged, so changes in result can be
attributed to physical upper/lower integration rather than an optimizer or
reward rewrite.

Inspect checkpoints at iterations `100, 200, ... 3000`.  Promote a checkpoint
only if all six motions remain non-falling and it improves the paired
floating-zero position error without increasing persistent root displacement
or foot slip.  Exact-strike position, velocity, normal, root drift, foot
displacement, upper tracking RMSE, and action clipping are mandatory metrics.

## Launch

```bash
# One-iteration configuration smoke.  It also records all source/checkpoint hashes.
NUM_ENVS=8 MAX_ITERATIONS=1 bash retrain_20260726_f1_combined/run_f1_combined.sh

# Formal F1 run.
NUM_ENVS=256 MAX_ITERATIONS=3000 bash retrain_20260726_f1_combined/run_f1_combined.sh
```

Override `WARM_START=/absolute/path/to/model_N.pt` only with a checkpoint that
uses the same current 126-D observation and 14-D Base action contract.
