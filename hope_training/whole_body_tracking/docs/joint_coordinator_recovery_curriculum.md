# Corrected Full-Cycle Recovery Curriculum

## Fixed contract

- One trainable 22-D joint coordinator: legs 12, waist 3, right arm 7.
- Frozen priors: Stage-A `model_3396` supplies leg support; fixed-base
  `model_900` supplies the strike prior.
- The V9 shoulder position lead and task-phase velocity feed-forward remain
  unchanged.
- The tail reference holds the final strike pose for 25 control steps, then
  returns every upper joint to the ready pose with a minimum-jerk trajectory.
  The final ready hold has zero target velocity.

## Why V9 is invalid as a recovery checkpoint

Before the tail-contract fix, the coordinator action bypassed
`MotionCommand.joint_pos` after the clip ended.  Waist and arm targets stayed
at the last strike frame instead of following the configured return.  V9 was
therefore trained on a different task from the full cycle it was meant to
solve.  Do not resume V9 checkpoints for recovery training.

## Stages

1. V12: 100-step return and 6.5 s episode.  Start actor-only from V6
   `model_999.pt`; reset critic, optimizer, and iteration.  Train all 22
   coordinator outputs jointly.
2. V13: 75-step return and 6.0 s episode.  Actor-only warm start from the best
   deterministic full-cycle V12 checkpoint.
3. V14: final 50-step return and 5.5 s episode.  Actor-only warm start from
   the best deterministic full-cycle V13 checkpoint.

## Promotion gate

Run the six-motion deterministic full-episode audit every 100 iterations.
Promote only when the current stage has at least 5/6 natural timeouts, no root
tilt above 30 degrees, no root-height termination, and no material regression
in strike position, velocity, or normal metrics.  Select checkpoints by this
audit, not by total training reward.

## Non-goals

- Do not reopen target-driven foot placement or root translation.
- Do not let the legacy Stage-A waist outputs control the waist.
- Do not increase residual authority unless a later audit shows persistent
  clipping.  V9 did not clip, so authority is not the current bottleneck.
