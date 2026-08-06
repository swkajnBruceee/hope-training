# V26 No-Reset Multi-Shot Re-arm

## Scope

V26 preserves the complete V25 single-shot plant and runtime safety contract.
It adds only a fail-closed re-arm lifecycle for starting another reference
strike without resetting root, joints, velocities, contacts, or policy
history.

The frozen policy is still:

```text
V22 model_1499
sha256:
849b994bb5d044f4ceeb7eec97aaf4de1e858538f3bec2a1d12867e631438a1a
```

## State Contract

```text
0 SUPPORT_ACTIVE
1 DECAYING
2 SETTLED
3 READY
4 RAMPING
```

After V25's five-step sagittal exit, re-arm requires all conditions for 20
consecutive control steps:

```text
reference has completed hold + return-to-ready
both feet in contact
|capture point relative to support center| <= 0.05 m
|body-frame forward velocity| <= 0.06 m/s
|root pitch rate| <= 0.10 rad/s
root tilt <= 0.10 rad
max right-arm ready-pose error <= 0.15 rad
```

READY is revoked if any condition becomes false before the next shot. Starting
a shot outside READY is rejected and latched in audit telemetry. An accepted
shot ramps the six Stage-A sagittal channels from zero to the unchanged V25
contract over eight control steps.

## No-Teleport Command Lifecycle

`MotionCommand.begin_next_shot()` changes only:

```text
motion ID
motion phase
tail/prelude counters
shot-cycle generation
```

It never calls `write_joint_state_to_sim()` or `write_root_state_to_sim()`.
Consequently, failure accumulated between shots remains physically observable.

## Audit Commands

Same motion twice:

```bash
hope_isaac_py scripts/play.py \
  task=HOPEA3JointCoordinatorV26MultiShotRearm \
  algo=ppo_joint_coordinator \
  headless=true video=true num_envs=1 seed=0 \
  checkpoint=logs/rsl_rl/agibot_a3_joint_coordinator_v22_wide_deep_stability_20260727/2026-07-27_20-39-41_v22_2d_support_from_zero_left004_wide004_knee042_256x1500/model_1499.pt \
  'multi_shot_sequence="0,0"' \
  multi_shot_max_steps=700 \
  multi_shot_report=eval_outputs/joint_coordinator_v26/same_0_0.json \
  video_name=v26_same_0_0
```

Mixed five-shot sequence:

```bash
hope_isaac_py scripts/play.py \
  task=HOPEA3JointCoordinatorV26MultiShotRearm \
  algo=ppo_joint_coordinator \
  headless=true video=true num_envs=1 seed=0 \
  checkpoint=logs/rsl_rl/agibot_a3_joint_coordinator_v22_wide_deep_stability_20260727/2026-07-27_20-39-41_v22_2d_support_from_zero_left004_wide004_knee042_256x1500/model_1499.pt \
  'multi_shot_sequence="0,4,2,5,1"' \
  multi_shot_max_steps=1750 \
  multi_shot_report=eval_outputs/joint_coordinator_v26/mixed_0_4_2_5_1.json \
  video_name=v26_mixed_0_4_2_5_1
```

## Qualification

V26 is not qualified until actual simulation produces:

```text
no non-timeout termination
one exact-hit record per requested shot
READY reached between every pair of shots
no rejected re-arm
state sequence contains READY -> RAMPING -> SUPPORT_ACTIVE
all requested shots complete without physical reset
```

At implementation time the first smoke run was blocked before environment
creation by a system CUDA driver/context error. Static compilation and Hydra
task composition passed, but this is not a substitute for the required
physics audit.
