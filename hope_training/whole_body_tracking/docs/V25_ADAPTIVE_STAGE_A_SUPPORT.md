# V25 Adaptive Stage-A Support Contract

## Purpose

Keep the frozen Stage-A (`model_3396`) sagittal support while it arrests a
forward swing impulse, without allowing the same residual to continue pushing
the robot through the rear support boundary.

This is a runtime contract around the frozen prior. It does not retrain
Stage-A, change the upper strike policy, or use a motion-ID-specific rule.

## Runtime Policy

The contract affects only Stage-A leg channels:

```text
left/right hip_pitch
left/right knee
left/right ankle_pitch
```

All roll/yaw outputs and the V22 coordinator remain active.

### Front-side risk support

After the hit phase begins, when both feet are in contact and both conditions
hold:

```text
capture front margin <= 0.07 m
body-frame forward velocity >= 0.02 m/s
```

the six Stage-A sagittal raw actions are multiplied by `1.25` before normal
action bounding. This increases urgency without exceeding the established
physical residual scale.

### Rear-side exit

After hit, Stage-A sagittal support is latched for retirement only after:

```text
positive forward velocity observed for >= 2 steps
capture point within +/-0.04 m of support center
forward velocity <= 0.03 m/s for >= 2 steps
both feet in contact
```

The selected channels then decay with smoothstep over 5 control steps and stay
off until the next episode reset. This prevents threshold chatter around the
velocity zero-crossing.

The current environment contains one swing per episode, so reset is the
validated re-arm boundary. Continuous multi-ball re-arm requires an explicit
multi-shot command lifecycle and must be tested separately; V25 does not
silently claim that capability.

## Evidence

Reference coordinator checkpoint:

```text
V22 model_1499
sha256:
849b994bb5d044f4ceeb7eec97aaf4de1e858538f3bec2a1d12867e631438a1a
```

Deterministic causal scans established:

```text
decay V23 recovery adapter: no material effect
decay V22 support adapter: worse
decay all coordinator: worse
decay only Stage-A sagittal channels: removes five rear falls
```

The formal V25 execution path was verified with full-cycle traces. For seed 0,
motion 4 triggered its rear exit at step 142 and completed the smooth 5-step
decay at step 147.

The front gain plus rear exit completed:

```text
seed 0: 6/6 safe
seed 1: 6/6 safe
seed 2: 6/6 safe
seed 3: 6/6 safe
seed 4: 6/6 safe
total : 30/30 safe
```

Mean exact-strike position error remained approximately `8.0-8.5 cm`; no scan
introduced a rearward-fall regression.

The preserved five-seed reports are:

```text
eval_outputs/joint_coordinator_v24/model1499_seed0_front_gain125_margin07.json
eval_outputs/joint_coordinator_v24/model1499_seed1_front_gain125_margin07.json
eval_outputs/joint_coordinator_v24/model1499_seed2_front_gain125_margin07.json
eval_outputs/joint_coordinator_v24/model1499_seed3_front_gain125_margin07.json
eval_outputs/joint_coordinator_v24/model1499_seed4_front_gain125_margin07.json
```

Recomputing directly from these reports gives:

```text
full-cycle safety: 30/30
mean exact-hit position error: 0.082455 m
position < 0.10 m: 25/30
```

## Remaining Limitation

The five-seed nominal matrix is now safe, but only `25/30` episodes are within
the current 10 cm exact-hit threshold. V25 is a stability-qualified runtime
contract, not a final accuracy-qualified controller. Any next change must be
evaluated against the same five-seed, six-motion full-cycle matrix.

## Files

```text
Task config:
cfg/task/HOPEA3JointCoordinatorV25AdaptiveStageASupport.yaml

Runtime action contract:
training/tasks/base_locomotion/mdp/actions.py

Full traces:
eval_outputs/joint_coordinator_v25/model1499_seed0_runtime_full_trace.json
eval_outputs/joint_coordinator_v25/model1499_seed3_runtime_trace.json
```
