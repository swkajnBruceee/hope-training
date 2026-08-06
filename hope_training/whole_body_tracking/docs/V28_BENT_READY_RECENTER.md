# V28 Bent-READY Recenter Adaptation

## Purpose

V28 is a local recovery adaptation for the V25/V27 stack. It improves the
post-hit return to the bent-READY manifold and reduces the time before the
next swing can be armed. It is not a new strike policy or a replacement for
the Stage-A stabilizer.

## Frozen Contracts

- `model_3396`: frozen Stage-A lower-body prior.
- `model_900`: frozen fixed-base upper-body strike prior.
- V25 runtime support contract: sagittal front-risk gain `1.25`, centered
  zero-crossing exit, five-step sagittal residual decay.
- V27 bent-READY pose and its qualified 50-step linear, zero-velocity bridge.
- Existing hit-time position, velocity, and normal objectives.

## New Policy Branch

The V28 actor uses the frozen 227-D V25/V27 policy and adds an 8-D physical
post-hit recovery observation. Its new adapter is zero-initialized, so model
zero is behaviorally identical to the V25/V27 baseline.

The adapter may modify only these coordinator channels:

- Left/right hip pitch, knee, and ankle pitch.
- Waist pitch.
- Right shoulder pitch, right shoulder yaw, and right elbow.

It is gated off through prelude, swing, impact, and follow-through. It acts
only during return and READY. Roll/yaw support channels remain under the
qualified V25 policy.

## Recovery Observation

The extra eight values are all runtime-available feedback quantities:

1. Capture-point velocity.
2. Capture-point position relative to sagittal support center.
3. Body-frame forward root velocity.
4. Body-frame pitch rate.
5. Right-arm distance to bent-READY joint pose.
6. Maximum right-arm joint speed.
7. Stage-A re-arm stable-step fraction.
8. Post-hit recovery gate.

No motion ID, target world position, future action, or future simulated state
is exposed to the adapter.

## Reward Scope

V28 preserves the qualified V2/V25 strike rewards. It adds only return-phase
terms:

- Quiet right arm near the configured bent-READY manifold.
- Progress toward lower capture error, low root velocity, low root angular
  velocity, and low arm pose/velocity error.

The inherited full-cycle safety terms remain active. Position, speed, and
normal rewards are not weakened to obtain apparent stability.

## Acceptance Sequence

1. Zero-adapter deterministic V28 rollout must reproduce V27/V25 single-shot
   safety and hit metrics within floating-point tolerance.
2. Short single-shot recovery training must reduce `time_to_rearm_sec` while
   preserving six of six exact-hit position passes and V25 safety.
3. Two-shot `0 -> 0` and `5 -> 1` must improve or preserve the V27 baseline.
4. Only then evaluate the five-shot sequence `0 -> 4 -> 2 -> 5 -> 1`.

Checkpoints are ranked first by continuous-cycle safety, then hit pass rate,
then re-arm time. A faster return is rejected if it reduces safety.
