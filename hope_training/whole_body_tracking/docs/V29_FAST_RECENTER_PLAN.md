# V29 Fast Recenter / Time-to-Rearm

## Starting Point

V29 starts only from the archived V28 `model_100` stack:

- V28 recovery coordinator `model_100`.
- Frozen `model_3396` lower-body prior and `model_900` upper strike prior.
- Frozen V25 Stage-A support contract and V27 bent-READY bridge/return.

V28 is safe for a single shot and the audited two-shot sequences, but it is
not a five-shot policy. In `0 -> 4 -> 2 -> 5 -> 1`, shots 0, 4, and 2 re-arm;
motion 5 falls during return before the fourth re-arm. The recovery delays are
`4.92 s`, `6.58 s`, and `5.08 s`, respectively.

## Scope

Optimize only the post-hit return-to-bent-READY phase. Do not alter:

- The impact reference, shoulder position lead, or velocity feedforward.
- The V25 Stage-A front-risk boost, exit trigger, or five-step decay.
- The frozen lower/upper priors.
- Motion-specific target geometry or the core strike reward weights.

The priority order is:

1. Five-shot physical safety.
2. Correct fail-closed re-arm.
3. Exact-hit pass rate at least equal to V28.
4. Lower time-to-rearm.

## Instrumentation Gate

Before training, every multi-shot report must atomically persist at hit,
recovery milestones, state transitions, re-arm, and termination. For each
shot, record the first control step satisfying:

- right-arm bent-READY pose error <= `0.15 rad`;
- right-arm velocity <= `0.15 rad/s`;
- capture point within `+/- 0.05 m` of support center;
- body-frame forward velocity <= `0.06 m/s`;
- pitch rate <= `0.10 rad/s`;
- root tilt <= `0.10 rad`;
- bilateral contact;
- low leg joint velocity.

The V28 trace already shows that arm pose and arm velocity settle much earlier
than READY. V29 must therefore target whole-body recentering and the stable
hold, not merely faster arm return.

## RSI Snapshot Gate

Every V29 snapshot is taken only at `post_physics_pre_observation`: physics
data has been refreshed, while the next observation, actor call, and action
write have not begun. The snapshot schema must reject any other phase.

Before a candidate can enter the bank it must pass three gates:

1. **Gate A:** root/joint state, command phase, action targets, action history,
   Stage-A discrete latches, counters, and recovery-observation cache match at
   restore step zero.
2. **Gate B:** without advancing physics, the next deterministic observation,
   model_3396 action, model_900 action, V28 adapter action, coordinator action,
   and final joint target match the golden trace.
3. **Gate C:** a 20-step deterministic continuation retains every Stage-A
   branch/transition and has no growing physical divergence.

Candidate reports must be classified as `candidate`, `replay_verified`, or
`rejected`, with a first failure category such as `step0_physics_mismatch`,
`actor_action_mismatch`, `stage_a_branch_mismatch`, or `contact_divergence`.
If contact warm-start state prevents direct loading, the candidate must instead
use a deterministic reconstruction prefix from a verified earlier anchor.

## Recovery-Only RSI

Build an RSI bank only from V28 safe rollout states. Sample after impact at:

- hit + 10, +30, and +60 control steps;
- capture-point zero crossing / centered state;
- arm-ready but body-not-ready state;
- near-ready state with residual root or leg motion.

Do not sample failed or pre-impact states for this phase. The adapter must not
receive a route to change the strike trajectory.

## Reward Contract

Use dense progress terms during return:

- capture point moves toward support center;
- absolute forward velocity, pitch rate, and roll rate decrease;
- lower-body pose/velocity approaches the READY manifold;
- right arm reaches and remains near bent READY.

Give a larger success reward only after the complete re-arm contract holds.
Apply a small post-hit time cost while not ready. Keep strong behavior cloning
against V28 in `hit - 30` through short follow-through, especially for the
right arm and waist. Reject checkpoints that reduce recovery time by weakening
the swing or reducing safety margin.

## Acceptance

P0: five-shot audit has no physical termination through all five shots, every
shot re-arms, and exact-hit pass rate does not decline from V28.

P1: median hit-to-READY time is `<= 4.0 s` without reducing the above safety
conditions. A later target is `2.5-3.0 s`; do not lower numerical thresholds
to claim this result. `STRIKE_READY` may eventually overlap a low-speed next
bridge, but only after five-shot audits prove there is no accumulated state.

Motion 5 precision remains a separate branch. Its single-shot `10.78 cm`
error is not to be fixed by changing the V29 recovery objective.
