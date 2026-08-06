# V27 Bent READY: Multi-Shot Audit

## Purpose

V27 tests whether a compact right-arm READY pose reduces the time between
successive floating-base strikes.  It does not retrain the coordinator or
either frozen prior:

- Coordinator: V22 `model_1499.pt`.
- Lower-body prior: Stage-A `model_3396.pt`.
- Upper-body strike prior: fixed-base `model_900.pt`.
- Safety contract: V25 adaptive sagittal support and V26 re-arm.

The selected B READY pose uses `right_elbow_joint=0.35 rad`, which is about
61.2 degrees of physical flexion from a straight elbow.  It keeps the V25
50-step linear, zero-velocity prelude.  The first V27 version used a quintic
bridge and continuous velocity reference; that altered the frozen policy's
launch dynamics and is rejected as the default contract.

## Deterministic Results (Seed 0)

All reports use the same V25 checkpoint, 50 Hz control rate, and no physical
reset between shots.

| Sequence | Contract | Completion | Hit-to-READY steps | Interpretation |
| --- | --- | --- | --- | --- |
| `0,0` | V26 straight READY | 2/2 | 246, 245 | Baseline |
| `0,0` | V27 bent, return 100 | 2/2 | 242, 238 | 80 ms / 140 ms faster |
| `0,0` | V27 bent, return 80 | 2/2 | 222, 222 | 480 ms / 460 ms faster; second-hit position 8.77 cm |
| `5,1` | V26 straight READY | 2/2 | 299, 223 | Baseline |
| `5,1` | V27 bent, return 100 | 2/2 | 284, 235 | First recovery 300 ms faster; second recovery 240 ms slower |
| `5,1` | V27 bent, return 80 | 2/2 | 277, 273 | First recovery 440 ms faster; second recovery 1.0 s slower |

Reports:

- `eval_outputs/v27_bent_ready/multishot_linear_legacy_same_0_0.json`
- `eval_outputs/v27_bent_ready/multishot_linear_legacy_transition_5_1.json`
- `eval_outputs/v27_bent_ready/multishot_linear_return80_same_0_0.json`
- `eval_outputs/v27_bent_ready/multishot_linear_return80_transition_5_1.json`

## Five-Shot Stress Result

The `0,4,2,5,1` sequence is not yet qualified.

- V26 straight READY: completes shots 0--3 and falls during shot 4 (`motion 1`) at step 1617.
- V27 bent READY, return 100: completes shots 0--2 and falls during shot 3 (`motion 5`) at step 1454.

The bent arm therefore improves recovery for some isolated transitions but
does not yet improve long-horizon state accumulation.  It must not be
described as a general multi-shot stability improvement yet.

## Decision

Use V27's bent READY with the 50-step linear V25 bridge as the conservative
candidate.  Keep `return_to_default_steps=100` as the default.

Do not globally set the return duration to 80 steps.  The scan proves that
the shorter return is safe for `0,0` and `5,1`, but it also shows that its
benefit depends on the state entering the next swing.  The next controller
change should make return speed state-dependent using the existing re-arm
signals (capture-point margin, root forward velocity, pitch rate, and arm
ready error), then re-run the five-shot matrix.
