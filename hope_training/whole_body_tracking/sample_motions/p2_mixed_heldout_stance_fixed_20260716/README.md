# Mixed Held-Out Stance Contract Set (2026-07-16)

This manifest is an independent validation set for the stance contract. It is
**not** an active training manifest and must not be appended to K8/K24/K32
training runs.

## Composition

```text
forehand: 4 prepositioned
backhand: 2 fixed
total:    6
walking:  disabled
```

The forehand entries come from the soft-limit-aware constant stance-offset
bank. Their base pose is relocated before the strike cycle. The backhand
entries are the two independent fixed-base motions that remain outside the
active K8 set after current-environment native recalibration. No entry contains
a walking policy or a mid-swing root translation.

## Purpose

This set tests that one manifest and one loader can validate both supported
current modes without silently mixing their semantics:

- `fixed`: use the native fixed-base strike executor;
- `prepositioned`: use a constant pre-hit base pose and its base-relative
  strike target;
- `walking`: reserved and disabled.

The source manifests are preserved unchanged. The builder adds explicit fixed
mode metadata for the backhand entries and records the source manifests in the
top-level manifest.

## Validation status

The dependency-free contract validator passed all six entries, including
NPZ path checks, root-at-hit checks, constant root XY checks, offset arithmetic,
and reconstruction of the original world hit point. The Isaac loader smoke is a
separate runtime check. This archive is evidence for the data and interface
contract only; it does not prove that the current actor executes a footstep or
that walking and striking are integrated.

The current Isaac zero-residual evaluation passed all six motions:

```text
hit composite:       6/6
robot posture:       6/6
wrist naturalness:   6/6
whole cycle:         6/6
```

The K8 baseline checkpoint was then evaluated with the explicitly enabled
`native_residual_scale=0.05` and also passed 6/6. The nominal policy result is
only a held-out compatibility check; it does not replace the K8 perturbation
bank or establish medium/strong disturbance robustness.

## Source manifests

```text
sample_motions/p2_fixed_forehand_heldout_stance_20260716/native_zero_residual_manifest.json
eval_outputs/backhand_heldout_native_recalibrated_20260716/native_zero_residual_manifest.json
```
