# Mixed Held-Out Stance Contract Set (2026-07-16)

This manifest is an independent validation set for the stance contract. It is
**not** an active training manifest and must not be appended to K8/K24/K32
training runs.

## Composition

```text
forehand: 4 prepositioned
backhand: 4 fixed
total:    8
walking:  disabled
```

The forehand entries come from the soft-limit-aware constant stance-offset
bank. Their base pose is relocated before the strike cycle. The backhand
entries come from the independent fixed-base pool and retain their native base
pose. No entry contains a walking policy or a mid-swing root translation.

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

The dependency-free contract validator passed all eight entries, including
NPZ path checks, root-at-hit checks, constant root XY checks, offset arithmetic,
and reconstruction of the original world hit point. The Isaac loader smoke is a
separate runtime check. This archive is evidence for the data and interface
contract only; it does not prove that the current actor executes a footstep or
that walking and striking are integrated.

## Source manifests

```text
sample_motions/p2_fixed_forehand_heldout_stance_20260716/native_zero_residual_manifest.json
sample_motions/p2_fixed_backhand_current_pool_v2/native_zero_residual_manifest.json
```
