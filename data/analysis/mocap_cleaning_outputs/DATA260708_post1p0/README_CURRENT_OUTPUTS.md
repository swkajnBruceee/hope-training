# DATA260708 Post1p0 Output Registry

Date: 2026-07-14

This directory contains retarget outputs, source pools, and archived diagnostic
experiments for DATA260708.

## Current Sources

```text
retarget_p2_fixed_a3_forehand_combined_gate_v1/
  current accepted forehand source

retarget_p2_fixed_a3_backhand_expand4_v2/
  current backhand expansion source, batch v2

retarget_p2_fixed_a3_backhand_expand4_v3/
  current backhand expansion source, batch v3

retarget_p2_fixed_a3_backhand_expand4_v4/
  current backhand expansion source, batch v4

retarget_p2_fixed_a3_backhand_current_pool_v2/
  merged current backhand pool used to build accepted_backhand_manifest

accepted_manual_retarget_v3/
  retained diagnostic source only; do not train directly
```

## Source Pool / Candidate Index Inputs

```text
manifest.json
manifest.md
RETARGET_DATASET_MAP.md
metadata/
packed/
samples/
final_audit_split/
```

These are preserved as source/candidate-pool data and should not be deleted.

## Archive

Historical debug, old funnel, old comfort scan, old wrist/torso probe, and
superseded retarget outputs are under:

```text
_archive_not_for_training/20260714_superseded_outputs/
```

Do not train directly from archived outputs. Re-promote a subset only after
rerunning the current combined gate and visual review.
