# DATA260708 Retarget Dataset Map

This folder is split into active training data, active support artifacts, and archived non-training artifacts.

## Use For Training

- `retarget_p2_fixed_a3_global_funnel_balanced20/`
  - Final balanced tracking manifest.
  - Contains `20` forehand and `20` backhand motions.
  - This is the only recommended manifest for the current training run.
  - Training library entry:
    `hope_training/whole_body_tracking/sample_motions/p2_fixed_competition_global_funnel_balanced20/manifest.json`

## Active Support, Do Not Train Directly

These folders contain source NPZ files referenced by the final balanced20 manifest and library symlinks. Keep them in place unless the final manifest/library is regenerated with new paths.

- `retarget_p2_fixed_a3_global_funnel/`
  - Global candidate index, first funnel pass, and first replay-ready set.
  - Also contains `candidate_index/competition_retarget_candidate_index.csv`, the current full candidate profile table.

- `retarget_p2_fixed_a3_global_funnel_existing_ik_supplement20/`
  - Supplement generated from samples that had already passed IK but had not yet been optimized.
  - Used to top up the final balanced20 set.

- `retarget_p2_fixed_a3_global_funnel_supplement20/`
  - Backhand-targeted supplement generated after the existing IK-pass pool was not enough to reach the desired backhand count.
  - Used to top up the final balanced20 set.

## Archived, Not For Training

- `_archive_not_for_training/legacy_offset_batches/`
  - Old fixed-base and offset/batch outputs.
  - Kept only for debugging, comparison, and audit history.
  - Do not point training at these manifests.

Archived contents include:

- `retarget_p2_fixed_a3/`
  - Older fixed-base output under earlier standards.

- `retarget_p2_fixed_a3_expand*`
  - Earlier offset/batch expansion runs.
  - These were superseded by the global candidate index and funnel workflow.

- `retarget_p2_fixed_a3_expand_cumulative/`
  - Cumulative manifest from the older offset/batch workflow.
  - Superseded by `retarget_p2_fixed_a3_global_funnel_balanced20/`.

## Current Rule

For current experiments, use only:

```text
hope_training/whole_body_tracking/sample_motions/p2_fixed_competition_global_funnel_balanced20/manifest.json
```

Everything under `_archive_not_for_training/` is historical context only.
