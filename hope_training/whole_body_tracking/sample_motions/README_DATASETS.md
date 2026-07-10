# Sample Motion Dataset Map

## Active Training Library

- `p2_fixed_competition_global_funnel_balanced20/`
  - Current recommended training library.
  - Contains `40` motions total: `20` forehand and `20` backhand.
  - Use this manifest:

```text
hope_training/whole_body_tracking/sample_motions/p2_fixed_competition_global_funnel_balanced20/manifest.json
```

## Archived, Not For Training

- `_archive_not_for_training/`
  - Documentation for removed old training-library entry points.
  - Old library symlink trees were deleted to prevent accidental use of obsolete or broken links.
  - Their source manifests and generated outputs are archived under `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/_archive_not_for_training/`.

## Source Artifacts

The active balanced20 library symlinks to NPZ files under:

```text
data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_global_funnel
data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_global_funnel_existing_ik_supplement20
data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_global_funnel_supplement20
```

Do not move those source folders without regenerating the final manifest and this training library.
