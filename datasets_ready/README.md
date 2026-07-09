# Ready Datasets

This directory contains short symlinks to the processed datasets. The original pipeline outputs stay under `data/analysis/mocap_cleaning_outputs/`.

## Recommended Files

| file | samples | frame window | coordinate frame | use |
|---|---:|---|---|---|
| `DATA260708_competition_core.npz` | 1034 | `-0.6s` to `+1.0s`, 321 frames | `competition_table_m` | primary dataset for success/failure, landing, player comparison, and table-frame analysis |
| `DATA260708_competition_full.npz` | 1053 | `-0.6s` to `+1.0s`, 321 frames | `competition_table_m` | full new dataset, including 19 QC/unknown-success samples |
| `DATA260708_qc_review.npz` | 19 | `-0.6s` to `+1.0s`, 321 frames | `competition_table_m` | manual review or exclusion |
| `DATA260703_local_only.npz` | 792 | `-0.6s` to `+0.4s`, 201 frames | `motive_global_m` | local/body-relative or motion-only analysis; do not use for table-frame competition labels |

## Sample Definition

One sample is one detected hit event for one racket: `gao01` or `liang01`.

For `DATA260708_*`, each sample is hit-centered:

- `time_rel = 0` is the detected hit time.
- `hit_index = 120`.
- The recommended files keep `0.6s` before hit and `1.0s` after hit.
- Success/failure labels are based on the first post-hit table bounce detected in the competition table frame.

## Competition Table Frame

The confirmed `DATA260708` competition frame is:

| marker | coordinate |
|---|---|
| `table:Marker 001` | `(0, 0, 0)` |
| `table:Marker 002` | `(2.74, 0, 0)` |
| `table:Marker 003` | `(2.74, -1.525, 0)` |
| `table:Marker 004` | `(0, -1.525, 0)` |

X points from P1/liang to P2. Y positive points to P1's left, so the P1 right corner is negative Y. Z points upward and tabletop height is `0`.

## Label Meaning

`success` encoding:

- `1`: ball landed in the opponent half according to the table-frame rule.
- `0`: landing was detected but did not land in the expected opponent half.
- `-1`: landing/success unknown; do not force to failure.

Current `DATA260708_competition_core.npz` distribution:

- success: 950
- failure: 84
- unknown: 0

Current `DATA260708_competition_full.npz` distribution:

- success: 950
- failure: 84
- unknown: 19

## Racket Reference Point

`racket_pos` is the Motive rigid-body center inside the racket, not the physical contact point on the racket face.

Therefore `dist` / hit distance is ball center to racket rigid-body center. A nonzero hit distance is expected and must not be interpreted as contact error.

## Key Fields

| field | meaning |
|---|---|
| `ball_pos`, `ball_vel` | ball trajectory and velocity in the current coordinate frame |
| `racket_pos`, `racket_quat`, `racket_vel`, `racket_omega` | racket rigid-body state |
| `body_center`, `body_right_axis` | player body reference derived from skeleton bones |
| `hit_pos`, `racket_pose_at_hit`, `racket_vel_at_hit` | hit-frame features |
| `landing_pos`, `landing_index`, `landing_side`, `success` | landing and success labels |
| `stroke_type_rule_v2` | body-local forehand/backhand relabel |
| `*_motive` | original Motive world-coordinate backup fields |
| `source_json`, `quality_flags_json`, `dataset_attrs_json` | provenance, quality flags, dataset metadata |

## Reports

- Final audit: `../data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/final_audit_split/final_dataset_audit_report.md`
- QC list: `../data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/final_audit_split/qc_review_samples.csv`
- First analysis report: `../data/analysis/reports/DATA260708_competition_core_analysis/competition_core_analysis_report.md`
- Coordinate status: `../data/analysis/mocap_cleaning_outputs/dataset_coordinate_status.md`
