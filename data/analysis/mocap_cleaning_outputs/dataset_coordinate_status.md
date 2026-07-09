# Dataset Coordinate Status

Generated: 2026-07-08

## Policy

Competition analysis must use the confirmed competition table coordinate frame. I am not reusing the `DATA260708` table transform for `DATA260703`, because the two Motive world-to-table relationships are not guaranteed to be identical. Without table anchors in `DATA260703`, that transform would be an underconstrained inference rather than a precise calibration.

Competition frame definition for `DATA260708`:

| marker | target coordinate |
|---|---|
| `table:Marker 001` | `(0, 0, 0)` |
| `table:Marker 002` | `(2.74, 0, 0)` |
| `table:Marker 003` | `(2.74, -1.525, 0)` |
| `table:Marker 004` | `(0, -1.525, 0)` |

X points from P1/liang to P2. Y positive points to P1's left, so the P1 right corner has negative Y. Z points upward and tabletop height is 0.

## Table-Frame Usable

| dataset | table-frame samples | output |
|---|---:|---|
| DATA260708 | 1053 | `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/packed/DATA260708_train_post1p0_competition_table_labeled_relabel.npz` |

Supporting outputs:

| item | path |
|---|---|
| Motive-frame packed data | `data/analysis/mocap_cleaning_outputs/DATA260708/packed/DATA260708_train.npz` |
| Earlier marker-estimated local table data | `data/analysis/mocap_cleaning_outputs/DATA260708/packed/DATA260708_train_table.npz` |
| Competition table-frame packed data | `data/analysis/mocap_cleaning_outputs/DATA260708/packed/DATA260708_train_competition_table.npz` |
| Competition table-frame relabeled data | `data/analysis/mocap_cleaning_outputs/DATA260708/stroke_relabel/DATA260708_train_competition_table_relabel.npz` |
| Recommended competition labeled + relabeled data | `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/packed/DATA260708_train_post1p0_competition_table_labeled_relabel.npz` |
| Recommended analysis core subset | `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/final_audit_split/DATA260708_train_post1p0_competition_table_labeled_relabel_analysis_core.npz` |
| QC review subset | `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/final_audit_split/DATA260708_train_post1p0_competition_table_labeled_relabel_qc_review.npz` |
| Final audit report | `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/final_audit_split/final_dataset_audit_report.md` |
| First analysis report | `data/analysis/reports/DATA260708_competition_core_analysis/competition_core_analysis_report.md` |
| Competition table transform report | `data/analysis/mocap_cleaning_outputs/DATA260708/competition_table_transforms/table_transform_report.json` |
| Ball reconstruction report | `data/analysis/mocap_cleaning_outputs/DATA260708/ball_reconstruction/unlabeled_ball_reconstruction_report.json` |
| Competition validation report | `data/analysis/mocap_cleaning_outputs/DATA260708/packed/competition_table_validation/validation_report.md` |
| Success label report | `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/packed/competition_success_label_report.md` |

DATA260708 summary:

| metric | value |
|---|---:|
| packed Motive-frame samples before table filtering | 1099 |
| table-frame samples retained | 1053 |
| samples dropped without valid table transform | 46 |
| CSV files with valid table geometry | 86 |
| CSV files with invalid/missing table geometry | 3 |
| unlabeled-ball files usable for hit candidates | 86 |
| unlabeled-ball files needing inspection | 3 |
| ball id-rule links | 63 |
| ball non-rule continuity links | 450 |

Competition success labels:

| label | count |
|---|---:|
| success | 950 |
| failure | 84 |
| unknown, no landing detected in current window | 19 |

The recommended labeled dataset uses a longer hit-centered window from `-0.6s` to `+1.0s`. Success/failure labels are reliable only when a post-hit table bounce is detected inside the sample window. Unknown samples are not forced to failure. The remaining unknown samples show abnormal or incomplete ball trajectories and should remain quality-control cases rather than forced labels.

Final subset split:

| subset | count | use |
|---|---:|---|
| analysis_core | 1034 | success/failure, landing, player comparison, competition-frame analysis |
| training_motion | 0 | currently empty because all non-QC samples have reliable labels |
| qc_review | 19 | manual inspection or exclusion |

Stroke relabel result on the table-frame data:

| label | count |
|---|---:|
| forehand | 507 |
| backhand | 335 |
| unknown | 211 |

Racket reference point:

`racket_pos` is the Motive rigid-body center inside the racket, not the physical ball-contact point on the racket face. Therefore ball-racket distance at hit is measured from ball center to racket rigid-body center, and a nonzero distance is expected.

## Not Table-Frame Transformable

| dataset | processed output | status |
|---|---|---|
| DATA260703_combined | `data/analysis/mocap_cleaning_outputs/DATA260703_combined/packed/DATA260703_combined_train.npz` | local/body-relative use only |

`DATA260703` can still be used for non-table analyses: stroke/motion pattern analysis in local or body-relative coordinates, player/racket timing inspection, and algorithm prototyping that does not require landing position, table side, or success labels. It should not be mixed with `DATA260708_train_post1p0_competition_table_labeled_relabel.npz` for table-frame competition analysis.
