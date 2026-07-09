# DATA260708 Final Competition Dataset Audit

Input: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/packed/DATA260708_train_post1p0_competition_table_labeled_relabel.npz`

## Summary

| Metric | Value |
|---|---:|
| Samples | 1053 |
| Frames | 321 |
| Success | 950 |
| Failure | 84 |
| Unknown | 19 |
| Analysis core | 1034 |
| Training motion | 0 |
| QC review | 19 |

## Subset Rules

- `analysis_core`: reliable success/failure label, finite hit-frame racket/ball data, hit distance <= 0.15 m, and normal landing height.
- `training_motion`: usable motion sample not in `analysis_core` or `qc_review`; may have unknown success label.
- `qc_review`: unknown success, abnormal landing height/ball trajectory, or suspicious hit distance. Valid failures are kept in `analysis_core`.

## Racket Reference Point

racket_pos is the Motive rigid-body center inside the racket, not the physical ball-contact point. Hit distance is ball center to racket rigid-body center, so nonzero distance is expected.

## Output Files

| Subset | Path |
|---|---|
| `analysis_core` | `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/final_audit_split/DATA260708_train_post1p0_competition_table_labeled_relabel_analysis_core.npz` |
| `training_motion` | `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/final_audit_split/DATA260708_train_post1p0_competition_table_labeled_relabel_training_motion.npz` |
| `qc_review` | `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/final_audit_split/DATA260708_train_post1p0_competition_table_labeled_relabel_qc_review.npz` |

## Success By Racket

| racket | success | failure | unknown | total |
|---|---:|---:|---:|---:|
| gao01 | 475 | 45 | 9 | 529 |
| liang01 | 475 | 39 | 10 | 524 |

## Key Percentiles

- `hit_dist_m`: p01=0.0196, p10=0.0295, p50=0.0404, p90=0.0630, p99=0.0883
- `landing_x_m`: p01=0.1810, p10=0.4935, p50=1.3965, p90=2.2560, p99=2.5861
- `landing_y_m`: p01=-1.4156, p10=-1.0607, p50=-0.7164, p90=-0.3914, p99=-0.1654
- `landing_z_m`: p01=-0.0190, p10=-0.0112, p50=-0.0012, p90=0.0070, p99=0.0144
- `ball_speed_mps`: p01=0.4268, p10=2.5789, p50=4.1681, p90=5.3195, p99=6.3752
- `racket_speed_mps`: p01=0.1248, p10=0.3321, p50=0.8720, p90=2.0728, p99=3.1705

## QC Review

QC rows: 19
QC CSV: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/final_audit_split/qc_review_samples.csv`
