# Competition Success Label Report

Input: `data/analysis/mocap_cleaning_outputs/DATA260708_longpost/packed/DATA260708_train_longpost_competition_table_relabel.npz`
Output: `data/analysis/mocap_cleaning_outputs/DATA260708_longpost/packed/DATA260708_train_longpost_competition_table_labeled_relabel.npz`

| Metric | Value |
|---|---:|
| Samples | 1053 |
| Landing detected | 1032 |
| Reliable success labels | 1032 |
| Success | 949 |
| Failure | 83 |
| Unknown | 21 |

## Rule

- Landing is the first post-hit frame near table height with vertical velocity changing from downward to upward.
- `liang01` success requires landing in P2 half: `1.37 <= x <= 2.74` and `-1.525 <= y <= 0`.
- `gao01` success requires landing in P1 half: `0 <= x <= 1.37` and `-1.525 <= y <= 0`.
- Samples without a detected landing remain `success=-1` instead of being forced to failure.
