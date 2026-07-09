# Competition Success Label Report

Input: `data/analysis/mocap_cleaning_outputs/DATA260708/packed/DATA260708_train_competition_table.npz`
Output: `data/analysis/mocap_cleaning_outputs/DATA260708/packed/DATA260708_train_competition_table_labeled.npz`

| Metric | Value |
|---|---:|
| Samples | 1053 |
| Landing detected | 252 |
| Reliable success labels | 252 |
| Success | 180 |
| Failure | 72 |
| Unknown | 801 |

## Rule

- Landing is the first post-hit frame near table height with vertical velocity changing from downward to upward.
- `liang01` success requires landing in P2 half: `1.37 <= x <= 2.74` and `-1.525 <= y <= 0`.
- `gao01` success requires landing in P1 half: `0 <= x <= 1.37` and `-1.525 <= y <= 0`.
- Samples without a detected landing remain `success=-1` instead of being forced to failure.
