# DATA260703 Retarget-Ready Split

- dataset: `analysis/mocap_cleaning_outputs/DATA260703_combined/stroke_relabel/DATA260703_combined_train_stroke_relabel.npz`
- label field: `stroke_type_rule_v2`
- min confidence: `0.85`
- keep unknown: `False`
- selected samples: `705`

## Label Counts

| label | kept | seen |
|---|---:|---:|
| backhand | 287 | 298 |
| forehand | 418 | 419 |
| unknown | 0 | 75 |

## Source CSV Counts

| source | count |
|---|---:|
| Csv/Point/Table Tennis_01_004.csv | 62 |
| Csv/Point/Table Tennis_01_012.csv | 41 |
| Csv/Point/Table Tennis_01_013.csv | 134 |
| Csv/Point/Table Tennis_01_014.csv | 159 |
| Csv/Rige Body/Table Tennis_01_005.csv | 32 |
| Csv/Rige Body/Table Tennis_01_006.csv | 72 |
| Csv/Rige Body/Table Tennis_01_007.csv | 50 |
| Csv/Rige Body/Table Tennis_01_008.csv | 50 |
| Csv/Rige Body/Table Tennis_01_009.csv | 105 |

## Outputs

- `retarget_manifest_json`: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_ready/retarget_manifest.json`
- `retarget_samples_csv`: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_ready/retarget_samples.csv`
- `backhand_manifest_json`: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_ready/backhand_manifest.json`
- `forehand_manifest_json`: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_ready/forehand_manifest.json`
- `low_confidence_review_manifest_json`: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_ready/low_confidence_review_manifest.json`
- `unknown_review_manifest_json`: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_ready/unknown_review_manifest.json`

## Notes

- `forehand` and `backhand` manifests are the current retarget queue.
- `unknown` is excluded from the main queue by default, but exported separately for review.
- Low-confidence known-label samples are also exported separately for review.
- This split still uses Motive global meters; table/world success labels remain unavailable.
