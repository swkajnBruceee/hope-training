# DATA260703 Retarget Jobs

- robot: `agibot_a3`
- source manifest: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_ready/retarget_manifest.json`
- jobs: `705`

## Label Counts

| label | count |
|---|---:|
| backhand | 287 |
| forehand | 418 |

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

- `jobs_manifest_json`: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3/jobs_manifest.json`
- `jobs_summary_md`: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3/jobs_summary.md`
- `jobs_root`: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3`

## Refinement

- Priority is racket-first: pose, face normal, velocity direction, then human-like arm/torso motion.
- The expected pipeline is generic retarget init -> A3 constrained refinement -> validation -> csv_to_npz.

## Notes

- Each job declares the input BVH, source clean sample, target retarget CSV path, and target motion NPZ path.
- Job status is initialized as `pending`; this script defines the refinement contract but does not solve A3 IK yet.
- This is the handoff point between dataset curation and robot-specific retarget implementation.
