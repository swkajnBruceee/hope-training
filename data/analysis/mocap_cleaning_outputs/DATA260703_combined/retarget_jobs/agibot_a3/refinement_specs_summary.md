# A3 Refinement Specs

- spec version: `1.1.0`
- contract version: `a3_refinement_contract_v1`
- source jobs: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3/jobs_manifest.json`
- specs generated: `705`

## Label Counts

| label | count |
|---|---:|
| backhand | 287 |
| forehand | 418 |

## Outputs

- `spec_manifest_json`: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3/refinement_spec_manifest.json`
- `specs_root`: `data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3/refinement_specs`

## Notes

- Each spec contains coordinate contract, full racket transform metadata, joint masks, phase windows, and warning/reject thresholds.
- These specs are solver inputs; they do not yet produce A3 joint trajectories.
