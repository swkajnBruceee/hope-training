# DATA260708 P2 Fixed-Base A3 Expansion Status

## Scope

- Dataset: `datasets_ready/DATA260708_competition_core.npz`
- Side: P2 / `gao01`
- Robot: `agibot_a3`
- Base mode: fixed
- Coordinate frame: `competition_table_m`
- Expansion config: `data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed_expand.yaml`

## Current Batch

- Expanded target preparation wrote `200` candidate samples to the expansion output directory.
- Under the current P2 fixed-base selection rule, the total candidate pool size is `475`.
- The first expanded processing batch used `50` targets.

## First Expanded Batch Result

- IK processed `50` targets:
- `pass`: 48
- `reject`: 2

- Trajectory optimization processed `48` IK-pass targets:
- `fixed_base_pass`: 39
- `fixed_base_dynamic_fail`: 6
- `fixed_base_hit_pose_fail`: 3
- `replay_ready`: 39

- Batch csv-to-npz conversion completed for all `39` replay-ready trajectories.
- NPZ validation passed for all `39` files:
- `fps = 50`
- `joint_pos = (80, 31)`
- `body_pos_w = (80, 32, 3)`

## Integration

- Tracking manifest:
- `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_expand/tracking_motion_manifest.json`

- ASCII motion library:
- `hope_training/whole_body_tracking/sample_motions/p2_fixed_competition_expand/manifest.json`

- Training smoke check on an expanded-only motion passed:
- `sample_motions/p2_fixed_competition_expand/backhand/T002_023_gao01_17p55_19p55.npz`
- Command mode: `task=TrackingFlat`, `num_envs=8`, `max_iterations=1`

## Recommendation

The expanded batch is healthy enough to continue, but the optimization latency is materially higher than the 20-sample pilot. For the next expansion stage, keep the processing batch size at `25` instead of `50` so that failure statistics and tuning feedback remain tight.
