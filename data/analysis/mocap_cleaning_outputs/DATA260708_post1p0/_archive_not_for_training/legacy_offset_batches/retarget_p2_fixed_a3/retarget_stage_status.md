# DATA260708 P2 Fixed-Base A3 Retarget Stage Status

## Scope

- Dataset: `datasets_ready/DATA260708_competition_core.npz`
- Side: P2 / `gao01`
- Robot: `agibot_a3`
- Base mode: fixed
- Coordinate frame: `competition_table_m`
- Current base: `[3.15, -0.35, 0.3084]`, quaternion xyzw `[0, 0, 1, 0]`
- Active joints: waist 3 joints + right arm 7 joints

## Outputs

- Config: `data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed.yaml`
- Target manifest: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/retarget_ready/retarget_target_manifest.json`
- IK manifest: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/ik_init_manifest.json`
- Optimized manifest: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/optimized_manifest.json`
- Optimized csv_to_npz commands: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/csv_to_npz_optimized_commands.sh`
- Optimized csv_to_npz jobs: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/csv_to_npz_optimized_jobs.json`
- Optimized motion NPZ manifest: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/optimized_motion_npz_manifest.json`
- Optimized motion NPZ summary: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/optimized_motion_npz_summary.md`
- Tracking motion manifest: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/tracking_motion_manifest.json`
- Tracking motion library: `hope_training/whole_body_tracking/sample_motions/p2_fixed_competition/manifest.json`
- IK diagnostic csv_to_npz commands: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/csv_to_npz_ik_commands.sh`

## Current Result

- Target preparation selected 20 samples: 10 forehand and 10 backhand.
- IK first batch processed 20 samples: 19 passed hit-position threshold and 1 failed in fixed-base IK.
- Trajectory optimization v1 processed the 19 IK-pass samples and produced 16 `replay_ready` trajectories.
- Current fixed-base classification over the 20 selected samples:
- `fixed_base_pass`: 16
- `fixed_base_dynamic_fail`: 1
- `fixed_base_hit_pose_fail`: 2
- `fixed_base_reach_fail`: 1
- Formal optimized csv_to_npz command count is 16 and includes only `replay_ready=true` samples.
- All 16 optimized replay-ready CSV trajectories have now been converted to local motion NPZ files.
- NPZ validation passed for all 16 files: `fps=50`, `joint_pos=(80, 31)`, `body_pos_w=(80, 32, 3)`.
- Local replay smoke checks passed for representative motions covering forehand and backhand.
- Tracking/training smoke checks passed for one forehand motion and one backhand motion with `task=TrackingFlat`, `num_envs=8`, `max_iterations=1`.
- IK csv_to_npz command count is 19 and should be treated as diagnostic replay only, not training data.

## Integration Checks

- Replay smoke samples:
- `T001_003_gao01_2p92_4p92`
- `T002_027_gao01_8p51_10p51`
- `T03_065_gao01_14p31_16p31`
- Training smoke motions:
- `forehand`: `sample_motions/p2_fixed_competition/forehand/T001_002_gao01_1p92_3p92.npz`
- `backhand`: `sample_motions/p2_fixed_competition/backhand/T002_022_gao01_20p68_22p68.npz`
- Training smoke log directories:
- `hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_flat/2026-07-09_00-51-02`
- `hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_flat/2026-07-09_00-54-04`

## Expansion Start

- Expansion config: `data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed_expand.yaml`
- Expanded export directory: `data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_expand`
- Current expanded target batch prepared: `200` samples
- Total candidates under the current P2 fixed-base selection rule: `475`

## Interpretation

The coordinate frame and fixed-base P2 reachability are usable for hit-frame IK after moving the fixed base laterally and opening waist joints. The v1 optimizer is now using a hit-first objective: strong hit-window constraints, weaker pre/post tracking, whole-trajectory dynamics penalties, jerk penalties, and replay prechecks. This is enough to make most of the first 20-sample batch replay-ready under fixed base.

The csv-to-npz conversion path was also upgraded from "one Isaac launch per clip" to "one Isaac launch for multiple jobs" through `csv_to_npz.py --batch_jobs_json ...`. This batch mode was used to finish the remaining conversions locally after validating the 16 replay-ready trajectories.

Do not promote the 3 reject trajectories into RL/training. The next retargeting iteration should focus on the remaining categories separately: hit-pose failures likely need stance/reach treatment or tighter swing-direction modeling, while the single dynamic-fail sample needs stronger smoothing without losing hit geometry.

## Reproduction

```bash
python data/analysis/mocap_cleaning/cli_prepare_competition_retarget_targets.py \
  --config data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed.yaml --limit 20

python data/analysis/mocap_cleaning/cli_generate_a3_fixed_base_ik_init.py \
  --config data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed.yaml --limit 20

python data/analysis/mocap_cleaning/cli_optimize_a3_fixed_base_trajectory.py \
  --config data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed.yaml --limit 20

python data/analysis/mocap_cleaning/cli_build_a3_csv_to_npz_commands.py \
  --config data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed.yaml --stage optimized

bash data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/csv_to_npz_optimized_commands.sh

python data/analysis/mocap_cleaning/cli_build_a3_csv_to_npz_commands.py \
  --config data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed.yaml --stage ik
```
