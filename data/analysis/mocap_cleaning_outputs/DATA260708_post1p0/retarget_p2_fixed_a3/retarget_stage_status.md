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
- IK csv_to_npz command count is 19 and should be treated as diagnostic replay only, not training data.

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
