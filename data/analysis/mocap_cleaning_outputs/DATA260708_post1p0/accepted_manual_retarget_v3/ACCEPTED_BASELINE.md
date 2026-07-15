# Manual Accepted Retarget Baseline v3

Date: 2026-07-14

This directory contains the current manually accepted A3 fixed-base retarget baseline. It is a reference motion set, not a trained policy result.

## Accepted Motions

| Stroke | Episode | Source Variant | Status |
| --- | --- | --- | --- |
| forehand | `T04_005_gao01_7p05_9p05` | `retarget_p2_fixed_a3_forehand_torso_probe_normalC_moderate` | accepted after visual review |
| forehand | `T001_003_gao01_2p92_4p92` | `retarget_p2_fixed_a3_forehand_expand4_v1` | accepted after visual review |
| backhand | `T03_030_gao01_0p99_2p99` | `retarget_p2_fixed_a3_wrist_naturalness_probe` | accepted after visual review |
| backhand | `T002_023_gao01_26p64_28p64` | `retarget_p2_fixed_a3_backhand_expand4_v2` | accepted after visual review |

## Current Count

- forehand: `2`
- backhand: `2`
- total: `4`

## Newly Promoted Motion

`T002_023_gao01_26p64_28p64` was promoted from pending visual review after playback approval.

Key metrics:

- hit position error: `0.00083 m`
- racket orientation error: `2.12 deg`
- racket velocity direction error: `0.37 deg`
- racket speed error: `0.18 m/s`
- max active joint velocity: `1.30 rad/s`
- max active joint jerk: `2411.06 rad/s^3`
- waist yaw max: `31.55 deg`
- right wrist pitch p95: `18.57 deg`
- right wrist yaw p95: `18.07 deg`
- right wrist roll p95: `40.24 deg`

## Published Training Library

Published symlink library:

`hope_training/whole_body_tracking/sample_motions/p2_fixed_manual_accepted_v3`

Manifest:

`data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/accepted_manual_retarget_v3/tracking_motion_manifest.json`

## Version Note

Use v3 as the current balanced smoke-training reference set. It contains `2 forehand + 2 backhand` manually accepted reference motions.
