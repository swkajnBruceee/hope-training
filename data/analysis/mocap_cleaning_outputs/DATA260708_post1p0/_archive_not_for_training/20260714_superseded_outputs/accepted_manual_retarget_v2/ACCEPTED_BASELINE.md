# Manual Accepted Retarget Baseline v2

Date: 2026-07-14

This directory contains the current manually accepted A3 fixed-base retarget baseline. It is a reference motion set, not a trained policy result.

## Accepted Motions

| Stroke | Episode | Source Variant | Status |
| --- | --- | --- | --- |
| forehand | `T04_005_gao01_7p05_9p05` | `retarget_p2_fixed_a3_forehand_torso_probe_normalC_moderate` | accepted after visual review |
| forehand | `T001_003_gao01_2p92_4p92` | `retarget_p2_fixed_a3_forehand_expand4_v1` | accepted after visual review |
| backhand | `T03_030_gao01_0p99_2p99` | `retarget_p2_fixed_a3_wrist_naturalness_probe` | accepted after visual review |

## Current Count

- forehand: `2`
- backhand: `1`
- total: `3`

## Newly Promoted Motion

`T001_003_gao01_2p92_4p92` was promoted from pending visual review after playback approval.

Key metrics:

- hit position error: `0.00337 m`
- racket orientation error: `10.49 deg`
- racket velocity direction error: `3.35 deg`
- racket speed error: `0.48 m/s`
- max active joint velocity: `2.21 rad/s`
- max active joint jerk: `779.05 rad/s^3`
- waist yaw max: `30.57 deg`
- right wrist pitch p95: `7.40 deg`
- right wrist yaw p95: `30.70 deg`
- right wrist roll p95: `44.51 deg`

## Published Training Library

Published symlink library:

`hope_training/whole_body_tracking/sample_motions/p2_fixed_manual_accepted_v2`

Manifest:

`data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/accepted_manual_retarget_v2/tracking_motion_manifest.json`

## Version Note

`accepted_manual_retarget_v1` is preserved as the earlier `1 forehand + 1 backhand` baseline. Use v2 for the current accepted reference set.
