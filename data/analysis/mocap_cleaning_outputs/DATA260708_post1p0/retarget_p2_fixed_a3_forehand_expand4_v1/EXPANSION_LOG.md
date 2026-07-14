# Forehand Expansion 4 v1

Date: 2026-07-13

Purpose: test whether the manually accepted forehand-C retarget constraints generalize to additional forehand candidates.

## Input Candidates

Source manifest:

`data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_global_funnel_supplement20/retarget_ready/retarget_target_manifest.json`

Selected forehand candidates:

| Episode | IK Status | Optimization Status | Replay Ready | Notes |
| --- | --- | --- | --- | --- |
| `T001_003_gao01_2p92_4p92` | pass | pass | yes | pending visual review |
| `T03_068_gao01_4p60_6p60` | pass | reject | no | hit geometry passes, dynamic jerk fails |
| `T_021_gao01_3p14_5p14` | reject | not run | no | unreachable |
| `T03_050_gao01_10p00_12p00` | reject | not run | no | position reachable only; pose not ready |

## Accepted-Constraint Result

The expansion produced one new replay-ready forehand candidate:

`T001_003_gao01_2p92_4p92`

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

## Published Pending Library

Published symlink library:

`hope_training/whole_body_tracking/sample_motions/p2_fixed_forehand_expand4_v1_pending_visual`

This candidate is not part of `accepted_manual_retarget_v1` yet. It requires visual review before being promoted.

## Takeaway

The accepted forehand-C constraints do generalize to at least one additional forehand, but the pass rate is still low:

- IK pass: `2 / 4`
- optimization replay-ready: `1 / 2`
- end-to-end replay-ready: `1 / 4`

Do not expand aggressively until several more forehands pass both numeric checks and visual review.
