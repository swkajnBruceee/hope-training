# Manual Accepted Retarget Baseline v1

Date: 2026-07-13

This directory contains the current manually accepted A3 fixed-base retarget baseline. It is a reference motion set, not a trained policy result.

## Accepted Motions

| Stroke | Episode | Source Variant | Status |
| --- | --- | --- | --- |
| forehand | `T04_005_gao01_7p05_9p05` | `retarget_p2_fixed_a3_forehand_torso_probe_normalC_moderate` | accepted after visual review |
| backhand | `T03_030_gao01_0p99_2p99` | `retarget_p2_fixed_a3_wrist_naturalness_probe` | accepted after visual review |

## Forehand Notes

The accepted forehand is the moderate normal-weight variant, not the earlier torso-only variant and not the aggressive normal A/B variants.

Key metrics:

- hit position error: `0.00658 m`
- racket orientation error: `13.05 deg`
- racket velocity direction error: `7.05 deg`
- racket speed error: `0.44 m/s`
- max active joint velocity: `3.83 rad/s`
- max active joint jerk: `2367.48 rad/s^3`
- waist yaw max: `18.14 deg`
- waist roll max: about `7.30 deg`
- waist pitch max: about `7.66 deg`
- right wrist bend pitch/yaw p95: `28.92 deg`

Manual visual result:

- side bending is acceptable for the current forehand baseline
- wrist folding is no longer the dominant artifact
- hit pose margin is thinner than the unconstrained variant, but the trajectory remains dynamically valid

## Backhand Notes

The accepted backhand keeps the wrist naturalness probe output unchanged.

Key metrics:

- hit position error: `0.00095 m`
- racket orientation error: `0.42 deg`
- racket velocity direction error: `1.06 deg`
- racket speed error: `0.17 m/s`
- max active joint velocity: `1.54 rad/s`
- max active joint jerk: `1566.07 rad/s^3`
- waist yaw max: `37.88 deg`
- right wrist bend pitch/yaw p95: `18.41 deg`

Manual visual result:

- backhand is acceptable and should not be modified in the next expansion step unless a new visual artifact appears

## Published Training Library

Published symlink library:

`hope_training/whole_body_tracking/sample_motions/p2_fixed_manual_accepted_v1`

Manifest:

`data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/accepted_manual_retarget_v1/tracking_motion_manifest.json`

## Do Not Mix

The following are probe or rejected-intermediate variants and should not be used as accepted training data:

- `retarget_p2_fixed_a3_forehand_torso_probe`
- `retarget_p2_fixed_a3_forehand_torso_probe_normalA`
- `retarget_p2_fixed_a3_forehand_torso_probe_normalB_wrist`
- the forehand entry inside `retarget_p2_fixed_a3_wrist_naturalness_probe`

## Next Expansion Rule

For the next forehand expansion, use the accepted forehand-C style constraints:

- keep waist roll/pitch constrained to avoid excessive side lean
- allow waist yaw as the main torso rotation degree of freedom
- keep wrist pitch/yaw in a comfort range rather than fully locking the wrist
- reject trajectories that pass hit geometry only by creating excessive side bend, folded wrist, or jerk spikes

For backhand expansion, start from the wrist naturalness probe constraints and only tighten torso constraints if visual review shows the same side-lean artifact.
