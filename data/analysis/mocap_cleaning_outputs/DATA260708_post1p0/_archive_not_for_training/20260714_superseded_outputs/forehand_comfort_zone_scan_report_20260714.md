# Forehand Comfort-Zone Scan 2026-07-14

## Purpose

Diagnose whether the failing forehand native posture gate is mainly caused by fixed-base reachability. Two forehand motions that hit exactly after native target calibration but failed posture were scanned by translating the racket target trajectory in world Y. These are diagnostic variants, not physical ball relabels.

## Inputs

- Source motions: `T002_015_gao01_15p25_17p25`, `T03_012_gao01_12p10_14p10`
- Baseline native-calibrated zero-action result: hit composite passes, posture fails.
- Failure mode: `torso_ref_err_deg > 20` and/or `right_shoulder_roll_joint` near soft limit.

## Negative Y Scan

Directory: `retarget_p2_fixed_a3_forehand_comfort_y_scan_v1`

- Tested world Y deltas: `0, -0.05, -0.10, -0.15, -0.20 m`
- IK: `10/10` pass
- Optimization: `5/10` replay-ready
- Native-calibrated zero-action: `hit_composite=5/5`, `posture=0/5`
- Trend: negative Y did not help; it generally worsened `torso_ref_err_deg`.

## Positive Y Scan

Directory: `retarget_p2_fixed_a3_forehand_comfort_y_pos_scan_v1`

- Tested world Y deltas: `+0.05, +0.10, +0.15, +0.20 m`
- IK: `8/8` pass
- Optimization: `8/8` replay-ready
- Native-calibrated zero-action: `hit_composite=8/8`, `posture=0/8`
- Trend: positive Y helped shoulder margin. Several variants removed non-waist near-limit violations, but torso remained just over the current strict threshold.

Best native-gate candidates from positive scan:

| Motion | dy | torso_ref_err_deg | arm_near_limit_frac | near-limit names | posture |
|---|---:|---:|---:|---|---:|
| `T002_015_gao01_15p25_17p25_dyp20cm` | +0.20 | 20.08 | 0.0000 | `waist_roll_joint|waist_pitch_joint` | 0 |
| `T03_012_gao01_12p10_14p10_dyp20cm` | +0.20 | 20.78 | 0.0000 | `waist_pitch_joint` | 0 |
| `T002_015_gao01_15p25_17p25_dyp10cm` | +0.10 | 20.90 | 0.0000 | `waist_pitch_joint` | 0 |

## Interpretation

The fixed-base reachability hypothesis is partially supported: moving the target in the correct direction improves shoulder/arm margin substantially. However, target translation alone did not fully close the native posture gate. These variants should not enter the training set yet.

Important: this scan shifted target trajectories, which is kinematically equivalent to a static base placement test for reachability, but it is not a physically valid ball dataset by itself. The original strike point and a required base translation should be preserved separately.

## Current Decision

- Do not train on these diagnostic forehand variants yet.
- Do not relax the posture gate just to accept them.
- Use this result to define a `requires_stance_offset` / `fixed_base_comfort` split.
- For immediate fixed-base training, prefer forehands already inside the comfort workspace.
- For side targets, store `required_base_translation` and handle them in a stance-placement stage before strike execution.
