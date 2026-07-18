# Forehand Held-Out Stance-Enhanced Pool (2026-07-16)

This is an independent evaluation pool. It is **not part of the active K8
training manifest**.

## Processing

The four motions came from the independent forehand refresh pool:

```text
T002_001_gao01_7p52_9p52
T03_004_gao01_3p08_5p08
T03_014_gao01_26p29_28p29
T_018_gao01_3p20_5p20
```

Each motion was retargeted with:

- constant `stance_offset_xy` optimized jointly with the upper-body trajectory;
- world hit target preserved;
- A3 Isaac soft-limit factor `0.9` used during optimization;
- 10DOF waist plus right-arm active set;
- NPZ conversion and native zero-residual calibration.

## Native-Calibrated Result

```text
hit composite:       4/4
robot posture:       4/4
wrist naturalness:   4/4
whole cycle:         4/4
minimum arm margin:  +0.1000 for all four
```

The selected base offsets are stored in the source quality reports under
`data/analysis/mocap_cleaning_outputs/.../retarget_p2_fixed_a3_forehand_stance_bank_20260716/`.
The final archive-manifest regression is recorded in
`eval_outputs/forehand_stance_heldout_archive_regression_20260716.log`.

## Promotion Rule

This pool proves that stance-aware retargeting can recover the fixed-base arm
limit failures. It does not yet prove that the current policy can consume a
stance command or execute a footstep. Keep it separate until a stance-aware
command term and deployment contract are implemented and evaluated.
