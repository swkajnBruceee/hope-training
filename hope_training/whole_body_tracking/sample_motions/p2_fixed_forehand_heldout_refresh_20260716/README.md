# Independent Forehand Held-Out Refresh (2026-07-16)

This directory is an independent evaluation pool. It is **not training data**.

## Provenance

- Source candidate index: archived global competition candidate index.
- Selection: 20 independent forehand candidates, excluding the two previously
  failing held-out episodes `T001_003...` and `T04_005...`.
- Cheap selection: 20 candidates.
- IK initialization: 9 pass, 11 reject.
- Fixed-base trajectory optimization: 4 replay-ready, 5 reject.
- NPZ conversion: 4 completed with the A3 articulation order.
- Native calibration: current Isaac `HOPEA3NativeStrikeManifest`, zero residual.

## Current Evaluation

Reports:

- Raw target evaluation:
  `eval_outputs/forehand_refresh_native_zero_20260716.log`
- Native-calibrated zero-residual evaluation:
  `eval_outputs/forehand_refresh_native_calibrated_20260716.log`
- Calibrated manifest:
  `native_zero_residual_manifest.json`

Native-calibrated result:

```text
hit composite:       4/4
wrist naturalness:   4/4
robot posture:       1/4
whole cycle:         1/4
```

The three failures are classified as `C_requires_stance_or_retarget`. They
have negative minimum arm margin and are near the right shoulder roll/yaw
limits. This is a fixed-base workspace/stance issue, not an NPZ or quaternion
conversion issue.

## Promotion Rule

Do not add these four motions to the active training manifest. Keep them as
held-out evidence for the stance-offset layer. A motion may be promoted only
after it passes the current robot-usable gate, or after a validated stance
offset is applied and the resulting base-relative command passes the same
native-calibrated evaluation.

