# Native Strike Sample Motion Registry

Last updated: 2026-07-14

This registry separates active training inputs from diagnostic and historical
artifacts. Do not infer usability from a directory name. Use only manifests
explicitly listed as active.

Current full workflow:

```text
../docs/CURRENT_NATIVE_STRIKE_WORKFLOW.md
```

## Active Training Inputs

Current balanced training candidate:

```text
p2_fixed_balanced_k8_current_v1/manifest.json
```

Status:

```text
4 forehands: visual accepted
4 backhands: current numeric gate passed and visual accepted
K8 zero-action gate: 8 / 8 whole-cycle pass
```

This is the current small training candidate for the native fixed-base
waist/right-arm residual-PPO smoke experiment.

The current combined gate is:

```text
hit task gate
+ robot posture / shoulder-elbow margin gate
+ wrist / forearm naturalness gate
+ visual replay review
```

The task config intentionally has `motion_manifest: null`. Pass the approved
manifest explicitly when training.

## Current Top-Level Sources

Only these non-archive directories remain at the top level:

| Source | Role | Status |
| --- | --- | --- |
| `p2_fixed_forehand_combined_gate_v1/` | Current forehand repair | `accepted_forehand_manifest.json` is visually accepted and is the current forehand training candidate source. |
| `p2_fixed_backhand_current_pool_v2/` | Current backhand pool | `accepted_backhand_manifest.json` contains 6 numeric-gate accepted backhands; 4 are selected for K8 and visually accepted. |
| `p2_fixed_balanced_k8_current_v1/` | Current balanced K8 candidate | 4 accepted forehands + 4 accepted backhands; K8 numeric gate passed 8/8 and visual review passed. |
| `p2_fixed_backhand_expand4_v2_pending_visual/` | Earlier backhand expansion | Superseded by `p2_fixed_backhand_current_pool_v2/`; keep as source trace only. |
| `p2_fixed_manual_accepted_v3/` | Manual accepted diagnostics | Historical diagnostic. Some paths were superseded by archive moves; do not train directly. |

## Current Diagnostic Sources

| Source | Role | Status |
| --- | --- | --- |
| `_archive_not_for_training/20260714_superseded_manifests/p2_fixed_forehand_comfort_y_pos_scan_v1/` | Forehand comfort-zone / base-y offset scan | Diagnostic only. Numeric robot posture can pass, but visual wrist/forearm naturalness was not sufficient. |
| `_archive_not_for_training/20260714_superseded_manifests/p2_fixed_forehand_native_margin_v1/` | Forehand wrist/native-margin repair | Diagnostic only. Wrist is better, but those forehand variants fail robot posture/shoulder margin after native calibration. |
| `_archive_not_for_training/20260714_superseded_manifests/p2_fixed_balanced_robot_gate_k4_v1/` | 2 FH from comfort scan + 2 BH accepted | Archived diagnostic. Passed older robot gate and PPO smoke, but visual replay showed unacceptable wrist/forearm folding on forehand. Do not use as K4/K8/K24 baseline. |
| `_archive_not_for_training/20260714_superseded_manifests/p2_fixed_balanced_native_margin_v1_k4/` | 2 FH native-margin + 2 BH accepted | Archived diagnostic. Hit gate passes, backhands pass robot gate, forehands fail shoulder/arm-margin gate. |
| `p2_fixed_manual_accepted_v3/` | Manual accepted reference set | Backhand entries are current usable diagnostics. Forehand entries require re-check against the combined gate before use. |

## Historical / Archived

The following families are history only unless a new registry entry explicitly
promotes a subset:

```text
_archive_not_for_training/20260714_superseded_manifests/p2_fixed_competition_global_funnel_balanced20*
_archive_not_for_training/20260714_superseded_manifests/p2_fixed_competition_global_funnel_tracking_union55*
_archive_not_for_training/20260714_superseded_manifests/p2_fixed_competition_global_funnel_waist1p0_strict2600*
_archive_not_for_training/20260714_superseded_manifests/p2_fixed_competition_global_funnel_balanced20_native_zero_residual*
```

Reasons:

- Some were created for the older whole-body / old posture gate route.
- Some use stale native-calibrated targets or pre-reset-fix checkpoints.
- Some passed racket-only or robot-posture-only gates but did not include the
  current wrist/forearm naturalness gate.

## Current Forehand Work Item

Build the next forehand set as:

```text
p2_fixed_forehand_combined_gate_v1
```

Current numeric result:

```text
accepted_forehand_manifest.json:
  4 / 4 hit task gate
  4 / 4 robot posture / arm-margin gate
  4 / 4 wrist / forearm naturalness gate
  4 / 4 whole-cycle numeric gate
```

Status: visual accepted on 2026-07-14; current forehand training candidate source.

Required properties:

- task hit gate passes at the exact hit frame;
- fixed-base robot posture is acceptable without excessive side tilt;
- right shoulder and right elbow retain soft-limit margin;
- wrist pitch/yaw remain in a comfort range;
- forearm-to-racket-center angle does not show visual folding;
- visual replay is accepted before training.

## Current Backhand Work Item

Current numeric-gate source:

```text
p2_fixed_backhand_current_pool_v2/accepted_backhand_manifest.json
```

Current numeric result:

```text
8 candidate backhands evaluated
6 / 8 whole-cycle numeric gate pass
4 selected for balanced K8 by arm-margin ranking
```

Rejected/currently excluded backhands remain useful diagnostics, but are not
part of the first K8 training set:

```text
T03_072_gao01_2p56_4p56
T04_001_gao01_7p78_9p78
```

Reason: hit and wrist gates pass after native calibration, but robot posture /
right-arm margin fails. Treat them as stance-offset or retarget candidates, not
as first-pass fixed-base training data.
