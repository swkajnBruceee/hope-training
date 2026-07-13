# Native Strike Training Status

Date: 2026-07-11

This file records the active native A3 strike-policy training line. It is a run log, not the architecture design document.

## Active Strategy

The current policy route is:

```text
A3 native standing / lower-body balance
        +
Isaac policy for waist + right arm strike execution
```

The policy is trained against native-calibrated manifest targets. These targets are the racket states produced by the native zero-residual replay library at the hit event. They are used to validate strike executability before reconnecting planner / ball targets.

Strict deterministic gate:

```text
position error < 0.10 m
velocity error < 2.0 m/s
normal error   < 20 deg
```

This racket-only gate is no longer sufficient. Every checkpoint or replay set
must also pass the posture gate. In the current native fixed-base task,
absolute torso upright is not a valid hard gate because even clean zero-action
references have `torso_upright ~= 0.86`. The current posture gate is
reference-relative:

```text
pelvis relative-reference error <= 15 deg
torso relative-reference error  <= 20 deg
non-waist native strike joint near-limit fraction <= 0.10
```

Do not expand a motion set if the current set does not pass both the racket gate
and the posture gate.

## Clean Motion Sets

| Set | Manifest | Count | Status |
| --- | --- | ---: | --- |
| K8 clean v1 | `sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_k8_clean_v1/manifest.json` | 4 FH + 4 BH | Passed |
| K16 clean v1 | `sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_k16_clean_v1/manifest.json` | 8 FH + 8 BH | Passed |
| K24 clean v2 | `sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_k24_clean_v2/manifest.json` | 12 FH + 12 BH | Racket gate passed, posture gate failed |
| K32 clean v1 | `sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_k32_clean_v1/manifest.json` | 16 FH + 16 BH | Failed expansion gate |
| K2 posture-balanced v1 | `sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_curated/k2_posture_balanced_v1/manifest.json` | 1 FH + 1 BH | Superseded: target/frame calibration stale |
| K4 posture-balanced v1 | `sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_curated/k4_posture_balanced_v1/manifest.json` | 2 FH + 2 BH | Superseded: target/frame calibration stale |
| K4 posture-balanced v2 | `sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_curated/k4_posture_balanced_v2/manifest.json` | 2 FH + 2 BH | Current zero-residual replay baseline passed |
| K8 posture-balanced v2 | `sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_curated/k8_posture_balanced_v2/manifest.json` | 4 FH + 4 BH | Zero-residual replay passed; residual PPO smoke passed |
| K16 posture-balanced v2 | `sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_curated/k16_posture_balanced_v2/manifest.json` | 8 FH + 8 BH | Zero-residual replay passed; residual PPO smoke passed |
| K24 posture-balanced v2 | `sample_motions/p2_fixed_competition_global_funnel_tracking_union55_v2_curated/k24_posture_balanced_v2/manifest.json` | 12 FH + 12 BH | Zero-residual replay passed; residual PPO smoke passed |

## Current Best Checkpoints

| Stage | Run | Checkpoint | Deterministic Eval |
| --- | --- | --- | --- |
| K1 reset-fix smoke | `2026-07-11_16-22-34_resetfix_k1_smoke_300` | `model_299.pt` | root reset fixed; racket position learned, normal/posture failed |
| K4 reset-fix baseline | `2026-07-11_16-28-21_resetfix_k4_baseline_500` | `model_499.pt` | 1/4 racket composite pass, 0/4 posture pass |
| K1 reward-E diagnostic | `2026-07-11_16-47-13_reward_e_k1_300` | `model_299.pt` | 1/1 racket composite pass, 0/1 posture pass |

No current native-strike checkpoint is usable for deployment or expansion.
The K1 reward-E checkpoint is useful as a reward diagnostic only: it fixes the
racket-normal failure on one motion, but still fails posture and joint-limit
quality gates.
Checkpoints trained before the root-reset fix are invalid for warm-starting.
The reset-fix K1/K4 runs are clean diagnostics only.

The current usable executor is not a PPO residual checkpoint. It is
zero-residual manifest replay on the v2 posture-compatible manifest:

```text
K4 posture-balanced v2 zero-action replay:
  forehand T_018_gao01_1p62_3p62
  forehand T001_002_gao01_5p67_7p67
  backhand T03_034_gao01_6p12_8p12
  backhand T002_022_gao01_20p68_22p68

  exact position error = 0.0000 m for all 4
  exact velocity error = 0.0000 m/s for all 4
  exact normal error   = 0.00-0.02 deg
  posture_pass_rate    = 1.000
```

The default task YAML is therefore set to reference-lock mode:

```text
actions.native_residual_scale = 0.0
```

Residual PPO correction must be treated as a separate experiment and enabled
explicitly. Do not use residual-training checkpoints as deployable executors
unless they beat zero-residual replay on both racket and posture gates.

Important evaluator correction on 2026-07-11:

```text
eval_manifest_zero_action.py previously printed cached command metrics for
exact-hit racket error. That made stale native-calibrated manifests look like
0-error replays.

The script now captures actual racket pos/vel/normal and target pos/vel/normal
at the exact hit frame and computes direct error, matching
eval_manifest_policy.py semantics.
```

This exposed that `k4_posture_balanced_v1` had stale target/frame calibration:

```text
v1 direct zero-action replay:
  position error ~= 0.63-0.73 m
  normal error   ~= 64-106 deg
  posture_pass_rate = 1.000
```

The manifest writer now also records:

```text
hit_event.motion_hit_frame
```

when writing native-calibrated manifests, so target state and exact-hit timing
remain aligned.

Old K24 is not a usable policy. The latest repeated racket-only eval of
`model_1800.pt` had maximum racket errors approximately:

```text
max position: 0.092 m
max velocity: 1.876 m/s
max normal:   18.92 deg
```

This looked usable under the old racket-only gate, but that conclusion was
wrong. The checkpoint was trained under the wrong reset distribution and must
not be used as a K32/K40 bootstrap checkpoint.

However, the visual replay on 2026-07-11 showed severe whole-body posture
failure: the robot leaned/back-bent heavily even though the racket hit-event
metrics passed. A posture-instrumented eval then reported:

```text
composite_pass_rate = 1.000  (24/24 racket gate)
posture_pass_rate   = 0.000  (0/24 posture gate)
```

Representative exact-hit posture values included torso upright as low as about
`0.52`, torso relative-reference errors commonly around `35-90 deg`, and native
strike joint near-limit fractions around `0.20-0.40`. The K24 checkpoint is
therefore archived as a diagnostic baseline only. Do not use it as a training
baseline for K32/K40 or deployment.

Root cause found on 2026-07-11:

```text
MotionCommand._resample_command() reset the articulation root from
self.body_pos_w[:, 0] / self.body_quat_w[:, 0].

In the native-strike task, cfg.body_names had been reduced to:
  torso_Link
  right_shoulder_roll_Link
  right_elbow_Link
  right_wrist_yaw_Link

So index 0 was torso_Link, not pelvis_link.
The task was writing the torso pose into the root state at reset.
```

The fix is to reset root state from the full motion arrays:

```text
motion._body_pos_w[..., 0]
motion._body_quat_w[..., 0]
motion._body_lin_vel_w[..., 0]
motion._body_ang_vel_w[..., 0]
```

After the fix, reset/no-step FK consistency is exact:

```text
max_pos_err_m = 0.00000
max_rot_err_deg = 0.00
```

All native-strike checkpoints trained before this fix are invalid for further
training, because they learned under the wrong reset distribution. Re-train from
K1/K2/K4 after this reset fix.

Clean reset-fix baseline results:

```text
K1 smoke:
  run: 2026-07-11_16-22-34_resetfix_k1_smoke_300
  checkpoint: model_299.pt
  deterministic result:
    position exact error = 0.0198 m
    velocity exact error = 1.9637 m/s
    normal exact error   = 25.45 deg
    composite_pass_rate  = 0.000
    posture_pass_rate    = 0.000

K4 baseline:
  run: 2026-07-11_16-28-21_resetfix_k4_baseline_500
  checkpoint: model_499.pt
  deterministic result:
    composite_pass_rate = 0.250
    posture_pass_rate   = 0.000
```

K4 exact-hit rows:

| Rank | Stroke | Episode | Position | Velocity | Normal | Racket Pass | Posture Pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | forehand | `T03_068_gao01_6p12_8p12` | 0.0061 | 1.5178 | 40.50 | 0 | 0 |
| 2 | backhand | `T002_022_gao01_1p61_3p61` | 0.0126 | 1.3830 | 19.90 | 1 | 0 |
| 3 | forehand | `T03_052_gao01_2p59_4p59` | 0.0386 | 0.9290 | 38.56 | 0 | 0 |
| 4 | backhand | `T03_034_gao01_1p26_3p26` | 0.1262 | 1.0460 | 26.27 | 0 | 0 |

This is now clean evidence, independent of the old reset bug, that the current
reward strongly favors racket position and under-weights racket normal and
whole-motion/posture quality. In the K4 run, the weighted normal reward stayed
around `0.004-0.005`, while the position rewards dominated learning. The next
training variant should adjust reward and gates before expanding beyond K4.

Reward diagnostics after this baseline:

```text
D variant:
  run: 2026-07-11_16-41-41_reward_d_k1_300
  change:
    added raw normal reward metrics
    increased normal contribution
    added a soft coupled hit reward
  conclusion:
    normal reward kernel is not dead, but the variant was still too weak.
    Early training improved normal, late training regressed toward position.
  deterministic model_299.pt:
    position exact error = 0.0149 m
    velocity exact error = 1.5703 m/s
    normal exact error   = 27.75 deg
    composite_pass_rate  = 0.000
    posture_pass_rate    = 0.000

E variant:
  run: 2026-07-11_16-47-13_reward_e_k1_300
  change:
    position weight reduced
    normal weight and coupled normal coefficient increased
    coupled reward made position success depend more strongly on normal
  conclusion:
    normal is learnable and reward balance is much better on K1.
    The next failure is posture / joint-limit quality, not racket normal.
  deterministic model_299.pt:
    position exact error = 0.0257 m
    velocity exact error = 1.2234 m/s
    normal exact error   = 6.28 deg
    composite_pass_rate  = 1.000
    posture_pass_rate    = 0.000
    torso relative-reference error = 29.22 deg
    native strike joint near-limit fraction = 0.3000
```

Current reward-E config:

```text
racket_position_weight = 4.0
racket_velocity_weight = 1.5
racket_normal_weight   = 2.0
racket_hit_coupled_weight = 4.0

racket_hit_coupled:
  pos_std = 0.08
  vel_std = 2.0
  normal_std = 0.45
  base = 0.15
  vel_coeff = 0.25
  normal_coeff = 0.60
```

Do not treat K1 reward-E as usable. Run the same variant on K4 next. If K4 keeps
the racket composite but posture remains zero, add process/posture/joint-limit
objectives before expanding to K8/K24.

Later native/posture-compatible residual experiments changed the diagnosis:

```text
J variant:
  run: 2026-07-11_18-03-22_reward_j_k2_posture_balanced_300
  data: k2_posture_balanced_v1 (superseded; stale target/frame calibration)
  change:
    selected two motions whose zero-action reference replay passes posture
    kept bounded residual PPO enabled
  deterministic model_299.pt:
    forehand position error = 0.7107 m
    forehand normal error   = 108.29 deg
    backhand position error = 0.8433 m
    backhand normal error   = 91.68 deg
    composite_pass_rate     = 0.000
    posture_pass_rate       = 1.000
  conclusion:
    invalid as a residual-PPO conclusion because the manifest target/frame
    calibration was stale. Keep only as a diagnostic artifact.

K variant:
  run: 2026-07-11_18-09-51_reward_k_k2_reference_lock_300
  stopped early around iteration 141
  data: k2_posture_balanced_v1 (superseded; stale target/frame calibration)
  change:
    lower exploration std
    entropy disabled
    stronger residual penalty
  observed:
    action residual remained around 0.46 mean action magnitude
    exact strike position error stayed around 0.72 m
  conclusion:
    invalid as a residual-PPO conclusion for the same stale-manifest reason.
    It still motivated the safer default: exact-manifest validation should use
    residual_scale=0 unless a residual-correction experiment is explicitly named.
```

Current corrective action:

```text
Reference replay / executor validation:
  native_residual_scale = 0.0

Residual correction research:
  enable residual scale only in a named experiment
  compare against zero-residual replay, not against old PPO checkpoints
```

Default K4 v2 policy eval sanity:

```text
checkpoint: 2026-07-11_18-03-22_reward_j_k2_posture_balanced_300/model_299.pt
task default manifest: k4_posture_balanced_v2
actions.native_residual_scale = 0.0

action scale abs max/mean = 0.000000 / 0.000000
composite_pass_rate = 1.000
posture_pass_rate   = 1.000
```

`action_abs_mean` in this mode still reports the policy's raw output, but the
processed action ignores it because the action scale is zero.

Residual PPO smoke test on the clean K4 v2 baseline:

```text
run: 2026-07-11_18-24-59_residual_probe_k4_v2_200
checkpoint: model_199.pt

train overrides:
  task.actions.native_residual_scale = 0.15
  task.actions.raw_clip = 0.25
  algo.policy.init_noise_std = 0.05
  algo.algorithm.entropy_coef = 0.0
  task.rewards.action_residual_weight = -0.2
```

Training signal stayed stable through 200 iterations:

```text
exact strike position error   ~= 0.007 m
exact strike velocity error   ~= 0.09 m/s
exact strike normal error     ~= 0.6 deg
exact composite success       = 1.000
```

Deterministic per-motion eval with the same residual scale enabled:

```text
action scale abs max/mean = 0.042000 / 0.022155
composite_pass_rate       = 1.000
posture_pass_rate         = 1.000

T002_022_gao01_20p68_22p68  pos=0.0048 vel=0.0089 normal=0.37
T_018_gao01_1p62_3p62       pos=0.0072 vel=0.0251 normal=0.93
T03_034_gao01_6p12_8p12     pos=0.0093 vel=0.0049 normal=0.49
T001_002_gao01_5p67_7p67    pos=0.0124 vel=0.0159 normal=1.35
```

Current conclusion:

```text
The policy architecture does not need immediate redesign.
With a self-consistent manifest (v2) and a small explicit residual contract,
PPO can preserve posture and exact-hit quality on K4.
```

Expansion result on K8 v2:

```text
source pool:
  sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_v2/manifest.json

K8 curated ids:
  forehand:
    T_018_gao01_1p62_3p62
    T001_002_gao01_5p67_7p67
    T002_015_gao01_17p03_19p03
    T002_021_gao01_1p73_3p73
  backhand:
    T03_034_gao01_6p12_8p12
    T002_022_gao01_20p68_22p68
    T03_034_gao01_2p48_4p48
    T002_027_gao01_7p12_9p12
```

Zero-residual replay on K8 v2:

```text
composite_pass_rate = 1.000
posture_pass_rate   = 1.000
```

Residual PPO smoke on K8 v2:

```text
run: 2026-07-11_18-32-31_residual_probe_k8_v2_200
checkpoint: model_199.pt

train overrides:
  task.actions.native_residual_scale = 0.15
  task.actions.raw_clip = 0.25
  algo.policy.init_noise_std = 0.05
  algo.algorithm.entropy_coef = 0.0
  task.rewards.action_residual_weight = -0.2
```

Deterministic per-motion eval with residual enabled:

```text
action scale abs max/mean = 0.042000 / 0.022155
composite_pass_rate       = 1.000
posture_pass_rate         = 1.000
```

Per-motion exact-hit errors:

```text
T002_015_gao01_17p03_19p03  pos=0.0061 vel=0.0162 normal=1.21
T002_022_gao01_20p68_22p68  pos=0.0068 vel=0.0155 normal=0.15
T03_034_gao01_2p48_4p48     pos=0.0070 vel=0.0104 normal=0.58
T002_027_gao01_7p12_9p12    pos=0.0072 vel=0.0102 normal=0.40
T002_021_gao01_1p73_3p73    pos=0.0076 vel=0.0213 normal=1.18
T_018_gao01_1p62_3p62       pos=0.0085 vel=0.0495 normal=1.07
T03_034_gao01_6p12_8p12     pos=0.0103 vel=0.0112 normal=0.56
T001_002_gao01_5p67_7p67    pos=0.0110 vel=0.0400 normal=0.45
```

Updated conclusion:

```text
The current policy route is viable through K8.
The next bottleneck is no longer "does PPO fundamentally break the executor?"
but "how far can the curated motion set expand before posture margin degrades?"
```

Expansion result on K16 v2:

```text
zero-residual replay:
  composite_pass_rate = 1.000
  posture_pass_rate   = 1.000
```

Residual PPO smoke on K16 v2:

```text
run: 2026-07-11_18-38-35_residual_probe_k16_v2_200
checkpoint: model_199.pt

train overrides:
  task.actions.native_residual_scale = 0.15
  task.actions.raw_clip = 0.25
  algo.policy.init_noise_std = 0.05
  algo.algorithm.entropy_coef = 0.0
  task.rewards.action_residual_weight = -0.2
```

Deterministic per-motion eval with residual enabled:

```text
action scale abs max/mean = 0.042000 / 0.022155
composite_pass_rate       = 1.000
posture_pass_rate         = 1.000
```

Representative K16 exact-hit errors:

```text
best:
  T001_002_gao01_5p67_7p67   pos=0.0022 vel=0.0244 normal=0.17
  T002_015_gao01_17p03_19p03 pos=0.0032 vel=0.0170 normal=0.87
  T_018_gao01_1p62_3p62      pos=0.0035 vel=0.0078 normal=0.50

harder-but-pass:
  T03_059_gao01_1p87_3p87    pos=0.0087 vel=0.0370 normal=0.84
  T03_083_gao01_6p90_8p90    pos=0.0116 vel=0.0093 normal=0.48
```

Current conclusion:

```text
The current strategy is now validated through K16.
No policy redesign is justified yet.
The next meaningful experiment is K24/K32 v2 expansion with the same residual contract.
```

Expansion result on K24 v2:

```text
source pool:
  sample_motions/p2_fixed_competition_global_funnel_tracking_union55_v2/manifest.json
  posture-pass candidates found: 12 forehand + 13 backhand

selected train manifest:
  sample_motions/p2_fixed_competition_global_funnel_tracking_union55_v2_curated/k24_posture_balanced_v2/manifest.json
```

Zero-residual replay on K24 v2:

```text
composite_pass_rate = 1.000
posture_pass_rate   = 1.000
overall:
  pos_mean    = 0.0000
  vel_mean    = 0.0000
  normal_mean = 0.00
```

Residual PPO smoke on K24 v2:

```text
run: 2026-07-11_18-58-37_residual_probe_k24_v2_200
checkpoint: model_199.pt
```

Deterministic nominal eval with residual enabled:

```text
composite_pass_rate = 1.000
posture_pass_rate   = 1.000
overall:
  pos_mean    = 0.0064
  vel_mean    = 0.0157
  normal_mean = 0.55
forehand:
  pos_mean    = 0.0050
  vel_mean    = 0.0193
  normal_mean = 0.76
backhand:
  pos_mean    = 0.0077
  vel_mean    = 0.0122
  normal_mean = 0.34
```

First mild perturbation comparison on K24:

```text
perturbation:
  pose_range:
    roll  = [-0.03, 0.03]
    pitch = [-0.03, 0.03]
    yaw   = [-0.05, 0.05]
  velocity_range:
    x/y   = [-0.05, 0.05]
    z     = [-0.02, 0.02]
    roll/pitch/yaw = [-0.1, 0.1]
  joint_position_range = [-0.02, 0.02]
```

Zero residual under perturbation:

```text
posture_pass_rate = 1.000
overall:
  pos_mean    = 0.0163
  vel_mean    = 0.0842
  normal_mean = 1.95
```

Learned residual under the same perturbation:

```text
composite_pass_rate = 1.000
posture_pass_rate   = 0.958
overall:
  pos_mean    = 0.0128
  vel_mean    = 0.0650
  normal_mean = 1.49
forehand:
  pos_mean    = 0.0112
  vel_mean    = 0.0745
  normal_mean = 1.64
backhand:
  pos_mean    = 0.0144
  vel_mean    = 0.0555
  normal_mean = 1.34
```

Observed tradeoff:

```text
Residual policy improves perturbed strike accuracy versus zero residual,
but loses one posture sample (24 -> 23 posture passes).

Failed posture sample:
  T03_059_gao01_1p87_3p87
  torso_ref_err_deg = 20.67
```

Updated conclusion:

```text
The route is validated through K24 on nominal conditions.
Residual PPO already shows measurable value under mild perturbation,
but posture margin has started to become the limiting factor.
```

K24 medium perturbation paired comparison

```text
seed = 7
perturbation:
  pose_range:
    roll  = [-0.06, 0.06]
    pitch = [-0.06, 0.06]
    yaw   = [-0.10, 0.10]
  velocity_range:
    x/y   = [-0.10, 0.10]
    z     = [-0.04, 0.04]
    roll/pitch/yaw = [-0.2, 0.2]
  joint_position_range = [-0.04, 0.04]
```

Zero residual under medium perturbation:

```text
hit_composite_pass_rate = 1.000
posture_pass_rate       = 0.958
whole_cycle_pass_rate   = 0.958
overall:
  pos_mean    = 0.0297
  vel_mean    = 0.1387
  normal_mean = 4.13
forehand:
  pos_mean    = 0.0272
  vel_mean    = 0.1651
  normal_mean = 4.94
backhand:
  pos_mean    = 0.0323
  vel_mean    = 0.1123
  normal_mean = 3.32
```

Learned residual under the same medium perturbation:

```text
hit_composite_pass_rate = 1.000
posture_pass_rate       = 1.000
whole_cycle_pass_rate   = 1.000
overall:
  pos_mean    = 0.0299
  vel_mean    = 0.1265
  normal_mean = 3.75
forehand:
  pos_mean    = 0.0204
  vel_mean    = 0.1545
  normal_mean = 3.45
backhand:
  pos_mean    = 0.0395
  vel_mean    = 0.0986
  normal_mean = 4.05
```

Interpretation:

```text
At medium perturbation the learned residual no longer improves every metric
uniformly, but it restores whole-cycle success to 24/24 while zero residual
stays at 23/24.

The paired rescued sample is:
  T04_021_gao01_10p13_12p13

zero residual:
  posture_pass      = 0
  torso_ref_err_deg = 23.02
  pos/vel/normal    = 0.0189 / 0.0746 / 3.86

learned residual:
  posture_pass      = 1
  torso_ref_err_deg = 18.64
  pos/vel/normal    = 0.0041 / 0.2085 / 3.25
```

Current reading:

```text
Residual policy provides clear robustness value by recovering at least one
medium-perturbation failure mode that zero residual does not recover.
The tradeoff has shifted from "does residual help at all?" to "how much margin
does it consume across stroke types and perturbation levels?"
```

Correction after adding fixed perturbation banks

```text
The earlier mild/medium perturbation conclusions above were computed from
same-seed but not fixed-bank perturbations. After adding explicit perturbation
banks, the paired comparison became stricter and supersedes those results.
```

K24 fixed-bank paired perturbation results

```text
bank support:
  eval_manifest_zero_action.py  +write_perturb_bank=...
  eval_manifest_zero_action.py  +perturb_bank=...
  eval_manifest_policy.py       +perturb_bank=...
```

Mild bank (`/tmp/k24_mild_bank_seed7.json`):

```text
zero residual:
  hit_composite_pass_rate = 1.000
  posture_pass_rate       = 0.958
  whole_cycle_pass_rate   = 0.958
  overall:
    pos_mean    = 0.0156
    vel_mean    = 0.0839
    normal_mean = 2.14

learned residual:
  hit_composite_pass_rate = 1.000
  posture_pass_rate       = 0.958
  whole_cycle_pass_rate   = 0.958
  overall:
    pos_mean    = 0.0165
    vel_mean    = 0.0861
    normal_mean = 2.23

rescue_count = 0
harm_count   = 0
both_fail    = [T04_021_gao01_10p13_12p13]
```

Medium bank (`/tmp/k24_medium_bank_seed7.json`):

```text
zero residual:
  hit_composite_pass_rate = 1.000
  posture_pass_rate       = 0.958
  whole_cycle_pass_rate   = 0.958
  overall:
    pos_mean    = 0.0297
    vel_mean    = 0.1387
    normal_mean = 4.13

learned residual:
  hit_composite_pass_rate = 1.000
  posture_pass_rate       = 0.958
  whole_cycle_pass_rate   = 0.958
  overall:
    pos_mean    = 0.0307
    vel_mean    = 0.1419
    normal_mean = 4.21

rescue_count = 0
harm_count   = 0
both_fail    = [T04_021_gao01_10p13_12p13]
```

Strong bank (`/tmp/k24_strong_bank_seed7.json`):

```text
zero residual:
  hit_composite_pass_rate = 0.875
  posture_pass_rate       = 0.833
  whole_cycle_pass_rate   = 0.750
  forehand whole_cycle    = 0.583
  backhand whole_cycle    = 0.917

learned residual:
  hit_composite_pass_rate = 0.875
  posture_pass_rate       = 0.833
  whole_cycle_pass_rate   = 0.750
  forehand whole_cycle    = 0.583
  backhand whole_cycle    = 0.917

rescue_count = 0
harm_count   = 0
both_fail    = [
    T002_015_gao01_17p03_19p03,
    T04_021_gao01_10p13_12p13,
    T002_021_gao01_1p73_3p73,
    T001_002_gao01_5p67_7p67
]
```

Revised conclusion:

```text
With fixed paired perturbations, the current learned residual policy does not
yet show net rescue over zero residual on the tested seed. It neither rescues
additional samples nor introduces extra failures at mild/medium/strong; the
main effect is small numeric changes in strike error.

The K24 route is still valid on nominal conditions, but the residual advantage
under perturbation is now unproven under strict pairing. The next step is a
multi-seed perturbation-bank sweep, not policy redesign and not forced K32.
```

K24 perturbation sweep (fixed banks, seeds 0/1/2)

Script:

```text
scripts/run_k24_perturbation_sweep.py
output:
  docs/eval_reports/k24_perturbation_sweep_seed012.json
```

Aggregate whole-cycle results:

```text
mild:
  zero_whole_mean    = 1.000
  learned_whole_mean = 1.000
  rescue_total       = 0
  harm_total         = 0

medium:
  zero_whole_mean    = 0.944
  learned_whole_mean = 0.944
  rescue_total       = 0
  harm_total         = 0

strong:
  zero_whole_mean    = 0.875
  learned_whole_mean = 0.875
  rescue_total       = 0
  harm_total         = 0
```

Per-stroke whole-cycle means:

```text
mild:
  zero     FH/BH = 1.000 / 1.000
  learned  FH/BH = 1.000 / 1.000

medium:
  zero     FH/BH = 0.917 / 0.972
  learned  FH/BH = 0.917 / 0.972

strong:
  zero     FH/BH = 0.833 / 0.917
  learned  FH/BH = 0.833 / 0.917
```

Interpretation:

```text
Across the first 3 fixed-bank seeds, learned residual does not yet show net
rescue over zero residual at mild/medium/strong.

The dominant degradation axis is forehand posture/whole-cycle robustness under
stronger perturbations; backhand remains consistently more robust.
```

## Archived Review Samples

These samples are not deleted. They are excluded from clean expansion sets and kept for later review.

| Episode | Reason |
| --- | --- |
| `T_010_gao01_8p47_10p47` | Failed K8 normal gate even after normal-focused refinement. |
| `T002_022_gao01_27p50_29p50` | Failed K24-v1 normal gate at best checkpoint by about 0.91 deg. |

The dominant failure mode in K24 experiments was not position reachability. It was backhand racket-normal error near or above the 20 deg gate. Blindly increasing normal reward made earlier K8 behavior worse, so the next fix should be targeted: inspect normal semantics / target consistency and per-stroke behavior before broad reward changes.

## Superseded Runs

The following runs were produced before the reset-root bug was found. They are
kept as diagnostic artifacts only and should not be warm-started.

K32 base training command:

```bash
source setup_train_env.sh >/dev/null && hope_isaac_py scripts/train.py \
  task=HOPEA3NativeStrikeManifest algo=ppo headless=true \
  logger=tensorboard num_envs=1024 max_iterations=600 \
  motion_manifest=sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_k32_clean_v1/manifest.json \
  manifest_subset_size=32 manifest_frame_z_offset=0.76 \
  resume=true \
  load_run=2026-07-11_12-32-10_native_calibrated_k24_clean_v2_from_v1_1800_300 \
  checkpoint=model_1800.pt \
  run_name=native_calibrated_k32_clean_v1_from_k24_600
```

Result:

```text
run: 2026-07-11_12-41-37_native_calibrated_k32_clean_v1_from_k24_600
best checkpoint checked: model_1800.pt
best deterministic pass count in saved report: 24/32
failure mode: newly added motions plus several backhand normal-gate failures
```

Later checkpoints improved position error but degraded racket-normal success. Therefore the last checkpoint is not usable as a K32 baseline.

Normal-balanced follow-up command:

```bash
source setup_train_env.sh >/dev/null && hope_isaac_py scripts/train.py \
  task=HOPEA3NativeStrikeManifest algo=ppo headless=true \
  logger=tensorboard num_envs=1024 max_iterations=300 \
  motion_manifest=sample_motions/p2_fixed_competition_global_funnel_balanced20_native_zero_residual_k32_clean_v1/manifest.json \
  manifest_subset_size=32 manifest_frame_z_offset=0.76 \
  task.rewards.racket_normal_weight=0.5 \
  task.rewards.racket_normal_std=0.7 \
  resume=true \
  load_run=2026-07-11_12-32-10_native_calibrated_k24_clean_v2_from_v1_1800_300 \
  checkpoint=model_1800.pt \
  run_name=native_calibrated_k32_clean_v1_normal_w05_from_k24_300
```

Result:

```text
run: 2026-07-11_12-53-16_native_calibrated_k32_clean_v1_normal_w05_from_k24_300
best checkpoint checked: model_1900.pt
best deterministic pass count in saved report: 25/32
effect: normal reward was applied correctly and slightly improved the best count, but K32 still did not pass
```

Saved deterministic eval reports:

```text
docs/eval_reports/k32_clean_v1_base_model_1800_eval.csv
docs/eval_reports/k32_clean_v1_base_model_1900_eval.csv
docs/eval_reports/k32_clean_v1_normal_w05_model_1900_eval.csv
```

Best K32 normal-balanced failures at `model_1900.pt`:

| Stroke | Episode | Position | Velocity | Normal |
| --- | --- | ---: | ---: | ---: |
| forehand | `T03_045_gao01_3p18_5p18` | 0.0050 | 1.5645 | 20.28 |
| backhand | `T_014_gao01_16p93_18p93` | 0.0173 | 0.8191 | 20.44 |
| backhand | `T03_031_gao01_5p42_7p42` | 0.0175 | 0.7800 | 27.03 |
| forehand | `T002_021_gao01_4p94_6p94` | 0.0221 | 1.0534 | 22.36 |
| backhand | `T002_023_gao01_1p15_3p15` | 0.0294 | 0.3546 | 22.69 |
| backhand | `T03_083_gao01_6p90_8p90` | 0.0313 | 0.7667 | 22.31 |
| forehand | `T_018_gao01_1p62_3p62` | 0.0950 | 0.9054 | 31.27 |

Expansion rule:

```text
No current native-strike checkpoint is usable.
Do not expand to K32/K40.
Before another expansion attempt, retrain from small K under the fixed root
reset and require both racket and posture gates.
```

## Next Diagnostic Step

The root-reset bug is fixed, and K1/K4 clean baselines have been run. The
immediate failure is no longer raw racket reachability: the policy can put the
racket near the manifest hit target, but does so with weak normal alignment,
unacceptable posture gate results, and frequent near-limit joint use. The next
useful work is:

```text
1. Keep the reset-root fix.

2. Add a reward variant before K4/K8 expansion:
   - strengthen racket normal reward
   - strengthen torso/right-arm process tracking
   - keep position reward from dominating the return

3. Extend deterministic eval to report whole-episode posture/process metrics:
   - pre-hit tracking pass
   - exact-hit racket pass
   - post-hit stability pass
   - whole-cycle posture pass

4. Re-run K1 then K4 with the adjusted reward.

5. Expand to K8 only after both racket composite and posture gates pass on K4.
```

## 2026-07-13 Waist-Yaw Retarget Invalidated Old K24/K32

The visual replay exposed a separate upstream data issue: many fixed-base
motions keep the feet/lower body facing forward while the waist and torso are
rotated close to backward. This is not a CSV-to-NPZ joint-order bug. The NPZ
joint vector is intentionally stored in Isaac articulation order after
name-based remapping. The large waist angle is already present in the generated
CSV and is then correctly transferred to NPZ.

Evidence from the fixed-base replay-ready pool:

```text
audited optimized CSVs: 55
max_abs_waist_yaw > 0.60 rad: 51
max_abs_waist_yaw > 1.00 rad: 46
max_abs_waist_yaw > 1.50 rad: 30
```

Worst examples include forehand clips with waist yaw around `-2.6 rad`
(`-150 deg`) and backhand clips around `+2.0 rad` to `+2.6 rad`. These motions
can pass racket-target gates while being visually and physically invalid for the
native-strike route.

Code changes now applied:

```text
cli_generate_a3_fixed_base_ik_init.py
  - adds waist_yaw_abs_limit_rad, default 1.0 rad
  - increases waist-yaw regularization
  - keeps fixed-base IK from using waist yaw as a global heading compensator

cli_generate_generic_retarget_init.py
  - adds the same waist-yaw protection for generic retarget initialization

cli_optimize_a3_fixed_base_trajectory.py
  - adds waist_yaw_abs_limit_rad and waist_yaw_delta_limit_rad
  - adds max_abs_waist_yaw_rad to quality metrics
  - adds max_abs_waist_yaw_reject_rad to replay-ready gating
```

All `retarget_DATA260708_p2_a3_fixed*.yaml` configs now set:

```yaml
ik:
  waist_yaw_abs_limit_rad: 1.0

optimization:
  waist_yaw_abs_limit_rad: 1.0
  waist_yaw_delta_limit_rad: 0.60

quality_thresholds:
  max_abs_waist_yaw_reject_rad: 1.0
```

Single-sample regression on `T_018_gao01_1p62_3p62`:

| Version | Waist yaw range | Status | Hit pos | Normal | Vel dir | Speed err |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| old optimized | `-2.214..-1.991 rad` | pass | 0.0013 m | 0.84 deg | 1.38 deg | 0.259 m/s |
| fixed optimized | `-0.889..0.001 rad` | pass | 0.0033 m | 1.41 deg | 1.50 deg | 0.027 m/s |

Small-batch regression on six old motions:

```text
IK: 5 pass, 1 reject
Optimization: 2 pass, 3 reject
Replay-ready after waist-yaw gate: 2/6
```

Representative outcomes:

| Episode | Old max abs waist yaw | New max abs waist yaw | New status | Reason |
| --- | ---: | ---: | --- | --- |
| `T_018_gao01_1p62_3p62` | 2.214 | 0.889 | pass | fixed-base pass |
| `T04_018_gao01_10p28_12p28` | 0.728 | 0.342 | pass | fixed-base pass |
| `T03_059_gao01_1p87_3p87` | 2.618 | n/a | reject | IK reject |
| `T04_014_gao01_5p96_7p96` | 2.618 | 0.105 | reject | dynamic fail |
| `T002_015_gao01_17p03_19p03` | 2.423 | 0.999 | reject | hit pose + dynamic fail |
| `T002_027_gao01_8p51_10p51` | 1.472 | 1.027 | reject | waist-yaw gate |

Updated conclusion:

```text
Old K24/K32 experiments are invalid as training-quality evidence.
They were run on a motion library where many clips had torso/waist orientation
inconsistent with the lower-body heading.

Do not resume old checkpoints as baselines.
Rebuild a clean replay-ready pool with the waist-yaw gate, then restart from
zero-action replay validation before any PPO expansion.
```

Follow-up rebuild status:

```text
combined waist1p0 target pool: 265
IK with waist_yaw_abs_limit_rad=1.0: 204 pass, 61 reject
IK pass split: 83 forehand, 121 backhand
optimized top40/stroke: 80 processed
strict2600 replay-ready: 19 total = 13 forehand + 6 backhand
review3500 replay-ready by audit: 30 total = 15 forehand + 15 backhand
review4000 replay-ready by audit: 36 total = 16 forehand + 20 backhand
```

Published strict library:

```text
hope_training/whole_body_tracking/sample_motions/
  p2_fixed_competition_global_funnel_waist1p0_strict2600/
```

Manifest:

```text
hope_training/whole_body_tracking/sample_motions/
  p2_fixed_competition_global_funnel_waist1p0_strict2600/manifest.json
```

Strict NPZ audit:

```text
motion count: 19
stroke counts: forehand=13, backhand=6
Isaac articulation waist_yaw_joint index: 2
max abs waist_yaw in NPZ: 0.984 rad
```

The strict set is usable for visual replay and smoke validation, but it is not
balanced enough for the previous `20 forehand + 20 backhand` training target.
The main blocker is now dynamic quality, especially jerk. On the optimized
top40/stroke audit, relaxing only the jerk gate gives:

```text
jerk <= 2600 rad/s^3: 19 ready = 13 FH + 6 BH
jerk <= 3500 rad/s^3: 30 ready = 15 FH + 15 BH
jerk <= 4000 rad/s^3: 36 ready = 16 FH + 20 BH
```

Do not silently treat the relaxed tiers as Golden data. They should be visual
review / coverage candidates unless the jerk threshold is deliberately revised.

Operational note:

```text
csv_to_npz initially failed because the root partition was full and Isaac wrote
temporary USD files under /tmp/IsaacLab.

Cleaned /tmp/IsaacLab and reran with:
  TMPDIR=/home/bruce/tmp_isaac
```

Use `TMPDIR=/home/bruce/tmp_isaac` for future Isaac batch conversions while the
root partition remains near full.
