# Current Native Strike Workflow

Date: 2026-07-14

This is the current operating workflow for A3 native-strike data and training.
Older plans, K-series manifests, whole-body PPO runs, and diagnostic retarget
experiments are archived unless this document or `sample_motions/README_DATASETS.md`
explicitly promotes them.

## Active Strategy

```text
A3 native standing / lower-body balance
        +
fixed-base waist + right-arm strike motion
        +
manifest command target:
  racket position
  racket velocity
  racket normal
  time-to-hit
  stroke type
```

Do not train from old 31-DOF whole-body PPO manifests or old K24/K32 checkpoints.
Those experiments are historical diagnostics only.

## A3 Body-Drive Validation Layer

The local A3 MuJoCo/AimRT body-drive environment is now configured under
`tools/` and documented in:

```text
docs/A3_MUJOCO_BODY_DRIVE_VALIDATION.md
```

It validates the real-shaped body-drive command/state contract, 500 Hz timing,
PD tracking, actuator limits and IMU logging. It does not contain the official
A3 MOTION/PD_STAND balance controller, so a full-body NPZ replay falling in
this environment is not a native-balance verdict. Do not promote a motion or
checkpoint from body-drive replay alone.

## Current Training Data Status

### Forehand

Current accepted source:

```text
sample_motions/p2_fixed_forehand_combined_gate_v1/accepted_forehand_manifest.json
```

Status:

```text
4 forehands
hit task gate:              4 / 4
robot posture / arm margin: 4 / 4
wrist / forearm gate:       4 / 4
visual replay review:       4 / 4 accepted
```

Source retarget output:

```text
data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_forehand_combined_gate_v1/
```

Config:

```text
data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed_forehand_combined_gate_v1.yaml
```

### Backhand

Current torso-controlled source:

```text
data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_backhand_torso_control_v1/tracking_motion_manifest_backhand.json
```

Status:

```text
8 candidate backhands re-IKed with waist pitch/roll comfort constraints
4 / 8 pass IK
3 / 4 optimized primary targets are replay-ready
12 supplemental archived candidates evaluated
6 / 12 supplemental candidates pass IK
5 / 6 supplemental optimized targets are replay-ready
4 selected for current torso-control K8
4 additional supplemental backhands kept as held-out/candidates
visual replay review: pending
```

Source retarget output:

```text
data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_backhand_torso_control_v1/
```

Config:

```text
data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed_backhand_torso_control_v1.yaml
```

Notes:

```text
Old backhand expand4_v4 is archived/reference only for current training.
It passed earlier geometry/numeric gates, but its optimized references used
about 18-23 deg mean waist pitch and about 24 deg peak waist pitch, which
matches the visually observed forward torso lean.

New backhand_torso_control_v1 reduces replay-ready backhands to about
1.5-2.6 deg mean waist pitch and about 2.8-5.8 deg peak waist pitch.
```

### Current Mixed Candidate

Current mixed candidate:

```text
sample_motions/p2_fixed_balanced_k8_torso_control_v1/manifest.json
```

Status:

```text
4 forehand + 4 backhand
status: candidate, not final training set
forehand: inherited from accepted current forehands
backhand: torso_control_v1 references
held-out/candidate backhands: 4
next gate: visual replay review + clean physical zero-residual tracking
```

## Current Gate

The current gate is robot/task-centric, not human-imitation-centric:

```text
hit task gate
+ robot posture / arm-margin gate
+ wrist / forearm naturalness gate
+ visual replay review
```

Evaluator fields and thresholds are documented in:

```text
docs/NATIVE_STRIKE_GATE_DESIGN.md
```

Final promotion requires visual review because the previous wrist-folding
failure was first caught visually, after an older numeric gate had already
passed.

## Current Evaluation Logs

Active eval logs:

```text
eval_outputs/forehand_combined_gate_v1_accepted_native_zero_action_20260714.log
eval_outputs/backhand_current_pool_v2_native_gate_20260714.log
eval_outputs/balanced_k8_current_v1_native_gate_20260714.log
eval_outputs/native_torso_tracking_k8_zero_default_servo_20260714/
```

All older eval logs are archived under:

```text
eval_outputs/_archive_not_for_training/20260714_superseded_eval_logs/
```

## Retarget / Export Workflow

For a new candidate set:

```text
1. Build target/probe manifest
2. IK init
3. trajectory optimization
4. CSV -> NPZ conversion
5. tracking manifest generation
6. native zero-residual calibration
7. combined gate evaluation
8. visual replay review
9. promote only accepted subset into sample_motions
```

Do not skip native zero-residual calibration before evaluating manifest targets.

## Replay Commands

Direct reference replay, no policy:

```bash
cd /home/bruce/桌面/HOPETableTennis/hope_training/whole_body_tracking
source setup_train_env.sh

TMPDIR=/home/bruce/tmp_isaac hope_isaac_py scripts/replay_npz.py \
  --robot agibot_a3 \
  --motion_file <absolute_npz_path> \
  --steps 1200
```

Manifest evaluation:

```bash
TMPDIR=/home/bruce/tmp_isaac hope_isaac_py scripts/eval_manifest_zero_action.py \
  task=HOPEA3NativeStrikeManifest headless=true num_envs=<N> \
  motion_manifest=<manifest.json> \
  manifest_subset_size=<N>
```

## Archive Policy

Archive, do not train:

```text
old balanced20 / K8 / K16 / K24 / K32 manifests
old whole-body PPO runs
old posture-only robot gate outputs
old forehand comfort-zone scans
old wrist / torso probe experiments
failed or diagnostic smoke training runs
```

Archive locations:

```text
sample_motions/_archive_not_for_training/20260714_superseded_manifests/
data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/_archive_not_for_training/20260714_superseded_outputs/
data/analysis/mocap_cleaning/configs/_archive_not_for_training/20260714_superseded_configs/
eval_outputs/_archive_not_for_training/20260714_superseded_eval_logs/
docs/_archive_not_for_training/20260714_superseded_eval_reports/
```

## Next Step

The visually accepted balanced K8 manifest has completed a 300-iteration
residual-PPO smoke run:

```text
sample_motions/p2_fixed_balanced_k8_current_v1/manifest.json
/workspace/hopetmp/whole_body_tracking_logs/rsl_rl/agibot_a3_native_strike_manifest/
  2026-07-14_17-22-45_balanced_k8_current_v1_res015_smoke_300_20260714/model_299.pt
```

2026-07-14 update: this checkpoint is **not** a promotion baseline. Visual
policy replay showed torso forward lean and loose upper-body wobble. A follow-up
diagnostic showed the same issue with zero residual action, so the root issue is
below the PPO residual layer.

Current diagnosis:

```text
kinematic NPZ replay:
  writes root + joint states directly
  hides actuator tracking error

native-strike play / training:
  sends reference joint positions as PD targets
  actual waist/shoulder joints can lag or collapse under physics
```

Measured K8 zero-residual physical tracking issue:

```text
eval_outputs/native_torso_tracking_k8_zero_20260714/

waist_pitch tracking p95: about 0.42 rad
waist_roll tracking p95:  about 0.35 rad
right_shoulder_pitch p95: about 0.53 rad

some forehand references:
  ref torso pitch mean:    about 2-3 deg
  physical torso pitch:    about 11-17 deg mean, 23-24 deg max
```

Therefore the next step is not more PPO iterations. The next step is:

```text
1. Establish a zero-residual physical tracking gate.
2. Fix native-strike actuator / PD / damping / reference-smoothing issues.
3. Add torso dynamic metrics: pitch/roll, tilt, angular velocity, recovery.
4. Only then retrain K8 residual PPO.
```

Diagnostic tool:

```bash
TMPDIR=/home/bruce/tmp_isaac hope_isaac_py scripts/diagnose_native_torso_tracking.py \
  task=HOPEA3NativeStrikeManifest algo=ppo headless=true num_envs=8 max_steps=90 \
  motion_manifest=sample_motions/p2_fixed_balanced_k8_current_v1/manifest.json \
  manifest_subset_size=8 \
  checkpoint=/workspace/hopetmp/whole_body_tracking_logs/rsl_rl/agibot_a3_native_strike_manifest/2026-07-14_17-22-45_balanced_k8_current_v1_res015_smoke_300_20260714/model_299.pt \
  +out_dir=eval_outputs/native_torso_tracking_k8_zero_YYYYMMDD \
  +rollout_mode=zero
```

Do not promote a manifest or checkpoint from visual NPZ replay alone. A candidate
must pass the zero-residual physical tracking gate first.

2026-07-14 A3-MC servo approximation update:

The A3 SDK shows that the real route should rely on MC standing / waist / arm
servo behavior, not a weak randomized bare-PD Isaac executor. The native-strike
environment now disables PD-gain randomization and uses a stronger waist/right
arm servo profile only for this task.

Updated zero-residual physical tracking:

```text
eval_outputs/native_torso_tracking_k8_zero_default_servo_20260714/

waist_pitch tracking p95: 0.421 rad -> 0.216 rad
waist_roll tracking p95:  0.347 rad -> 0.169 rad
right_shoulder_pitch p95: 0.526 rad -> 0.390 rad
joint tracking p95:       0.528 rad -> 0.367 rad
```

This is a meaningful improvement and should be the new Isaac execution baseline
for native-strike experiments. It is still not a final promotion gate, because
some forehand follow-through phases retain visible torso pitch. Next fixes
should target reference smoothing / follow-through recovery, not more PPO
iterations.
```

Numeric deterministic policy eval passed the current robot/task gate:

```text
hit_composite_pass_rate     = 8 / 8
robot_posture_pass_rate     = 8 / 8
wrist_naturalness_pass_rate = 8 / 8
whole_cycle_pass_rate       = 8 / 8
```

Visual replay showed that the hand/racket motion is much improved, but the
policy still allows excessive upper-body forward lean and loose torso motion.
This checkpoint is therefore not promoted as an expansion baseline.

Next train a tighter K8 residual variant. Keep the same visually accepted K8
manifest, but reduce waist roll/pitch residual authority and strengthen torso /
balance-aware smoothness objectives. Do not expand beyond K8 until the trained
policy is visually checked for wrist folding, excessive torso lean/side tilt,
loose upper-body motion, and shoulder/elbow margin issues.

Training commands must pass `motion_manifest=...` explicitly. The task config
uses `motion_manifest: null` to prevent accidental use of stale data.
