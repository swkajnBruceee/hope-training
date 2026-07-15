# Native Strike Gate Design

This document records the current gate semantics for A3 native strike motion
selection. The gate is intended to select robot-usable strike motions, not
motions that exactly mimic the human source.

Current workflow entry point:

```text
docs/CURRENT_NATIVE_STRIKE_WORKFLOW.md
```

## Gate Layers

### Task Gate

The hit gate checks racket task-space accuracy at the strike event:

- position error
- velocity vector error
- racket normal error

This remains a hard requirement for training candidates.

### Reference Posture Gate

The legacy posture gate is still printed as `posture_pass` for continuity:

```text
pelvis_ref_err_deg <= 15
torso_ref_err_deg  <= 20
arm_near_limit_frac <= 0.10
```

This gate measures similarity to the reference trajectory. It should no longer
be treated as a hard robot-usability reject by itself, especially for
`torso_ref_err_deg`, because yaw, pitch, and roll are mixed into one scalar.

### Robot Posture Gate

The robot posture gate adds robot-centric metrics:

```text
pelvis_ref_err_deg <= 15
torso_tilt_abs_deg <= 32
torso_roll_abs_deg <= 25
torso_pitch_abs_deg <= 35
arm_near_limit_frac <= 0.10
min_arm_margin >= 0.05
```

The key distinction is that torso yaw is not treated as inherently bad.
Forehand strikes may use yaw differently from a human reference. Tilt, roll,
pitch, and arm joint margin are more relevant to stability and control margin.

### Wrist / Forearm Naturalness Gate

The robot posture gate alone is still insufficient. A forehand can keep the
torso and shoulder inside the robot gate while folding the wrist/hand relative
to the forearm in a visually and mechanically poor way.

Current evaluator metrics:

```text
right_wrist_roll_abs_deg
right_wrist_pitch_abs_deg
right_wrist_yaw_abs_deg
right_wrist_bend_pitch_yaw_deg
forearm_racket_angle_deg
```

Current hard screen:

```text
right_wrist_roll_abs_deg <= 65
right_wrist_pitch_abs_deg <= 35
right_wrist_yaw_abs_deg <= 35
right_wrist_bend_pitch_yaw_deg <= 45
forearm_racket_angle_deg <= 75
```

This is not a human-biomechanics imitation rule. It is a robot-quality rule to
prevent the optimizer from using the wrist as a cheap redundant solution for
paddle orientation. The roll limit is intentionally wider than pitch/yaw because
roll may be needed for racket orientation; pitch/yaw and forearm-racket angle
catch the visually unacceptable folding failure.

### Tier Labels

The evaluators now print `robot_posture_pass` and `gate_tier`:

- `A_robot_usable_candidate`: task gate, robot posture gate, and wrist/forearm
  naturalness gate all pass.
- `B_wrist_retarget_required`: task and robot posture pass, but wrist/forearm
  naturalness fails. Do not train; retarget the forehand.
- `B_robot_borderline`: task gate passes, arm margin is acceptable, but posture
  is close to or outside the robot posture gate.
- `C_requires_stance_or_retarget`: task gate passes but the fixed-base posture
  or arm margin is not acceptable.
- `D_task_fail`: task gate fails.

These labels are screening aids. `A_robot_usable_candidate` is not automatically
the final clean training set; candidates still need dynamic smoothness, waist
margin, and visual review when they are close to limits.

### Physical Tracking Gate

2026-07-14 correction: kinematic NPZ replay is necessary but not sufficient.
`scripts/replay_npz.py` writes root and joint states directly every frame, so it
can look clean even when the same motion cannot be physically tracked by the
Isaac PD actuators.

Before PPO training, each candidate manifest must pass a zero-residual physical
tracking check:

```text
action = 0
processed action = manifest reference joint position
robot executes through Isaac actuators / PD tracking
```

Required metrics:

```text
q_ref / q_target / q_actual for native strike joints
torso pitch / roll / tilt over the whole cycle
torso angular velocity over the whole cycle
waist pitch/roll/yaw tracking error
right shoulder/elbow/wrist tracking error
recovery posture and angular velocity
```

Current known failure on K8:

```text
zero-residual K8 physical tracking
  waist_pitch tracking p95: about 0.42 rad
  waist_roll tracking p95:  about 0.35 rad
  right_shoulder_pitch p95: about 0.53 rad

forehand example:
  reference torso pitch mean: about 2-3 deg
  physical torso pitch mean:  about 11-17 deg
  physical torso pitch max:   about 23-24 deg
```

This means the candidate can be visually good in kinematic replay but still not
ready for training, because the physical actuator layer collapses into a forward
lean. This gate must pass before a checkpoint can be promoted.

2026-07-14 servo baseline:

Native-strike Isaac execution now uses a deterministic A3-MC-like servo profile
instead of randomized weak PD:

```text
PD gain randomization: disabled for native-strike
waist pitch/roll/yaw: stronger deterministic servo gains
arms: 2x deterministic Isaac arm servo gains
```

This improved K8 zero-residual tracking but did not completely solve torso
follow-through pitch:

```text
waist_pitch tracking p95: 0.421 rad -> 0.216 rad
right_shoulder_pitch p95: 0.526 rad -> 0.390 rad
```

Therefore the physical tracking gate should now evaluate against this updated
servo baseline, and any remaining failures should be treated as reference
smoothing / recovery problems unless a later AimSim/A3 test contradicts that.

### Torso Reference Pitch Gate

2026-07-14 correction: backhand forward lean was not only a physical tracking
problem. The old `backhand_expand4_v4` optimized references already contained
large waist pitch:

```text
old backhand_expand4_v4 replay-ready samples
  waist_pitch mean: about 18-23 deg
  waist_pitch max:  about 22-24 deg
  waist_roll max:   about 20 deg
```

Those samples should not be promoted for current training even if their racket
hit geometry is good. The retarget/reference itself is using forward torso lean
as a cheap redundant solution.

The current replacement is `backhand_torso_control_v1`, which adds waist
pitch/roll neutral deadbands and comfort ranges during IK and trajectory
optimization:

```text
waist_roll comfort:  [-0.10, 0.10] rad
waist_pitch comfort: [-0.12, 0.12] rad
```

Current replay-ready result:

```text
new backhand_torso_control_v1 primary samples
  count: 3
  waist_pitch mean: about 1.5-2.6 deg
  waist_pitch max:  about 2.8-5.8 deg
  waist_roll max:   about 3.5-4.8 deg

new backhand_torso_control_supplement_v1 samples
  replay-ready count: 5
  best K8 supplement: T002_023_gao01_26p64_28p64
  best supplement waist_pitch mean/max: about 1.2/3.2 deg
  best supplement waist_roll max: about 3.8 deg
```

This gate is robot-centric, not human-imitation-centric. Torso yaw may remain
wide for backhand/forehand reach if the A3 waist yaw limit and recovery margin
are acceptable. The strict checks are pitch, roll, tracking smoothness, and
joint margin.

## Why This Changed

The previous `torso_ref_err_deg <= 20` hard gate rejected forehand variants that
hit exactly and had acceptable robot-centric posture after the strike target was
mapped toward a more comfortable region. The failure was mostly reference
difference, not necessarily robot instability.

The new output separates:

- `torso_ref_err_deg`: total reference orientation error.
- `torso_ref_yaw_delta_deg`: yaw difference from reference.
- `torso_tilt_abs_deg`: absolute torso tilt from world vertical.
- `torso_roll_abs_deg` / `torso_pitch_abs_deg`: rough tilt components.
- `min_arm_margin`: closest non-waist controlled joint margin.

This allows the project to reject dangerous shoulder/elbow/wrist usage while
not rejecting a robot-specific torso yaw solution just because it differs from a
human reference.

2026-07-14 correction:

`p2_fixed_balanced_robot_gate_k4_v1` passed an earlier robot-posture-only
screen, but visual replay showed forehand wrist/forearm folding. That manifest
and the PPO smoke checkpoint trained from it are diagnostic only and must not be
used as a baseline.

## Current Forehand Re-Evaluation

For `p2_fixed_forehand_comfort_y_pos_scan_v1/native_zero_residual_manifest.json`:

```text
legacy posture_pass_rate = 0/8
robot_posture_pass_rate  = 6/8
hit_composite_pass_rate  = 8/8
```

This means the old gate was too strict for reference similarity. The two
remaining rejects still have insufficient arm margin and should remain
`C_requires_stance_or_retarget`.

For `p2_fixed_forehand_combined_gate_v1/accepted_forehand_manifest.json`:

```text
hit_composite_pass_rate       = 4/4
robot_posture_pass_rate       = 4/4
wrist_naturalness_pass_rate   = 4/4
whole_cycle_pass_rate         = 4/4
```

This set was visually reviewed on 2026-07-14. All 4 accepted forehands were
reported as particularly good and are promoted to the current forehand training
candidate source.

## Open Items

- Add a separate waist/native-controller margin review. Current robot posture
  gate focuses on non-waist arm margin.
- Tune native-strike actuator/PD gains or reference smoothing so zero-residual
  physical tracking does not drive waist pitch/roll toward limits.
- Add torso angular velocity reward/gate. Current native reward tracks elbow and
  wrist angular velocity, but not torso angular velocity.
- Add COM/support-foot/contact metrics when available.
- Add post-hit recovery metrics before treating a candidate as final clean
  training data.
- Synchronize these fields into perturbation sweep reports.
