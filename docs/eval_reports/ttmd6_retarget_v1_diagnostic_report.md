# TTMD6 A3 Retarget v1 Diagnostic Report

Date: 2026-07-17

## Scope

This report records the first locked TTMD6 pilot pass using the project-wide
coordinate contract:

```text
lateral mapping: source_right_to_a3_minus_y
paddle orientation: velocity_plane_neg
```

The package is diagnostic only. It is not admitted to RL training.

## Pipeline

```text
30 locked position/orientation candidates
    -> A3 fixed-base IK initialization
    -> diagnostic optimization input
    -> A3 fixed-base trajectory optimization
    -> NPZ conversion
    -> complete Isaac reference replay
```

The formal IK gate rejected all 30 pilot candidates. Twenty candidates had
position and normal reachability but failed the tangent gate, so they were
passed to optimization through an explicitly marked diagnostic-only manifest.
The original formal status is retained in `ik_status_original`.

## Counts

```text
locked candidates:             30
formal IK passes:               0
diagnostic optimization input: 20
optimized replay-ready:        16
optimized rejects:               4
training eligible:               0
```

The 16 replay-ready records are distributed as:

```text
class1: 3
class2: 5
class3: 5
class4: 1
class5: 1
class6: 1
```

Using the current high-confidence class interpretation, this is 13 forehand
and 3 backhand candidates. It is not a balanced training pool and cannot be
used to build K12.

## Objective optimized metrics

Across the 16 replay-ready records:

```text
hit position error:       median 2.8 mm, max 7.9 mm
hit orientation error:    median 3.6 deg, max 12.5 deg
velocity direction error: median 3.7 deg, max 7.8 deg
speed error:              median 0.32 m/s, max 0.66 m/s
waist yaw magnitude:      median 0.60 rad, max 0.99 rad
wrist bend p95:           median 11.0 deg, max 13.9 deg
active-joint jerk:        median 1097 rad/s^3, max 1800 rad/s^3
```

These numbers show that several task-space trajectories are executable under
the current fixed-base optimizer. They do not prove that the constructed
paddle orientation is source ground truth, that the tangent target is correct,
or that the clips are suitable for the native A3 controller.

## Replay artifacts

```text
source manifest:
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/a3_position_candidates_locked_v1/manifest.json

optimized manifest:
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v1/a3_ik_locked_v1/optimized_manifest.json

videos:
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v1/a3_ik_locked_v1/replay_video_v1/videos/
```

All 16 videos were generated successfully at `960x720`, `60 fps`, with hold
frames before and after the motion. The videos are reference replays, not PPO
rollouts.

## Decision

Do not promote this package to the training manifest. The next TTMD6 action is
to expand the locked `minus_y + velocity_plane_neg` source pool, maintaining
the same contract, until there are enough independent forehand and backhand
records for train and held-out splits. Do not relax the formal tangent gate
solely to increase the count; first determine whether tangent is a genuine
task requirement or an over-constrained diagnostic term.

In parallel, the current K8 official-PD baseline remains the training
reference. K12 is blocked until its held-out waist-margin and paired
perturbation gates are addressed.

## Expansion Intake Probe (2026-07-17)

A balanced 24-clip intake batch was selected from the audited 11,900
structurally eligible local records, excluding the 30-clip pilot. The batch
contains four clips from each numeric class. Its source labels remain
high-confidence inferred labels and are not treated as authoritative metadata.

```text
data/analysis/mocap_cleaning_outputs/TTMD6_expansion_intake_v1/
```

The first formal A3 probe used one clip per class and the current wrist/waist
comfort configuration. With the source lacking a measured paddle tangent, the
formal task gate was position + normal + robot limits; tangent error was kept as
a diagnostic metric. This distinction is deliberate: the constructed
`velocity_plane_neg` branch is a locked orientation hypothesis, but a source
that contains paddle center position only cannot provide a ground-truth
around-normal tangent label.

```text
formal tangent-gate probe:  0 / 6 pass
formal task-space IK probe: 5 / 6 pass
optimization after task gate: 5 / 5 replay-ready
```

The single task-space rejection was `class2_sample52`, with a 52.4 degree
normal error. The five task-space passes had 0.4--2.0 mm hit-position error
and 0.4--3.5 degree normal error before trajectory optimization. The large
100--160 degree tangent errors are therefore not evidence that those five
motions are unusable; they show that the tangent comparison is not calibrated
to the source contract. The tangent value remains available for ranking,
visual review, and later source-orientation calibration.

This is a gate definition correction, not a quality relaxation: position,
normal, joint comfort, continuity, dynamics, native zero-residual execution,
and replay review remain mandatory before training admission.
