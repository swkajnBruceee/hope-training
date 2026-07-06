# Landing Decision Design

This document defines the first dynamic landing-point decision module for the
HOPE planner. The module replaces the fixed-center target with a two-stage
selection policy:

1. hard constraints reject unsafe or physically invalid candidates;
2. soft constraints rank all remaining candidates without rejecting them.

The implementation lives in `src/landing_decision/` with the public interface in
`include/landing_decision.h`.

## Inputs and Outputs

Inputs:

- `msgs/msg/PredictedStrike` from `/ball/predicted_strike` and
  `/ball/post_bounce_predicted_strike`
- table geometry from `common::TableParams`
- ball/racket physics from `common::BallPhysics` and `common::PlannerConfig`

Output:

- `msgs/msg/TargetDecision` on `/target_decision`

The solver remains responsible for final racket-command generation. The decision
module only chooses `target_land`, `delta_t_flight`, and policy limits.

## Hard Constraints

Hard constraints are used only for failures that should never be selected:

- invalid or non-finite strike prediction
- `time_to_strike < 0.12 s`
- landing point outside opponent table half
- `delta_t_flight` outside `[0.40, 0.70] s`
- solver plan invalid
- net not cleared
- trajectory bypasses net posts
- predicted table contact before the target landing time
- racket velocity above the planning cap

The HOPE rules document limits the robot gripping/racket-mount region composite
linear speed to `6 m/s` during human-robot exhibition safety inspection. The
planner uses `5.4 m/s` by default to leave 10% margin because the solver's
`racket_velocity` is a model target, not exactly the IMU measurement point.

No competition rule in the local HOPE rules or ITTF laws defines a ball outgoing
speed cap. Therefore `max_ball_out_speed` stays disabled by default (`-1.0`) and
ball speed is handled as a soft control-quality term.

## Soft Constraints

Soft constraints rank candidates that already passed hard checks:

- edge margin: avoid landing close to net, side lines, and end line
- net clearance comfort: no extra reward above the comfort margin
- racket speed: no penalty below `3.0 m/s`, increasing penalty from `3.0` to
  `5.4 m/s`
- ball speed: comfort band `[4.5, 8.5] m/s`; slower is not automatically better
- flight time: comfort band `[0.48, 0.60] s`
- competitiveness: target a configurable competitiveness level around the fixed
  center comfort point `(2.055, -0.7625, 0)`

The competitiveness term is inspired by Li et al. 2015, "Designation and Control
of Landing Points for Competitive Robotic Table Tennis". The first version uses
only the useful idea: map points farther from the opponent comfort point to
higher competitiveness. It does not copy paper-specific formulas.

## Candidate Set

The first version uses a fixed candidate grid:

- `center_mid = (1.95, -0.7625, 0)`
- `center_deep = (2.30, -0.7625, 0)`
- `side_neg_mid = (1.95, -1.10, 0)`
- `side_pos_mid = (1.95, -0.425, 0)`
- `side_neg_deep = (2.30, -1.10, 0)`
- `side_pos_deep = (2.30, -0.425, 0)`
- `short_center = (1.65, -0.7625, 0)`

Each point is paired with `delta_t_flight = 0.45, 0.50, 0.55, 0.60 s`, giving 28
candidates per strike.

## Failure Behavior

If no dynamic candidate passes hard constraints, the planner evaluates the
previous fixed-center target as `fallback_fixed_center`.

If fallback also fails, the planner publishes:

- `valid = false`
- `mode = "no_feasible_landing"`

The solver treats this mode as an explicit current-strike failure and does not
reuse a stale previous target.

## Future Extension

The Li et al. paper also proposes using historical target-vs-actual landing
error to compensate outgoing velocity. This project does not yet have a stable
`actual_landing_point` feedback interface, so the first version only reserves the
concept. Add a feedback message before enabling learned compensation.
