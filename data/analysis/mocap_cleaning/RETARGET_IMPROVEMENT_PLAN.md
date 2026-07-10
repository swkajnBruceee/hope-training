# A3 Fixed-Base Retarget Improvement Plan

This plan tracks the practical changes to move the retargeting pipeline from
racket-motion reproduction toward executable strike-state reproduction.

## Done

- Stage 3 now includes hit racket speed magnitude in the optimizer objective.
- Stage 3 quality reports now include actual hit racket speed, target hit
  racket speed, and speed magnitude error.
- Stage 3 replay gating now treats hit speed magnitude as part of hit geometry.
- Stage 4/5 motion and tracking manifests now preserve hit timing metadata and
  strike targets when target specs are available.
- Stage 2 IK initialization now evaluates normal and tangent errors and only
  marks `seed_ready` samples as `ik_status=pass`.
- Stage 1 target selection supports deterministic diversity sampling over hit
  position, incoming ball velocity, racket velocity, and racket normal.

## P0

1. Add a closed-loop tracking validation stage.
   - Replay replay-ready trajectories through the controller rather than only
     checking kinematic schema.
   - Record joint tracking error, racket hit position/orientation/speed error,
     actuator limit pressure, NaN/termination status, and collision flags when
     available.

## P1

1. Make IK initialization hit-anchored.
   - Solve the hit frame first with multiple seeds.
   - Warm-start backward before hit and forward after hit from the selected hit
     posture.

2. Increase hit-window control authority.
   - Add hit-near control points such as `hit_enter=-3` and `hit_exit=+3`, or
     introduce a Hermite-style `qdot_hit` variable.

3. Revisit spline boundary conditions.
   - Store boundary velocity intent in target specs.
   - Consider explicit derivative boundary conditions instead of unconditional
     natural cubic spline boundaries.

## P2

1. Expand manifest QA summaries.
   - Include hit-position ranges, incoming ball speed ranges, racket hit speed
     ranges, normal distributions, duration ranges, and dynamic metric
     distributions.

2. Add strike smoke tests.
   - Start with schema smoke, then full tracking rollout smoke, then ball-contact
     smoke once the physics validation harness exists.
