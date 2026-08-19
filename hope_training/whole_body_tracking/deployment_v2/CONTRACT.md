# Deployment-Aligned High-Level SAC V2 — HOPE Open-Source Contract v1

## Status

`PROTOTYPE_NOT_TRAINED`

## Canonical authority

The frozen `model_21800` ONNX, final open-source deployment parser/policy, and
matching HOPE ROS trajectory/solver implementation are the only source of
truth. The prototype requires model SHA256:

`6bf1a2418f8538e23577a0153f2fe6a1e78dee91f41650a232259432a84a4dc8`

## Low level

`frozen model_21800`; policy contract remains 110D input to 31D output.
High-Level SAC V2 never produces 31D joints or the 110D low-level observation.

## Input

7D:

```text
[
  hit_y_world,
  hit_z_world,
  incoming_vx_world,
  incoming_vy_world,
  incoming_vz_world,
  control_tts,
  swing_sign,
]
```

Fields originate from structured `msgs/RacketCommand`: position y/z, incoming
ball velocity xyz, and time-to-strike aged from `header.stamp`. `swing_sign` is
the adapter's nearest-station decision, locked for the flight. V1 physical
launch-to-plane flight time and Isaac ground truth are forbidden.

## Action

3D normalized `[-1,1]^3`, mapped per axis and per side only into ONNX metadata
`hitter_pure_vel_planner_range_per_clip`:

```text
physical = low(side) + 0.5 * (normalized + 1) * (high(side) - low(side))
```

The normal training action domain excludes the native ±0.30 m/s gate margin.
It does not use the CORE+PLANNER bounding union. The native `CORE OR PLANNER`
helper exists only for parity/audit.

## Position

`POSITION_CONTRACT = HOPE_OPEN_SOURCE_SOLVER_POSITION`

Nominal target position is exactly the open-source solver
`/racket/command.position`, currently the predicted ball centre at strike.
No 17 mm proxy, radius, paddle thickness, or TCP offset is added.

This reproduces the current HOPE open-source software pipeline. It is not
evidence of a validated real-hardware racket TCP calibration.

## Side and station

For the first valid command of a new flight:

```text
anchor = held_station if available else current_base_xy
candidate[c] = target_xy - reach_offset_clip[c]
side = +1 if distance(candidate[0], anchor) <= distance(candidate[1], anchor)
       else -1
```

Tie selects forehand. `+1→clip0`, `-1→clip1`. Side is immutable for later
revisions of that flight. The station mirror updates only after a candidate
passes the adapter contract gate.

`ADAPTER_STATION_MIRROR = PROTOTYPE_REQUIRES_NATIVE_PARITY_TEST`

## Frame

`FRAME_CONTRACT = HOPE_OPEN_SOURCE_WORLD_TABLE_FRAME_CODE_0`

All packets use `frame_code=0` and world-labelled HOPE software data. No
base-link action is accepted.

This is an open-source software-frame contract, not proof of physical
mocap/table calibration.

## Timing

```text
age = max(0, current_source_clock - command.header.stamp)
control_tts = command.time_to_strike - age
absolute_strike_wall = encoded_producer_wall + control_tts
```

Non-finite, incomparable-clock, or expired commands fail closed. Native code
converts the wall deadline once to its monotonic countdown. This prototype does
not claim that source and hardware wall clocks are synchronized.

## Schema-2

The output is `numpy.float64 shape (19,)` with the native fixed layout.
`estimator_sample_count=0` and `estimator_span_s=0.0` are permitted by the
native parser and do not enter the actor/control contract.

Flight/revision lifecycle is deterministic `POSSIBLE_WITH_RULE`, based on
absolute strike-time shot reuse rather than every invalid→valid transition.
IDs and command sequence are positive and strictly increasing as required.

## Boundary

- no hardware TCP calibration claim;
- no mocap/table physical calibration claim;
- no sim2real claim;
- no real robot validation;
- no SAC V2 training yet;
- no modification to model_21800, deploy.yaml, native runner, or 110D contract.
