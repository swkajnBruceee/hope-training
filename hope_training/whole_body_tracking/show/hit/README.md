# Hit Overlay (Isaac viewport)

This package is the Isaac-viewport display half of the HOPE hit plan. It does
not predict the ball path, choose a hit point, solve racket orientation, or
adjust timing.

It mirrors [`show/trajectory/`](../trajectory/README.md) 1:1:

* Same backend (`debug_draw`), same stale-frame policy, same UDP shape (only
  the magic differs -- `"HITS"` here vs `"HTRJ"` for the trajectory).
* Subscribed topic: `/hit/state` (`msgs/msg/HitState`).
* UDP source: `hit_state_udp_bridge` (C++ node in `hope_ws/src/bringup`).
* Default UDP endpoint: `127.0.0.1:19533`.

## What gets drawn

For every received `HitState` (when `valid == true` and the planned hit
point is inside the scene bounds):

| Element | Color | Source field |
|---|---|---|
| Hit-point cross | red-orange | `hit_position` |
| Target cross | blue | `target_land` |
| Hit -> target line | blue | `hit_position` -> `target_land` |
| Ball velocity (outgoing) arrow | orange-red | `ball_velocity_outgoing` |
| Ball velocity (incoming) arrow | gray | `ball_velocity_incoming` |
| Racket velocity arrow | teal | `racket_velocity` |
| Racket normal arrow | dark slate | `racket_normal` |

When `valid == false` or the planner has no plan yet, the previous frame is
held for `stale_keep_s` (default 0.3 s) and then cleared.

## Run

The hit overlay is started automatically by
[`scripts/play_table_tennis_ros.sh`](../../../scripts/play_table_tennis_ros.sh)
alongside the trajectory overlay and the ROS 2 nodes. There is no separate
launcher -- the Isaac-side Python overlay runs inside the same Isaac Sim
process that drives the simulation.

To disable just the hit overlay (and keep the trajectory overlay), pass
`--no-hit-overlay` to the launcher script. To disable both, use
`--no-trajectory-overlay` and `--no-hit-overlay`.

## Computation

The hit-state computation lives in the ROS solver node
(`hope_ws/src/solver`). This directory is only for display.
