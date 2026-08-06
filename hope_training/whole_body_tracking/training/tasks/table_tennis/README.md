# HOPE Table-Tennis Match Environment (Isaac Lab)

A clean, modular Isaac Lab task that simulates a **standard table-tennis competition scene** — floor,
table, net (+ posts), a dynamic ITTF ball, and the **Agibot A3** humanoid — with realistic ball flight,
bounce, and racket/table contact. Built as a manager-based `ManagerBasedRLEnv` so RL training can be
layered on later; the first goal here is a **correct physics + visualization scene**.

Gym id: **`HOPE-TableTennis-AgibotA3-v0`**

## Coordinate frame (canonical HOPE frame, used everywhere)

The simulation world frame **is** the HOPE frame (ROS 2 REP-103), identical to the planner / mocap docs
and `hope_ws/.../hope_world_frame.yaml`:

| Axis | Direction | Range on the table |
|------|-----------|--------------------|
| **X** | toward Player Two (P2), along the table length | `0 → +2.74 m` |
| **Y** | left, from P1's perspective, along the table width | `0 → −1.525 m` |
| **Z** | up; **z = 0 is the table surface** | floor at `z = −0.76 m` |

Origin = the **near-side left corner of the table surface** (P1 perspective). Each parallel environment
is an independent court whose local origin coincides with this HOPE origin, so an asset's
environment-local position *is* its HOPE-frame position. Landmarks (net center `(1.37, −0.7625, 0)`,
P1/P2 half centers, floor at `−0.76`) all match the reference docs — see [`geometry.py`](geometry.py),
which is the single source of truth and is regression-tested against the ITTF/HOPE constants in
`tests/test_table_tennis_geometry.py`.

## How to run (visualization, no policy)

This needs Isaac Sim / Isaac Lab. The launcher
`hope_isaac_py` (defined by `setup_train_env.sh`) runs Isaac's bundled Python with the working-tree
PYTHONPATH, so your local edits to this task win:

```bash
cd /path/to/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh                          # defines hope_isaac_py + sets PYTHONPATH

hope_isaac_py scripts/play_table_tennis.py                 # 1 court, robot free-standing, drag on
hope_isaac_py scripts/play_table_tennis.py --num_envs 9    # 9 courts
hope_isaac_py scripts/play_table_tennis.py --fix_base      # pin the pelvis (stable view; no balance policy yet)
hope_isaac_py scripts/play_table_tennis.py --disable_aero  # PhysX gravity + contacts only (no air drag)
hope_isaac_py scripts/play_table_tennis.py --magnus 0.1    # enable Magnus (spin) lift + serve spin
hope_isaac_py scripts/play_table_tennis.py --headless --steps 300   # no-GUI smoke run (loads + steps)
```

Each reset serves a ball from over the P2 half toward the P1-side robot; you should see it arc (with
drag), bounce on the table (restitution ≈ `C_v`), and continue toward the robot. The robot holds its
default standing pose (zero action). The console prints `ball aerodynamics active: True/False` so you
can confirm the drag callback registered.

Notes:
* **First launch is slow** — `UrdfFileCfg` converts the A3 URDF → USD once, then caches it.
* **Without `--fix_base` the robot may drift/topple** after a few seconds (there is no balance/return
  controller yet — that is the RL follow-up). Use `--fix_base` for a stable view of the ball physics.

## Physics model

* **Gravity + all rigid-body contacts** (ball↔table / net / floor / racket) are handled natively by
  PhysX, with per-surface contact materials defined in `geometry.BounceMaterials`.
* **Aerodynamic drag** is the one thing PhysX cannot model for a 40 mm ball. It is added every physics
  substep (360 Hz) by `TableTennisEnv` via a physics-step callback, using the HOPE-calibrated model
  `a_drag = −k|v|v` (`k = 0.5 s/m`, matching `hope_planner.constants.BallPhysics`). See
  [`ball.py`](ball.py). **Magnus (spin) lift** is provided but **off by default** (the HOPE planner
  neglects spin); enable via `ball_aerodynamics.magnus_coefficient`.
* Physics runs at **360 Hz** (`sim.dt = 1/360 s`), control at **90 Hz** (`decimation = 4`). The high
  physics rate keeps the small, fast ball from tunnelling through the thin racket blade / net.

## Modularity / extension points

| Concern | Where |
|---|---|
| Table / net / ball dimensions, landmarks, materials, serve, bounds | [`geometry.py`](geometry.py) |
| Ball aerodynamics (drag / Magnus) math + config | [`ball.py`](ball.py) |
| Scene assets (one `build_*` helper per prim) + MDP managers | [`table_tennis_env_cfg.py`](table_tennis_env_cfg.py) |
| Per-substep force application / env class | [`table_tennis_env.py`](table_tennis_env.py) |
| Ball/robot observations, serve event, rewards, terminations | [`mdp/`](mdp/) |
| Robot choice, stand pose, action scale, Gym registration | [`config/agibot_a3/`](config/agibot_a3/) |

To swap the box table for a USD asset, change `build_table_top_cfg`. To add a second robot (P2), add an
articulation to the scene cfg. To add real match rewards (return-over-net, landing in the opponent half,
racket-to-ball tracking), add terms in `RewardsCfg` / `mdp/rewards.py`.

## Known limitations / calibration TODOs

These are first-pass defaults chosen for a visibly correct scene; they need an in-sim calibration pass
(the PhysX analogue of `hope_planner.calibrate_ball_physics`). All knobs live in `geometry.py` /
`ball.py` so calibration touches one place:

1. **Horizontal restitution `C_h = 0.75`** (tangential velocity loss on bounce) is governed by PhysX
   friction, not by a restitution coefficient, so it is only approximated. **Ball↔racket restitution
   `C_r = 0.88`** differs slightly from the table `C_v = 0.85`; the racket currently inherits the
   robot's contact material. Exact `C_h` / `C_r` matching is a calibration TODO.
2. **Robot facing**: the A3 is spawned with identity orientation (assumed +X-forward). If the URDF
   forward axis is −X, flip `init_state.rot` in `config/agibot_a3/table_tennis_env_cfg.py`.
3. **Balance**: there is no balance/return policy yet, so the free-standing robot may drift/topple over
   several seconds — use `--fix_base` for a stable visualization. A returner policy is the RL follow-up.
4. **Very fast smashes** (≫ 8 m/s) may still need PhysX CCD on the ball to avoid tunnelling through the
   2.9 mm racket blade; the 360 Hz physics rate covers normal serve/rally speeds.

## Verification status

* `tests/test_table_tennis_geometry.py` (frame/geometry + drag/Magnus math) — **passing** on a plain
  Python host (drag/Magnus tests auto-skip if `torch` is missing).
* All modules pass `py_compile`.
* **In-sim runtime** (asset spawn, contacts, the aero callback, robot stand) must be verified inside the
  Isaac Lab environment with `scripts/play_table_tennis.py` — it has not been run on a host without
  Isaac Sim.
