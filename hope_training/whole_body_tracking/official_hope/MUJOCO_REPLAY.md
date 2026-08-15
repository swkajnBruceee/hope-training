# Project-local MuJoCo replay

This directory contains the project-local MuJoCo runtime for the
`model_21800` actor. It uses the 110-D `hitter_pure` observation contract and
the project 31-D action adapter. It is separate from the older project-level
`a3_deploy_example` directory.

Run from the training workspace:

```bash
./scripts/run_mujoco_play.sh --view --realtime --duration 120
```

Useful variants:

```bash
# Run without a viewer for a fixed duration.
./scripts/run_mujoco_play.sh --duration 20

# Hold the robot without the example command feed.
./scripts/run_mujoco_play.sh --view --realtime --idle --duration 60
```

The launcher uses the `hope-isaac` Python environment by default. To use a
different environment, set `HOPE_MUJOCO_PYTHON` to its Python executable.
