# Development Environment

This repository should keep code, reproducible environment definitions, and small
configuration templates in Git. Machine-specific paths, generated outputs, and
large runtime artifacts should stay outside Git.

## Supported Baseline

Use this as the shared target environment for collaborative development:

| Component | Baseline |
|-----------|----------|
| OS | Ubuntu 22.04 or compatible Linux |
| Python | 3.10+ |
| ROS 2 | Jazzy for the preserved mocap/planner references; keep package metadata explicit |
| Isaac Lab | 2.1.0 for the public A3 Isaac starter path |
| CUDA / GPU driver | Match the local Isaac Sim / Isaac Lab installation |
| C++ toolchain | CMake-based builds; cross-compile settings live under `agibot/code_deployment/a3_deploy_example/cmake/` |

If a developer runs a newer local version, they should keep the shared smoke
commands working or document the required change before merging.

## Local Setup Files

Local files are intentionally ignored:

| File | Purpose |
|------|---------|
| `hope_training/whole_body_tracking/setup_train_env.local.sh` | Local Isaac Sim, Isaac Lab, optional venv, and WandB paths |
| `.env`, `.env.*` | Local shell/environment secrets or overrides |
| `*.local.yaml`, `*.local.yml`, `*.local.json` | Local config overrides |

Commit templates instead:

| Template | Purpose |
|----------|---------|
| `hope_training/whole_body_tracking/setup_train_env.local.example.sh` | Shows required local Isaac variables |
| `.env.example` | Add one only if the project needs shared env variable names |

## Standard A3 Isaac Smoke Path

From the repository root:

```bash
python3 hope_training/whole_body_tracking/scripts/prepare_a3_isaac_asset.py --force
cd hope_training/whole_body_tracking
source setup_train_env.sh
hope_isaac_py -c "import training.tasks; print('HOPE tasks import ok')"
hope_isaac_py scripts/play_table_tennis.py --headless --steps 300
```

This path depends on local Isaac installation paths, but those paths must stay in
`setup_train_env.local.sh` or the developer shell, not in tracked files.

## Repository Check

Before pushing or merging, run:

```bash
./check.sh
```

The check is intentionally lightweight: it validates local-file ignore rules,
compiles repository Python modules, and runs planner unit tests when `pytest` and
its Python dependencies are available. It does not require Isaac Sim.
