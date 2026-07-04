# Development Workflow

This repository is used by multiple developers. Keep the main branch runnable,
keep local machine state out of Git, and make environment changes visible.

## Branches

Use short feature branches from `main`:

```text
feature/planner-<topic>
feature/training-<topic>
fix/<topic>
docs/<topic>
```

Keep `main` as the branch that another developer can pull and run using the
documented setup.

## Before Committing

Run:

```bash
git status --short
./check.sh
```

Review the file list before committing. Do not commit:

```text
build/
install/
log/
external_repos/
vendor_assets/
setup_train_env.local.sh
.env
large generated CSV/NPZ/BAG/checkpoint/model files
```

## Pull Request / Merge Notes

Every merge should state which layer changed:

| Layer | Examples |
|-------|----------|
| Planner / ROS | `hope_ws/`, `docs/interfaces/ros_topics.md` |
| Training / Isaac | `hope_training/whole_body_tracking/` |
| Assets | `agibot/URDF/`, generated Isaac asset tooling |
| Deployment | `agibot/code_deployment/` |
| Environment | Dockerfiles, setup scripts, dependency versions |
| Docs | `README.md`, `docs/`, reference setup files |

If an environment version changes, update `docs/dev_environment.md` and the
related setup script or template in the same merge.

## Conflict Policy

Do not resolve conflicts by overwriting another developer's local changes. First
identify whether the conflict is code, configuration, generated output, or local
machine state. Generated output and local machine state should normally be
removed from the commit rather than reconciled as source.
