# =============================================================================
# HOPE WBC training environment. SOURCE this inside your GPU/Isaac shell before
# running scripts/train.py / scripts/play.py:
#
#   cd /path/to/HOPE/hope_training/whole_body_tracking
#   source setup_train_env.sh
#
# It (1) sets the PYTHONPATH that lets Isaac's bundled python see hydra/omegaconf
# plus isaaclab/isaaclab_rl, (2) defines the `hope_isaac_py` launcher, and
# (3) exposes optional WandB placeholders for teams that choose to use it.
#
# MUST be SOURCED, not executed (`./setup_train_env.sh` would set everything in a
# subshell that then exits, leaving your shell unchanged). Re-source it in every
# new terminal. Safe to re-source. Idempotent.
#
# ---------------------------------------------------------------------------
# SITE-SPECIFIC PATHS — EDIT THESE FOR YOUR MACHINE.
# This is a public branch, so the values below are PLACEHOLDERS, not a working
# install. Set them to match your own Isaac Sim / Isaac Lab install in ONE of:
#   1. your shell profile / the GPU environment (export ... before sourcing), or
#   2. a git-ignored `setup_train_env.local.sh` next to this file (auto-sourced
#      below) — recommended, keeps machine paths out of git.
# A from-scratch Isaac Sim / Isaac Lab install is not vendored here; follow the
# upstream Isaac Lab install guide for your OS and GPU driver. Typical values are:
#   HOPE_ISAAC_PYTHON   = <isaacsim>/python.sh
#   HOPE_ISAACLAB_ROOT  = <IsaacLab checkout containing source/isaaclab*>
#   HOPE_ISAAC_VENV_SITE= site-packages providing hydra/omegaconf (if Isaac's
#                         python lacks them)
# =============================================================================

# Absolute path to this script's dir (.../hope_training/whole_body_tracking), so
# sourcing works from any cwd and any clone location.
_WBT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional git-ignored local override (real machine paths + optional WandB identity).
if [ -f "${_WBT_DIR}/setup_train_env.local.sh" ]; then
  # shellcheck disable=SC1091
  source "${_WBT_DIR}/setup_train_env.local.sh"
fi

# --- site-specific defaults (overridden by exports above or the local file) ---
: "${HOPE_ISAAC_PYTHON:=/opt/isaacsim/python.sh}"      # EDIT: your Isaac Sim python.sh
: "${HOPE_ISAACLAB_ROOT:=/opt/IsaacLab}"               # EDIT: Isaac Lab checkout (has source/isaaclab*)
: "${HOPE_ISAAC_VENV_SITE:=}"                          # EDIT (optional): venv site-packages with hydra/omegaconf

# Training PYTHONPATH: the package parent comes FIRST so ``import training`` and
# the sibling ``show/*`` overlays resolve from the working tree instead of any
# stale site-packages install. Then append optional venv deps (hydra/omegaconf)
# and the four Isaac Lab source dirs. isaaclab_rl is required for training
# (replay/csv_to_npz did not need it).
_il="${HOPE_ISAACLAB_ROOT}/source"
HOPE_WBT_PYTHONPATH="${_WBT_DIR}"
if [ -n "${HOPE_ISAAC_VENV_SITE}" ]; then
  HOPE_WBT_PYTHONPATH="${HOPE_WBT_PYTHONPATH}:${HOPE_ISAAC_VENV_SITE}"
fi
HOPE_WBT_PYTHONPATH="${HOPE_WBT_PYTHONPATH}:${_il}/isaaclab:${_il}/isaaclab_tasks:${_il}/isaaclab_assets:${_il}/isaaclab_rl"
export HOPE_WBT_PYTHONPATH
export HOPE_ISAAC_PYTHON

# Launch Isaac's python with the training PYTHONPATH. The `${VAR:-<default>}` form
# guards against an empty PYTHONPATH (the `ModuleNotFoundError: No module named
# 'hydra'` cause) if only the function got redefined in a fresh shell.
hope_isaac_py () {
  PYTHONPATH="${HOPE_WBT_PYTHONPATH}" "${HOPE_ISAAC_PYTHON:-/opt/isaacsim/python.sh}" "$@"
}

# WandB is optional. The public quickstart uses logger=tensorboard and motion_file=...
# without these values. Set them only if you use WandB logging or a motion registry.
export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-team}"          # team (run logging)
export WANDB_REGISTRY_ORG="${WANDB_REGISTRY_ORG:-your-wandb-org}"  # org  (motion registry)
export WANDB_PROJECT="${WANDB_PROJECT:-hope_wbc}"

unset _WBT_DIR _il

echo "[hope] training env ready."
echo "[hope]   hope_isaac_py -> ${HOPE_ISAAC_PYTHON} (+ hydra/omegaconf + isaaclab_rl)"
echo "[hope]   WANDB_ENTITY=$WANDB_ENTITY  WANDB_REGISTRY_ORG=$WANDB_REGISTRY_ORG  WANDB_PROJECT=$WANDB_PROJECT"
if [ ! -x "${HOPE_ISAAC_PYTHON}" ]; then
  echo "[hope]   WARNING: HOPE_ISAAC_PYTHON='${HOPE_ISAAC_PYTHON}' is not executable — edit the site-specific paths (see header)."
fi
