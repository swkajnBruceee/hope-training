# Local Isaac paths for HOPE. Copy this file to setup_train_env.local.sh and
# edit the values for your machine. The copied file is git-ignored.

export HOPE_ISAAC_PYTHON=/absolute/path/to/isaacsim/python.sh
export HOPE_ISAACLAB_ROOT=/absolute/path/to/IsaacLab

# Optional: site-packages with hydra/omegaconf if Isaac's Python cannot import
# them directly.
# export HOPE_ISAAC_VENV_SITE=/absolute/path/to/venv/lib/python3.10/site-packages

# Optional: only needed if you choose WandB logging or motion registry loading.
# export WANDB_ENTITY=your-wandb-team
# export WANDB_REGISTRY_ORG=your-wandb-org
# export WANDB_PROJECT=hope_wbc
