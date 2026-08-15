#!/usr/bin/env bash
# Run the model_21800 policy through the project-local MuJoCo replay.
# The existing project-level a3_deploy_example is intentionally not modified.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
BUNDLE_DIR="${PROJECT_DIR}/mujoco_reference"
REF_DIR="${BUNDLE_DIR}/reference"
CONFIG_PATH="${BUNDLE_DIR}/config/hope_pingpong_runtime.yaml"

# Override this when using another compatible Python environment.
HOPE_MUJOCO_PYTHON="${HOPE_MUJOCO_PYTHON:-/home/bistu/anaconda3/envs/hope-isaac/bin/python}"

if [[ ! -x "${HOPE_MUJOCO_PYTHON}" ]]; then
  echo "MuJoCo Python not found: ${HOPE_MUJOCO_PYTHON}" >&2
  echo "Set HOPE_MUJOCO_PYTHON to an environment containing mujoco, glfw, and onnxruntime." >&2
  exit 2
fi

export MUJOCO_GL="${MUJOCO_GL:-glx}"
export PYTHONPATH="${REF_DIR}:${PYTHONPATH:-}"

exec "${HOPE_MUJOCO_PYTHON}" -m a3_deploy_onnx_ref_pingpong \
  --config "${CONFIG_PATH}" \
  --backend mujoco \
  "$@"
