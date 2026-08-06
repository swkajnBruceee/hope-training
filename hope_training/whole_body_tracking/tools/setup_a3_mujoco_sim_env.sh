#!/usr/bin/env bash

# Environment for the local A3 MuJoCo/AimRT body_drive validation path.
# This is a simulator-side validation environment, not the official MC runtime.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this file: source $0" >&2
  exit 2
fi

_HOPE_WBT_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_HOPE_WBT_ROOT="$(cd "${_HOPE_WBT_TOOLS_DIR}/.." && pwd)"
_HOPE_PROJECT_ROOT="$(cd "${_HOPE_WBT_ROOT}/../.." && pwd)"
export HOPE_PROJECT_ROOT="${HOPE_PROJECT_ROOT:-${_HOPE_PROJECT_ROOT}}"

export HOPE_A3_MUJOCO_SIM_ROOT="${HOPE_A3_MUJOCO_SIM_ROOT:-${HOPE_PROJECT_ROOT}/agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim}"
export HOPE_A3_MUJOCO_SIM_INSTALL="${HOPE_A3_MUJOCO_SIM_INSTALL:-${HOPE_A3_MUJOCO_SIM_ROOT}/build_ascii/install}"
export HOPE_A3_MUJOCO_SIM_BIN="${HOPE_A3_MUJOCO_SIM_INSTALL}/bin"
export HOPE_A3_MUJOCO_SIM_PYTHON="${HOPE_A3_MUJOCO_SIM_INSTALL}/lib/python3.11/site-packages"
export HOPE_ROS_PREFIX="${HOPE_ROS_PREFIX:-/home/bruce/hope_ws_hopett_ros/install}"
export HOPE_ROS_ENV="${HOPE_ROS_ENV:-/workspace/anaconda3/envs/hope_ros}"

if [[ -f "${HOPE_ROS_PREFIX}/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${HOPE_ROS_PREFIX}/setup.bash"
else
  echo "[a3-mujoco] ROS prefix not found: ${HOPE_ROS_PREFIX}" >&2
  return 1
fi

for _pkg in aimrt_msgs irobot_events_executor irobot_lock_free_events_queue joint_msgs mujoco_sim_msgs ros2_plugin_proto; do
  if [[ -f "${HOPE_A3_MUJOCO_SIM_INSTALL}/share/${_pkg}/local_setup.bash" ]]; then
    # shellcheck disable=SC1090
    source "${HOPE_A3_MUJOCO_SIM_INSTALL}/share/${_pkg}/local_setup.bash"
  fi
done

export PKG_CONFIG_PATH="${HOPE_ROS_ENV}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${HOPE_ROS_ENV}/lib:${HOPE_A3_MUJOCO_SIM_BIN}:${HOPE_A3_MUJOCO_SIM_INSTALL}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${HOPE_A3_MUJOCO_SIM_PYTHON}:${PYTHONPATH:-}"
export HOPE_A3_MUJOCO_SIM_READY=1

if [[ ! -x "${HOPE_A3_MUJOCO_SIM_BIN}/aimrt_main" ]]; then
  echo "[a3-mujoco] simulator is not built: ${HOPE_A3_MUJOCO_SIM_BIN}/aimrt_main" >&2
  return 1
fi

echo "[a3-mujoco] environment ready"
echo "[a3-mujoco] install: ${HOPE_A3_MUJOCO_SIM_INSTALL}"
echo "[a3-mujoco] body_drive ROS2 topics are enabled at 500 Hz"
