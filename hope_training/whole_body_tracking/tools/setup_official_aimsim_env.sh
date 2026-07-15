#!/usr/bin/env bash

# Official A3 AimSim + Motion Control runtime.
# This script only selects the official package paths; control logic stays in
# AimSim and motion_control_v3.0.19_x86_humble.tar.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this file: source $0" >&2
  exit 2
fi

_OFFICIAL_WBT_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_OFFICIAL_WBT_ROOT="$(cd "${_OFFICIAL_WBT_TOOLS_DIR}/.." && pwd)"
_OFFICIAL_PROJECT_ROOT="$(cd "${_OFFICIAL_WBT_ROOT}/../.." && pwd)"

export HOPE_PROJECT_ROOT="${HOPE_PROJECT_ROOT:-${_OFFICIAL_PROJECT_ROOT}}"
export AIMSIM_OFFICIAL_ROOT="${AIMSIM_OFFICIAL_ROOT:-${HOPE_PROJECT_ROOT}/third_party/aimsim_official}"
export AIMSIM_MOTION_CONTROL_ROOT="${AIMSIM_MOTION_CONTROL_ROOT:-${AIMSIM_OFFICIAL_ROOT}/motion_control_humble}"
export AIMSIM_USER_CONFIG="${AIMSIM_USER_CONFIG:-${AIMSIM_OFFICIAL_ROOT}/user_config}"
export AIMSIM_MUJOCO_USER_DIR="${AIMSIM_MUJOCO_USER_DIR:-${AIMSIM_USER_CONFIG}/mujoco}"
export AIMSIM_PYTHON="${AIMSIM_PYTHON:-${HOPE_PROJECT_ROOT}/.venv/aimsim/bin/python}"
export AIMSIM_NATIVE_LIB="${AIMSIM_NATIVE_LIB:-${AIMSIM_PYTHON%/bin/python}/lib/python3.10/site-packages/aimsim/native/lib}"
export AGIBOT_ROBOT_MODEL="${AGIBOT_ROBOT_MODEL:-A3_T2D5}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-232}"
export SIM_MODE="${SIM_MODE:-sil}"
export MUJOCO_ROBOT="${MUJOCO_ROBOT:-raise_a3_t2d5}"
export LOG_PATH="${LOG_PATH:-${AIMSIM_OFFICIAL_ROOT}/logs/motion_control}"

if [[ ! -x "${AIMSIM_PYTHON}" ]]; then
  echo "[official-aimsim] missing Python: ${AIMSIM_PYTHON}" >&2
  return 1
fi
if [[ ! -f "${AIMSIM_PYTHON%/bin/python}/bin/aimsim" ]]; then
  echo "[official-aimsim] AimSim is not installed in ${AIMSIM_PYTHON%/bin/python}" >&2
  return 1
fi
if [[ ! -x "${AIMSIM_MOTION_CONTROL_ROOT}/bin/motion_control" ]]; then
  echo "[official-aimsim] missing official motion_control binary under ${AIMSIM_MOTION_CONTROL_ROOT}" >&2
  return 1
fi
if [[ ! -x "${AIMSIM_MOTION_CONTROL_ROOT}/scripts/motion_control/start_motion_control.sh" ]]; then
  echo "[official-aimsim] missing official start_motion_control.sh" >&2
  return 1
fi

# The official x86 bundle was built against GLIBCXX_3.4.32. The host image
# ships an older libstdc++; AimSim bundles a compatible one, so preload only
# that library without changing the system installation.
if [[ -f "${AIMSIM_NATIVE_LIB}/libstdc++.so.6" ]]; then
  export LD_PRELOAD="${AIMSIM_NATIVE_LIB}/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
else
  echo "[official-aimsim] missing bundled libstdc++.so.6: ${AIMSIM_NATIVE_LIB}" >&2
  return 1
fi

mkdir -p "${LOG_PATH}"
export PATH="${AIMSIM_PYTHON%/bin/python}/bin:${PATH}"

echo "[official-aimsim] environment ready"
echo "[official-aimsim] python=${AIMSIM_PYTHON}"
echo "[official-aimsim] robot=${AGIBOT_ROBOT_MODEL} mujoco_robot=${MUJOCO_ROBOT} mode=${SIM_MODE} domain=${ROS_DOMAIN_ID}"
echo "[official-aimsim] motion_control=${AIMSIM_MOTION_CONTROL_ROOT}"
echo "[official-aimsim] user_config=${AIMSIM_USER_CONFIG}"
