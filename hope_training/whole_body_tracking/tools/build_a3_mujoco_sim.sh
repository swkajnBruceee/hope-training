#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WBT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${WBT_ROOT}/../.." && pwd)"
SIM_ROOT="${PROJECT_ROOT}/agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim"
ASCII_ROOT="${HOME}/hopett_sim"
BUILD_DIR="${ASCII_ROOT}/build_ascii"

if [[ ! -e "${ASCII_ROOT}" ]]; then
  ln -s "${SIM_ROOT}" "${ASCII_ROOT}"
fi

source "${SCRIPT_DIR}/setup_a3_mujoco_sim_env.sh" 2>/dev/null || true
source "${HOPE_ROS_PREFIX:-/home/bruce/hope_ws_hopett_ros/install}/setup.bash"
export PKG_CONFIG_PATH="${HOPE_ROS_ENV:-/workspace/anaconda3/envs/hope_ros}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${HOPE_ROS_ENV:-/workspace/anaconda3/envs/hope_ros}/lib:${LD_LIBRARY_PATH:-}"

cmake -S "${ASCII_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \
  -DAIMRT_MUJOCO_SIM_BUILD_WITH_ROS2=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-I${HOPE_ROS_ENV:-/workspace/anaconda3/envs/hope_ros}/include" \
  -DCMAKE_CXX_FLAGS="-I${HOPE_ROS_ENV:-/workspace/anaconda3/envs/hope_ros}/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${HOPE_ROS_ENV:-/workspace/anaconda3/envs/hope_ros}/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L${HOPE_ROS_ENV:-/workspace/anaconda3/envs/hope_ros}/lib"
cmake --build "${BUILD_DIR}" --target install --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
echo "[a3-mujoco] build complete: ${BUILD_DIR}/install"
