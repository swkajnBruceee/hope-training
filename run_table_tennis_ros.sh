#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCII_REPO="${HOPE_ASCII_REPO:-/tmp/HOPETableTennis}"
BUILD_BASE="${HOPE_BUILD_BASE:-/tmp/hope_build}"
INSTALL_BASE="${HOPE_INSTALL_BASE:-/tmp/hope_install}"
LOG_BASE="${HOPE_LOG_BASE:-/tmp/hope_log}"
ROS_ENV="${HOPE_ROS_ENV:-hope_ros}"
ROS_UNDERLAY="${HOPE_ROS_UNDERLAY_SETUP:-/workspace/anaconda3/envs/${ROS_ENV}/setup.bash}"

ln -sfn "${REPO_DIR}" "${ASCII_REPO}"

if [ ! -f "${INSTALL_BASE}/setup.bash" ] || [ "${HOPE_FORCE_BUILD:-0}" = "1" ]; then
  set +u
  # shellcheck disable=SC1091
  source /workspace/anaconda3/etc/profile.d/conda.sh
  conda activate "${ROS_ENV}"
  # shellcheck disable=SC1090
  source "${ROS_UNDERLAY}"
  set -u
  (
    cd "${ASCII_REPO}/hope_ws"
    colcon --log-base "${LOG_BASE}" build \
      --build-base "${BUILD_BASE}" \
      --install-base "${INSTALL_BASE}" \
      --packages-select common msgs trajectory solver decision bringup \
      --event-handlers console_direct+
  )
fi

export HOPE_ROS_UNDERLAY_SETUP="${ROS_UNDERLAY}"
export HOPE_ROS_WS_SETUP="${INSTALL_BASE}/setup.bash"
export LD_LIBRARY_PATH="/workspace/anaconda3/envs/${ROS_ENV}/lib:${INSTALL_BASE}/common/lib:${INSTALL_BASE}/msgs/lib:${INSTALL_BASE}/trajectory/lib:${INSTALL_BASE}/solver/lib:${INSTALL_BASE}/decision/lib:${INSTALL_BASE}/bringup/lib:${LD_LIBRARY_PATH:-}"

cd "${ASCII_REPO}"
exec ./hope_training/whole_body_tracking/scripts/play_table_tennis_ros.sh --fix_base --hide-robot "$@"
