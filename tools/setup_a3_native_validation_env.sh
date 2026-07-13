#!/usr/bin/env bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROS_SETUP="/home/bruce/hope_ws_hopett_ros/install/setup.bash"
ROS2_PLUGIN_SETUP="${REPO_ROOT}/agibot_a3_aimdk/prebuilt/ros2_plugin_proto_x86_64/share/ros2_plugin_proto/local_setup.bash"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "[a3-native] missing ROS setup: ${ROS_SETUP}" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "${ROS2_PLUGIN_SETUP}" ]]; then
  echo "[a3-native] missing ros2_plugin_proto x86_64 setup: ${ROS2_PLUGIN_SETUP}" >&2
  echo "[a3-native] build it from agibot_a3_aimdk/protocol/ros2/ros2_plugin_proto first." >&2
  return 1 2>/dev/null || exit 1
fi

source "${ROS_SETUP}"
source "${ROS2_PLUGIN_SETUP}"

ROS2_PLUGIN_PREFIX="${REPO_ROOT}/agibot_a3_aimdk/prebuilt/ros2_plugin_proto_x86_64"

export PATH="${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${ROS2_PLUGIN_PREFIX}/lib:${ROS2_PLUGIN_PREFIX}/lib/python3.11/site-packages/ros2_plugin_proto:${LD_LIBRARY_PATH:-}"
export HOPE_A3_NATIVE_VALIDATION_READY=1

echo "[a3-native] validation env ready."
echo "[a3-native] ROS setup: ${ROS_SETUP}"
echo "[a3-native] ros2_plugin_proto: ${ROS2_PLUGIN_SETUP}"
