#!/usr/bin/env bash
# Official HOPE Gate3 runner adapter for the local A5 model_2800 package.
# The Gate3 conductor launches this file from its official MuJoCo/ROS2 chain.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="${PP_DIST:-$PROJECT_ROOT/a3_deploy/a3_deploy_example/dist/a3_deploy_x86_64}"
POLICY_DIR="${PP_POLICY_DIR:-$PROJECT_ROOT/a3_deploy/a3_deploy_example/models/model_a5_2800/policy}"
RUNTIME_CFG="${PP_RUNTIME_CFG:-$DIST/config/a3_runtime_config.pingpong.hitter_pingpong.yaml}"
ROS2_AIMRT_CFG="${PP_ROS2_AIMRT_CFG:-$DIST/config/a3_aimrt_config.pingpong_ros2body.yaml}"
RUNNER="$DIST/a3_deploy_onnx_ref_pingpong"
ORT_DIR="${PP_ORT_DIR:-$PROJECT_ROOT/a3_deploy/a3_deploy_example/thirdparty/onnxruntime/onnxruntime-linux-x64-1.19.2/lib}"
SIM_INSTALL="${PP_SIM_INSTALL:-$PROJECT_ROOT/agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/build/install}"
ROS_LIB="${PP_ROS_LIB:-/home/bistu/anaconda3/envs/hope-ros/lib}"

for required in "$RUNNER" "$RUNTIME_CFG" "$ROS2_AIMRT_CFG" "$POLICY_DIR/params/deploy.yaml" "$POLICY_DIR/exported/policy.onnx"; do
  if [[ ! -e "$required" ]]; then
    echo "[gate3-model2800] missing required path: $required" >&2
    exit 2
  fi
done

export LD_LIBRARY_PATH="$DIST:$ORT_DIR:$SIM_INSTALL/lib:$SIM_INSTALL/bin:$ROS_LIB:${LD_LIBRARY_PATH:-}"
exec "$RUNNER" \
  --runtime-cfg "$RUNTIME_CFG" \
  --aimrt-cfg "$ROS2_AIMRT_CFG" \
  --policy-dir "$POLICY_DIR" \
  "$@"
