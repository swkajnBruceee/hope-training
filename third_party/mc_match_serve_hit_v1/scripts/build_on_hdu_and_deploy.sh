#!/usr/bin/env bash
set -Eeuo pipefail

# This script is intentionally limited to user-package build/deploy actions.
# It never starts, stops, restarts, or signals a robot controller.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Keep the HDU source and build products together on the persistent /agibot
# partition. Override with MC_BUILD_DIR when another persistent location is
# required.
BUILD_DIR="${MC_BUILD_DIR:-$ROOT/build_hdu}"
RUNTIME_DIR="${MC_RUNTIME_DIR:-$ROOT/dist/rockchip}"
MDU_HOST="${MDU_HOST:-mdu}"
MDU_USER="${MDU_USER:-agi}"
MDU_PASSWORD=1
MDU_ROOT="${MDU_ROOT:-/agibot/data/user_deploy/mc}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "[mc] refusing to build: this entry point must run on the HDU (aarch64)" >&2
  exit 2
fi

if [[ -r /opt/ros/jazzy/setup.bash ]]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u
fi
if [[ -r /agibot/software/v0/entry/env/env.sh ]]; then
  # shellcheck disable=SC1091
  set +u
  source /agibot/software/v0/entry/env/env.sh
  set -u
fi

# The board environment may define an unrelated MDU_PASSWORD variable. Use a
# dedicated override so the deployment credential is not silently replaced by
# the environment setup script.
MDU_PASSWORD=1

export HAS_ROS2=1
export CMAKE_PREFIX_PATH="${ROOT}/thirdparty/unitree_sdk2:${CMAKE_PREFIX_PATH:-}"

MC_FETCH_ROOT="$ROOT/deps/fetchcontent"
AIMRT_LOCAL_SOURCE="${AIMRT_LOCAL_SOURCE:-}"
if [[ -d "$MC_FETCH_ROOT/aimrt-src" ]]; then
  AIMRT_LOCAL_SOURCE="$MC_FETCH_ROOT/aimrt-src"
elif [[ -z "$AIMRT_LOCAL_SOURCE" ]]; then
  echo "[mc] packaged AimRT source is missing: $MC_FETCH_ROOT/aimrt-src" >&2
  exit 2
fi

cmake_args=(
  -S "$ROOT" -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DGS_PACKAGE_ARCH_NAME=rockchip
  -DGS_RUNTIME_OUTPUT_DIR="$RUNTIME_DIR"
  -DENABLE_A3_AIMRT_BACKEND=ON
  -DENABLE_A3_ROS_MSGS=ON
  -DENABLE_RKNN_INFERENCE=OFF
  -DBUILD_MC_TESTS=OFF
)
fetch_src() {
  local cmake_name="$1" dir_name="$2"
  if [[ -d "$MC_FETCH_ROOT/$dir_name" ]]; then
    cmake_args+=("-DFETCHCONTENT_SOURCE_DIR_${cmake_name}=$MC_FETCH_ROOT/$dir_name")
  fi
}
fetch_src ZSTD zstd-src
fetch_src LZ4 lz4-src
fetch_src TBB tbb-src
fetch_src PROTOBUF protobuf-src
fetch_src JSONCPP jsoncpp-src
fetch_src YAML_CPP yaml-cpp_src
fetch_src FMT fmt-src
fetch_src ASIO asio-src
fetch_src GFLAGS gflags_src
fetch_src CPPTOML cpptoml-src
fetch_src LIBUNIFEX libunifex-src
fetch_src BACKWARD backward-src
fetch_src MCAP mcap_src
fetch_src IROBOT_EVENTS_EXECUTOR events-executor-src
fetch_src ICEORYX iceoryx-src
cmake_args+=("-DFETCHCONTENT_FULLY_DISCONNECTED=ON")
if [[ -n "$AIMRT_LOCAL_SOURCE" ]]; then
  cmake_args+=("-Daimrt_LOCAL_SOURCE=$AIMRT_LOCAL_SOURCE")
else
  echo "[mc] warning: no packaged/local AimRT source found; CMake may need network access" >&2
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --target a3_deploy_model3396 -j"${MC_BUILD_JOBS:-4}"

install -Dm755 "$RUNTIME_DIR/a3_deploy_model3396" "$ROOT/bin/a3_deploy_model3396"
mkdir -p "$ROOT/bin/runtime"
# These are generated/imported by this package build and are not guaranteed
# to exist in the MDU system image. Keep only the CPU ONNX Runtime and the
# generated joint_msgs type-support libraries; CUDA/TensorRT providers are not
# part of this RK3588 runtime path.
ORT_LIB_DIR="$BUILD_DIR/_deps/onnxruntime_aarch64/onnxruntime_gpu_1.19.2/lib"
if [[ -d "$ORT_LIB_DIR" ]]; then
  find "$ORT_LIB_DIR" -maxdepth 1 -type f -name 'libonnxruntime.so*' \
    -exec cp -f {} "$ROOT/bin/runtime/" \;
fi
if [[ -d "$BUILD_DIR/joint_msgs_build" ]]; then
  find "$BUILD_DIR/joint_msgs_build" -maxdepth 1 -type f -name 'libjoint_msgs*.so*' \
    -exec cp -f {} "$ROOT/bin/runtime/" \;
fi
# hope_msgs ships its own ROS message package; without these, the binary
# fails to dlopen libhope_msgs__rosidl_typesupport_cpp.so at startup.
if [[ -d "$BUILD_DIR/hope_msgs_build" ]]; then
  find "$BUILD_DIR/hope_msgs_build" -maxdepth 1 -type f -name 'libhope_msgs*.so*' \
    -exec cp -f {} "$ROOT/bin/runtime/" \;
fi

if [[ "${MC_SKIP_DEPLOY:-0}" == "1" ]]; then
  echo "[mc] build succeeded on HDU; MDU deployment skipped (MC_SKIP_DEPLOY=1)"
  exit 0
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "[mc] build succeeded, but sshpass is missing; binary was not copied to MDU" >&2
  exit 3
fi

sshpass -p "$MDU_PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$MDU_USER@$MDU_HOST" "mkdir -p '$MDU_ROOT/bin' '$MDU_ROOT/config' '$MDU_ROOT/model' '$MDU_ROOT/models' '$MDU_ROOT/arm' '$MDU_ROOT/scripts'"
sshpass -p "$MDU_PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$MDU_USER@$MDU_HOST" "chmod -R u+rwX '$MDU_ROOT/bin' '$MDU_ROOT/config' '$MDU_ROOT/model' '$MDU_ROOT/models' '$MDU_ROOT/arm' '$MDU_ROOT/scripts' 2>/dev/null || true"
sshpass -p "$MDU_PASSWORD" scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$ROOT/bin/a3_deploy_model3396" "$MDU_USER@$MDU_HOST:$MDU_ROOT/bin/a3_deploy_model3396"
sshpass -p "$MDU_PASSWORD" scp -rq -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$ROOT/config" "$ROOT/model" "$ROOT/models" "$ROOT/arm" "$ROOT/scripts" \
  "$MDU_USER@$MDU_HOST:$MDU_ROOT/"
find "$ROOT/bin/runtime" -maxdepth 1 -type f -name '*.so*' -print0 2>/dev/null |
  while IFS= read -r -d '' lib; do
    sshpass -p "$MDU_PASSWORD" scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "$lib" "$MDU_USER@$MDU_HOST:$MDU_ROOT/bin/"
  done

echo "[mc] build succeeded on HDU and user binary/runtime libraries were copied to $MDU_USER@$MDU_HOST:$MDU_ROOT"
