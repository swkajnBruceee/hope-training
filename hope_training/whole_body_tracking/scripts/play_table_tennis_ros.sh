#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WBT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${WBT_DIR}/../.." && pwd)"

ROS_SETUP_CANDIDATES=()
if [ -n "${HOPE_ROS_WS_SETUP:-}" ]; then
  ROS_SETUP_CANDIDATES+=("${HOPE_ROS_WS_SETUP}")
else
  ROS_SETUP_CANDIDATES+=(
    "${REPO_DIR}/hope_ws/install/setup.bash"
    "${REPO_DIR}/hope_ws/install/setup.sh"
  )
fi

ROS_SETUP=""
for candidate in "${ROS_SETUP_CANDIDATES[@]}"; do
  if [[ "${candidate}" != /* ]]; then
    candidate="${REPO_DIR}/${candidate}"
  fi
  if [ -f "${candidate}" ]; then
    ROS_SETUP="${candidate}"
    break
  fi
done

ROS310_CANDIDATES=()
if [ -n "${HOPE_ROS310:-}" ]; then
  ROS310_CANDIDATES+=("${HOPE_ROS310}")
else
  ROS310_CANDIDATES+=(
    "${REPO_DIR}/hope_ws/install/hope_planner"
    "${REPO_DIR}/hope_ws/install"
    "/home/gzy/isaac_venv"
  )
fi

ROS310_PREFIX=""
for candidate in "${ROS310_CANDIDATES[@]}"; do
  if [[ "${candidate}" != /* ]]; then
    candidate="${REPO_DIR}/${candidate}"
  fi
  if [ -d "${candidate}/lib/python3.10/site-packages" ]; then
    ROS310_PREFIX="${candidate}"
    break
  fi
done

BALL_TOPIC="${HOPE_BALL_TOPIC:-/ball/point}"
START_PLANNER=1
PLANNER_PID=""

ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --no-planner)
      START_PLANNER=0
      ;;
    *)
      ARGS+=("${arg}")
      ;;
  esac
done

cleanup() {
  if [ -n "${PLANNER_PID}" ]; then
    kill -- "-${PLANNER_PID}" 2>/dev/null || kill "${PLANNER_PID}" 2>/dev/null || true
    wait "${PLANNER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [ -z "${ROS_SETUP}" ]; then
  echo "[hope_ros_run] ROS setup not found. Tried:" >&2
  for candidate in "${ROS_SETUP_CANDIDATES[@]}"; do
    if [[ "${candidate}" != /* ]]; then
      candidate="${REPO_DIR}/${candidate}"
    fi
    echo "  - ${candidate}" >&2
  done
  exit 1
fi

if [ -z "${ROS310_PREFIX}" ]; then
  echo "[hope_ros_run] ROS Python 3.10 packages not found. Tried:" >&2
  for candidate in "${ROS310_CANDIDATES[@]}"; do
    if [[ "${candidate}" != /* ]]; then
      candidate="${REPO_DIR}/${candidate}"
    fi
    echo "  - ${candidate}/lib/python3.10/site-packages" >&2
  done
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u

if [ "${START_PLANNER}" -eq 1 ]; then
  setsid ros2 launch hope_planner hope_planner.launch.py &
  PLANNER_PID=$!
  echo "[hope_ros_run] planner started: pid=${PLANNER_PID}"
  sleep 2
else
  echo "[hope_ros_run] planner startup skipped."
fi

cd "${WBT_DIR}"
set +u
# shellcheck disable=SC1091
source "${WBT_DIR}/setup_train_env.sh"
set -u

export HOPE_ROS310="${ROS310_PREFIX}"
export PYTHONPATH="${HOPE_WBT_PYTHONPATH}:${ROS310_PREFIX}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ROS310_PREFIX}/lib"

echo "[hope_ros_run] starting Isaac table-tennis sim"
echo "[hope_ros_run] publishing ball truth on ${BALL_TOPIC}"
"${HOPE_ISAAC_PYTHON}" "${SCRIPT_DIR}/play_table_tennis.py" \
  --publish-ball-truth \
  --ball-truth-topic "${BALL_TOPIC}" \
  "${ARGS[@]}"
