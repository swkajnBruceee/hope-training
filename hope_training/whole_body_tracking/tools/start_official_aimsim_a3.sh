#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_official_aimsim_env.sh"

if [[ "${SIM_MODE}" != "sil" ]]; then
  echo "[official-aimsim] only SIL is supported by this project launcher; got ${SIM_MODE}" >&2
  exit 2
fi

"${AIMSIM_PYTHON}" -m aimsim.cli mujoco init-config \
  --user-config-path "${AIMSIM_USER_CONFIG}" \
  --robot "${MUJOCO_ROBOT}" >/dev/null

MC_SCRIPT="${AIMSIM_MOTION_CONTROL_ROOT}/scripts/motion_control/start_motion_control.sh"
MC_LOG_DIR="${AIMSIM_OFFICIAL_ROOT}/logs/motion_control"
mkdir -p "${MC_LOG_DIR}"

cleanup() {
  trap - TERM INT EXIT
  if [[ -n "${MC_PID:-}" ]] && kill -0 "${MC_PID}" 2>/dev/null; then
    kill -TERM "${MC_PID}" 2>/dev/null || true
    wait "${MC_PID}" 2>/dev/null || true
  fi
  # The official start script owns this router in this validation session.
  if [[ -x "${AIMSIM_MOTION_CONTROL_ROOT}/bin/iox-roudi" ]]; then
    pkill -TERM -f "${AIMSIM_MOTION_CONTROL_ROOT}/bin/iox-roudi" 2>/dev/null || true
  fi
}
trap cleanup TERM INT EXIT

echo "[official-aimsim] starting official motion_control first"
(
  export AGIBOT_ROBOT_MODEL
  export LOG_PATH="${MC_LOG_DIR}"
  exec bash "${MC_SCRIPT}"
) >"${MC_LOG_DIR}/motion_control.stdout.log" 2>&1 &
MC_PID=$!

sleep 4
if ! kill -0 "${MC_PID}" 2>/dev/null; then
  echo "[official-aimsim] official motion_control exited during startup" >&2
  sed -n '1,240p' "${MC_LOG_DIR}/motion_control.stdout.log" >&2 || true
  exit 1
fi

echo "[official-aimsim] starting official AimSim MuJoCo SIL"
"${AIMSIM_PYTHON}" -m aimsim.cli mujoco start \
  --user-config-path "${AIMSIM_USER_CONFIG}"
