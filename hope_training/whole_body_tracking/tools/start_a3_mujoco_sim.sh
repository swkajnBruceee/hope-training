#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "${SCRIPT_DIR}/setup_a3_mujoco_sim_env.sh"
exec "${HOPE_A3_MUJOCO_SIM_BIN}/start_a3_pingpong_iceoryx.sh"
