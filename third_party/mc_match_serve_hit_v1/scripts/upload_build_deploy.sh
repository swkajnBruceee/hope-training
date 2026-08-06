#!/usr/bin/env bash
set -Eeuo pipefail

# Host-side entry point for the HaLow SSH path. Upload this complete directory
# to the HDU through the MDU, then let the HDU's own compiler and board
# environment build it and deploy the runtime package back to the MDU.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
MDU_ADDR="${MDU_ADDR:-192.168.50.30}"
MDU_USER="${MDU_USER:-agi}"
HDU_ADDR="${HDU_ADDR:-10.42.10.10}"
HDU_USER="${HDU_USER:-agi}"
ROBOT_PASSWORD="${ROBOT_PASSWORD:-1}"
HDU_ROOT="${HDU_ROOT:-/agibot/data/user_deploy/mc}"

command -v sshpass >/dev/null 2>&1 || {
  echo "[mc] sshpass is required for the HaLow upload path" >&2
  exit 2
}

tar -C "$ROOT" -cf - . |
  sshpass -p "$ROBOT_PASSWORD" ssh -o StrictHostKeyChecking=no \
    "$MDU_USER@$MDU_ADDR" \
    "sshpass -p '$ROBOT_PASSWORD' ssh -o StrictHostKeyChecking=no '$HDU_USER@$HDU_ADDR' \
      \"mkdir -p '$HDU_ROOT' && tar -xf - -C '$HDU_ROOT'\""

sshpass -p "$ROBOT_PASSWORD" ssh -o StrictHostKeyChecking=no \
  "$MDU_USER@$MDU_ADDR" \
  "sshpass -p '$ROBOT_PASSWORD' ssh -o StrictHostKeyChecking=no '$HDU_USER@$HDU_ADDR' \
    \"chmod +x '$HDU_ROOT/scripts/build_on_hdu_and_deploy.sh' && \
     '$HDU_ROOT/scripts/build_on_hdu_and_deploy.sh'\""
