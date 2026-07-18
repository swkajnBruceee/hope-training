#!/usr/bin/env bash
set -euo pipefail

# First ONNX branch gate: state/sync + static-standing inference only.
# --probe hard-disables command publishing in a3_deploy_onnx_ref.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGE_DIR="${ROOT_DIR}/a3_deploy_example/dist/a3_deploy_x86_64"
RUNTIME_CFG="${ROOT_DIR}/experiments/a3_onnx_balance_sil/stand_shadow.yaml"
REPORT_DIR="${ROOT_DIR}/experiments/a3_onnx_balance_sil/artifacts"
DUMP_PATH="/tmp/a3_onnx_balance_stand_shadow.bin"

mkdir -p "${REPORT_DIR}"
rm -f "${DUMP_PATH}"

cd "${PACKAGE_DIR}"
export A3_SOURCE_ROBOT_ENV=0
export A3_TRANSPORT=iceoryx
export LD_LIBRARY_PATH="/workspace/anaconda3/envs/hope_ros/lib:${LD_LIBRARY_PATH:-}"
timeout --signal=INT 12s ./a3_deploy_onnx_ref \
  --runtime-cfg="${RUNTIME_CFG}" \
  --aimrt-cfg="${PACKAGE_DIR}/config/a3_aimrt_config.iceoryx.yaml" \
  --probe --probe-source=a3 --frame-log-interval=100 || status=$?
status=${status:-0}

# timeout intentionally sends SIGINT after enough post-warmup samples to fill
# the 100-record dump.  GNU timeout returns 124 for that expected stop.
if [[ "${status}" -ne 0 && "${status}" -ne 124 ]]; then
  exit "${status}"
fi
python3 "${ROOT_DIR}/experiments/a3_onnx_balance_sil/analyze_stand_shadow.py" \
  "${DUMP_PATH}" \
  --output "${REPORT_DIR}/stand_shadow_report.json"
