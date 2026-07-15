#!/usr/bin/env bash
set -euo pipefail

# Report whether the checked-in official deployment example can be built and
# launched on this machine. Do not download proprietary robot artifacts or
# fabricate a placeholder policy model.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DEPLOY_ROOT="${PROJECT_ROOT}/agibot/code_deployment/a3_deploy_example"

pass_count=0
warn_count=0
fail_count=0
pass_msg() { printf '[PASS] %s\n' "$*"; pass_count=$((pass_count + 1)); }
warn_msg() { printf '[WARN] %s\n' "$*"; warn_count=$((warn_count + 1)); }
fail_msg() { printf '[FAIL] %s\n' "$*"; fail_count=$((fail_count + 1)); }

printf '%s\n' "A3 native deploy prerequisite check" \
  "project: ${PROJECT_ROOT}" "deploy:  ${DEPLOY_ROOT}" ""

if [[ -f "${DEPLOY_ROOT}/src/a3/a3_deploy_onnx_ref/include/a3_policy_parameters.hpp" ]]; then
  pass_msg "official A3 policy parameters and PD_STAND gains are present"
else
  fail_msg "official A3 policy parameter header is missing"
fi

if [[ -f "${DEPLOY_ROOT}/mujoco_sim_standalone/bin/start_mujoco_sim.sh" ]]; then
  pass_msg "official standalone simulator executable wrapper is present"
else
  warn_msg "official standalone simulator binary is not present in the checked-in package"
fi

onnx_header=""
onnx_library=""
while IFS= read -r path; do
  [[ -z "${onnx_header}" ]] && onnx_header="${path}"
done < <(find /usr /opt /workspace/anaconda3/envs "${DEPLOY_ROOT}" -type f \
  -name onnxruntime_cxx_api.h 2>/dev/null | head -1)
while IFS= read -r path; do
  [[ -z "${onnx_library}" ]] && onnx_library="${path}"
done < <(find /usr /opt /workspace/anaconda3/envs "${DEPLOY_ROOT}" -type f \
  -name 'libonnxruntime.so*' 2>/dev/null | head -1)

if [[ -n "${onnx_header}" ]]; then
  pass_msg "x86 ONNX Runtime header: ${onnx_header}"
else
  fail_msg "x86 ONNX Runtime C++ header not found"
fi
if [[ -n "${onnx_library}" ]]; then
  pass_msg "x86 ONNX Runtime library: ${onnx_library}"
else
  fail_msg "x86 ONNX Runtime shared library not found"
fi

model_count="$(find "${DEPLOY_ROOT}" -type f \( -name '*.onnx' -o -name '*.rknn' \) 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${model_count}" -gt 0 ]]; then
  pass_msg "deployment policy artifacts found: ${model_count}"
else
  fail_msg "no deployment policy model (*.onnx/*.rknn) found in the checked-in A3 package"
fi

if [[ -n "${ROS_DISTRO:-}" ]]; then
  pass_msg "ROS_DISTRO is set to ${ROS_DISTRO}"
else
  warn_msg "ROS_DISTRO is not set in this shell; source the project ROS environment first"
fi

printf '\nSummary: pass=%d warn=%d fail=%d\n' "${pass_count}" "${warn_count}" "${fail_count}"
if [[ "${fail_count}" -gt 0 ]]; then
  printf '%s\n' "Native A3 MOTION deployment is not locally runnable yet." \
    "The project-local body-drive simulator remains available for actuator-contract validation."
  exit 2
fi
