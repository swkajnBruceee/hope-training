#include "model3396_safety_filter.hpp"
#include "robot_io/a3_layout_extra.hpp"

#include <cmath>
#include <algorithm>
#include <array>

namespace a3_deploy::model3396 {
namespace {
// A3 T2D5 actuator limits from the packaged a3_t2d5.xml, in the 31-slot
// backend layout. Using the older T2D0 waist-pitch limit here caused gravity
// effort to pin q_des to the measured angle during shutdown.
constexpr std::array<double, 31> kMaxTorqueNm = {
    220.0, 46.0, 118.0, 6.0, 6.0,
    60.0, 60.0, 24.0, 24.0, 24.0, 6.0, 6.0,
    60.0, 60.0, 24.0, 24.0, 24.0, 6.0, 6.0,
    220.0, 220.0, 220.0, 320.0, 118.2, 54.75,
    220.0, 220.0, 220.0, 320.0, 118.2, 54.75};
}

bool SafetyFilter::Finite(const std::array<float, kObsDim>& obs,
                          const std::array<float, kActionDim>& action) const noexcept {
  for (float v : obs) if (!std::isfinite(v)) return false;
  for (float v : action) if (!std::isfinite(v)) return false;
  return true;
}
bool SafetyFilter::LimitTorque(const robot_io::RobotState& state,
                               robot_io::RobotCommand& command,
                               double limit_ratio) const noexcept {
  if (state.q.size() != 31 || state.dq.size() != 31 || state.tau_est.size() != 31) return false;
  if (!std::isfinite(limit_ratio) || limit_ratio <= 0.0 || limit_ratio > 0.8) return false;
  for (int i = 0; i < 31; ++i) {
    if (!std::isfinite(state.q[i]) || !std::isfinite(state.dq[i]) ||
        !std::isfinite(state.tau_est[i]) || !std::isfinite(command.q_des[i]) ||
        !std::isfinite(command.dq_des[i]) || !std::isfinite(command.tau_ff[i]) ||
        !std::isfinite(command.kp[i]) || !std::isfinite(command.kd[i])) return false;

    // The upper-body trajectory is an explicitly commanded position/velocity
    // reference. Do not reshape its command from the generic torque envelope:
    // the wrist fast-motion clips can legitimately require more feed-forward
    // effort than the conservative stand envelope. Keep finite-value checks
    // above, while retaining torque limiting for waist and legs below.
    if (i >= robot_io::kA3ArmStart &&
        i < robot_io::kA3ArmStart + robot_io::kA3ArmCount) {
      continue;
    }

    const double cap = limit_ratio * kMaxTorqueNm[i];
    // If measured effort is already at the cap, remove the position error
    // and feed-forward contribution before the next publish.
    if (std::abs(state.tau_est[i]) >= cap) {
      command.tau_ff[i] = 0.0;
      command.dq_des[i] = 0.0;
      command.q_des[i] = state.q[i] + command.kd[i] * state.dq[i] /
          std::max(command.kp[i], 1e-9);
      continue;
    }

    const double velocity_term = command.kd[i] * (command.dq_des[i] - state.dq[i]);
    const double requested = command.kp[i] * (command.q_des[i] - state.q[i]) +
        velocity_term + command.tau_ff[i];
    const double limited = std::clamp(requested, -cap, cap);
    if (command.kp[i] > 1e-9) {
      command.q_des[i] = state.q[i] + (limited - velocity_term - command.tau_ff[i]) /
          command.kp[i];
    } else {
      command.tau_ff[i] = std::clamp(command.tau_ff[i], -cap, cap);
    }
  }
  return true;
}
}
