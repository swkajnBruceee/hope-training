#pragma once

#include "robot_io/robot_io_backend.hpp"
#include "model3396_config.hpp"

#include <array>

namespace a3_deploy::model3396 {

class SafetyFilter {
 public:
  bool Finite(const std::array<float, kObsDim>& obs,
              const std::array<float, kActionDim>& action) const noexcept;
  // Limit the commanded PD + feed-forward torque and react to measured
  // effort. Returns false when the complete 31-DOF torque state is absent or
  // non-finite; the caller must then refuse to publish the command.
  bool LimitTorque(const robot_io::RobotState& state,
                   robot_io::RobotCommand& command,
                   double limit_ratio = 0.8) const noexcept;
};

}  // namespace a3_deploy::model3396
