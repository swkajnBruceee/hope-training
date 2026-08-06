#pragma once

#include "robot_io/robot_io_backend.hpp"
#include "model3396_config.hpp"

#include <array>

namespace a3_deploy::model3396 {

class ObservationBuilder {
 public:
  void Build(const robot_io::RobotState& state,
             std::array<float, kObsDim>& out) noexcept;
  const std::array<float, kActionDim>& PreviousAction() const noexcept { return previous_action_; }
  void SetPreviousAction(const std::array<float, kActionDim>& action) noexcept { previous_action_ = action; }

 private:
  std::array<float, kActionDim> previous_action_{};
};

}  // namespace a3_deploy::model3396
