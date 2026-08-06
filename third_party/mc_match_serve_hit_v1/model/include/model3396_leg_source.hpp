#pragma once

#include "a3_deploy/a3_policy_runtime.hpp"
#include "control_source.hpp"
#include "robot_io/robot_io_backend.hpp"
#include "model3396_config.hpp"
#include "model3396_observation_builder.hpp"
#include "model3396_action_decoder.hpp"
#include "model3396_safety_filter.hpp"

#include <array>
#include <memory>

namespace a3_deploy::model3396 {

class Model3396LegSource final {
 public:
  Model3396LegSource(std::unique_ptr<A3PolicyRuntime> policy,
                     double output_gain, double outer_hz);

  bool Update(const robot_io::RobotState& state,
              control::LegTarget& target) noexcept;

 private:
  static constexpr double kLegPolicyHz = 50.0;

  std::unique_ptr<A3PolicyRuntime> policy_;
  double output_gain_{1.0};
  double outer_hz_{50.0};
  double phase_{0.0};
  bool initialized_{false};
  ObservationBuilder observation_builder_;
  ActionDecoder decoder_;
  SafetyFilter safety_;
  std::array<float, kObsDim> observation_{};
  std::array<float, kActionDim> raw_action_{};
  std::array<double, 6> active_left_{};
  std::array<double, 6> active_right_{};
};

}  // namespace a3_deploy::model3396
