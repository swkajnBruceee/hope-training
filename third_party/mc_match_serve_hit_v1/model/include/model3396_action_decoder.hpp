#pragma once

#include "model3396_config.hpp"

#include <array>

namespace a3_deploy::model3396 {

class ActionDecoder {
 public:
  void Decode(const std::array<float, kActionDim>& raw, double gain,
              std::array<double, 6>& left, std::array<double, 6>& right) noexcept;
  const std::array<float, kActionDim>& PreviousAction() const noexcept { return previous_; }

 private:
  std::array<float, kActionDim> previous_{};
};

}  // namespace a3_deploy::model3396
