#include "model3396_action_decoder.hpp"

#include <algorithm>
#include <cmath>

namespace a3_deploy::model3396 {
void ActionDecoder::Decode(const std::array<float, kActionDim>& raw, double gain,
                           std::array<double, 6>& left, std::array<double, 6>& right) noexcept {
  for (std::size_t i = 0; i < kActionDim; ++i) {
    const double masked = static_cast<double>(raw[i]) * kActionMask[i];
    const double bounded = kRawClip * std::tanh(masked / kRawClip);
    previous_[i] = static_cast<float>(bounded);
    const double residual = bounded * kActionScale[i] * gain;
    if (i < 6) left[i] = kBaseDefault[i] + residual;
    else if (i < 12) right[i - 6] = kBaseDefault[i] + residual;
  }
  previous_[12] = 0.0f; previous_[13] = 0.0f;
}
}
