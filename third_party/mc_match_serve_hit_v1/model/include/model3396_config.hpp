#pragma once

#include <array>
#include <cstddef>

namespace a3_deploy::model3396 {

inline constexpr std::size_t kObsDim = 126;
inline constexpr std::size_t kActionDim = 14;
inline constexpr double kPolicyHz = 50.0;
inline constexpr double kRawClip = 0.25;
inline constexpr std::array<float, kActionDim> kActionMask = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
inline constexpr std::array<double, 14> kActionScale = {
    0.03666666666666667, 0.1375, 0.18333333333333332, 0.04,
    0.0591, 0.027375, 0.03666666666666667, 0.1375,
    0.18333333333333332, 0.04, 0.0591, 0.027375, 0.023, 0.059};
inline constexpr std::array<double, 14> kBaseDefault = {
    -0.1600, 0.0800, -0.0348, 0.3200, -0.1550, -0.0078,
    -0.1600, -0.0800, 0.0348, 0.3200, -0.1550, 0.0078, 0.0, 0.0};
inline constexpr std::array<int, 6> kLeftLegBackend = {19, 20, 21, 22, 23, 24};
inline constexpr std::array<int, 6> kRightLegBackend = {25, 26, 27, 28, 29, 30};


}  // namespace a3_deploy::model3396
