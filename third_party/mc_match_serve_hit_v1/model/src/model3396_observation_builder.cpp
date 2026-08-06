#include "model3396_observation_builder.hpp"

#include "a3_policy_parameters.hpp"
#include "math_utils.hpp"
#include "robot_io/a3_layout_extra.hpp"

#include <algorithm>
#include <cmath>

namespace a3_deploy::model3396 {
namespace {
float q(const Eigen::VectorXd& v, int i) noexcept { return i < v.size() ? static_cast<float>(v[i]) : 0.0f; }
void fill3(float* dst, float x, float y, float z) noexcept { dst[0] = x; dst[1] = y; dst[2] = z; }
}

void ObservationBuilder::Build(const robot_io::RobotState& state,
                               std::array<float, kObsDim>& out) noexcept {
  out.fill(0.0f);
  // RobotState currently has no root linear velocity channel. Static profile
  // intentionally uses zero here; this is surfaced in the runtime contract.
  fill3(out.data(), 0.0f, 0.0f, 0.0f);
  out[3] = static_cast<float>(state.imu_gyro[0]);
  out[4] = static_cast<float>(state.imu_gyro[1]);
  out[5] = static_cast<float>(state.imu_gyro[2]);

  const auto g = GetGravityOrientation_d({state.imu_quat_wxyz[0], state.imu_quat_wxyz[1],
                                          state.imu_quat_wxyz[2], state.imu_quat_wxyz[3]});
  const int base_sdk[14] = {19,20,21,22,23,24,25,26,27,28,29,30,1,2};
  const double base_default[14] = {-0.160,0.080,-0.0348,0.320,-0.155,-0.0078,
                                   -0.160,-0.080,0.0348,0.320,-0.155,0.0078,0.0,0.0};
  for (int i = 0; i < 14; ++i) {
    out[6 + i] = q(state.q, base_sdk[i]) - static_cast<float>(base_default[i]);
    out[20 + i] = q(state.dq, base_sdk[i]);
    out[34 + i] = previous_action_[i];
  }
  out[48] = static_cast<float>(g[0]); out[49] = static_cast<float>(g[1]); out[50] = static_cast<float>(g[2]);

  // Static preparation profile. These are finite, physically-shaped stand
  // features, not zeros: position uses the training-distribution mean and
  // normals are valid unit vectors. A future racket-FK provider can replace
  // this block without changing the actor contract.
  out[51] = 2.8535f; out[52] = 0.1144f; out[53] = -0.2790f;
  out[54] = 0.0f; out[55] = 0.0f; out[56] = 0.0f;
  out[57] = 0.0f; out[58] = 0.0f; out[59] = 1.0f;
  out[60] = 0.4093f; out[61] = -0.3225f; out[62] = -0.1682f;
  out[63] = 0.0f; out[64] = 0.0f; out[65] = 0.0f;
  out[66] = 0.0f; out[67] = 0.0f; out[68] = 1.0f;
  out[69] = -0.4912f; out[70] = 0.0f;

  // strike_v2 order: waist_yaw, waist_pitch, right arm seven joints.
  const int strike_sdk[9] = {0, 2, 12, 13, 14, 15, 16, 17, 18};
  const double strike_default[9] = {0.0, 0.0, 0.3, -0.12, 0.0, 0.8, 0.0, 0.0, 0.0};
  for (int i = 0; i < 9; ++i) {
    out[71 + i] = q(state.q, strike_sdk[i]) - static_cast<float>(strike_default[i]);
    out[80 + i] = q(state.dq, strike_sdk[i]);
    out[89 + i] = out[71 + i];
    out[98 + i] = 0.0f;
    out[107 + i] = 0.0f;
    out[116 + i] = 0.0f;
  }
  out[125] = 0.0f;
}

}
