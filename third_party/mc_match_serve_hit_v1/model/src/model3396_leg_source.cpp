#include "model3396_leg_source.hpp"

#include <algorithm>

namespace a3_deploy::model3396 {

Model3396LegSource::Model3396LegSource(
    std::unique_ptr<A3PolicyRuntime> policy, double output_gain,
    double outer_hz)
    : policy_(std::move(policy)),
      output_gain_(output_gain > 0.0 ? output_gain : 1.0),
      outer_hz_(outer_hz > 0.0 ? outer_hz : kLegPolicyHz),
      active_left_{kBaseDefault[0], kBaseDefault[1], kBaseDefault[2],
                   kBaseDefault[3], kBaseDefault[4], kBaseDefault[5]},
      active_right_{kBaseDefault[6], kBaseDefault[7], kBaseDefault[8],
                    kBaseDefault[9], kBaseDefault[10], kBaseDefault[11]} {}

bool Model3396LegSource::Update(const robot_io::RobotState& state,
                                control::LegTarget& target) noexcept {
  if (!policy_ || state.q.size() < 31 || state.dq.size() < 31) return false;

  bool update_policy = !initialized_;
  if (!update_policy) {
    phase_ += kLegPolicyHz / outer_hz_;
    if (phase_ >= 1.0) {
      phase_ -= 1.0;
      update_policy = true;
    }
  }

  if (update_policy) {
    observation_builder_.Build(state, observation_);
    if (!safety_.Finite(observation_, raw_action_)) return false;
    std::copy(observation_.begin(), observation_.end(),
              policy_->MutableInputData());
    if (!policy_->Infer()) return false;
    for (std::size_t i = 0; i < kActionDim; ++i) {
      raw_action_[i] = policy_->ActionData()[i];
    }
    if (!safety_.Finite(observation_, raw_action_)) return false;

    std::array<double, 6> left{};
    std::array<double, 6> right{};
    decoder_.Decode(raw_action_, output_gain_, left, right);
    // The deployment experiment intentionally removes the per-policy-tick
    // 0.08 rad target-change rejection.  Finite-value, torque, and startup
    // protections remain active in the surrounding control path.
    active_left_ = left;
    active_right_ = right;
    observation_builder_.SetPreviousAction(decoder_.PreviousAction());
    initialized_ = true;
  }

  for (int i = 0; i < 6; ++i) {
    target.q[i] = active_left_[i];
    target.q[i + 6] = active_right_[i];
  }
  return true;
}

}  // namespace a3_deploy::model3396
