#include "target_selector.h"

namespace decision {

TargetSelector::TargetSelector() = default;

TargetDecisionData TargetSelector::selectDefault() const {
  TargetDecisionData out;
  out.target_land = Eigen::Vector3d(2.055, -0.7625, 0.0);
  out.delta_t_flight = 0.5;
  out.desired_ball_speed = -1.0;
  out.max_ball_out_speed = -1.0;
  out.max_racket_speed = 6.0;
  out.net_clearance_margin = 0.03;
  out.valid = true;
  out.mode = "fixed_center";
  return out;
}

}  // namespace decision
