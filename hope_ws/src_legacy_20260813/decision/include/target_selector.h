#ifndef DECISION_TARGET_SELECTOR_H
#define DECISION_TARGET_SELECTOR_H

#include <Eigen/Dense>
#include <string>

namespace decision {

/**
 * Fixed-strategy target decision for the first version.
 */
struct TargetDecisionData {
  Eigen::Vector3d target_land;
  double delta_t_flight = 0.0;
  double desired_ball_speed = -1.0;
  double max_ball_out_speed = -1.0;
  double max_racket_speed = 6.0;
  double net_clearance_margin = 0.03;
  bool valid = false;
  std::string mode;
};

/**
 * Placeholder target selector. Future versions will replace the fixed center
 * strategy with stateful target selection.
 */
class TargetSelector {
 public:
  TargetSelector();

  /// Return the default decision (fixed P2 half-center, 0.5 s flight time).
  TargetDecisionData selectDefault() const;
};

}  // namespace decision

#endif  // DECISION_TARGET_SELECTOR_H
