#ifndef TRAJECTORY_BALL_STATE_ESTIMATOR_H
#define TRAJECTORY_BALL_STATE_ESTIMATOR_H

#include <Eigen/Dense>
#include <vector>

#include "constants.h"

namespace trajectory {

/**
 * Stage 1: Ball state estimation.
 *
 * Fits a 2nd-order polynomial to the most recent N position samples and
 * differentiates analytically to obtain a smoothed position and velocity.
 * Callers should reset the buffer when their higher-level state machine
 * detects a table bounce so the polynomial never fits across the velocity
 * discontinuity.
 *
 * See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 3.
 */
class BallStateEstimator {
 public:
  explicit BallStateEstimator(const common::PlannerConfig & config);

  /// Add a new position measurement.
  void push(double t, const Eigen::Vector3d & p);

  /// Clear the estimation buffer (call on bounce detection).
  void reset();

  /// True if enough samples exist for a reliable fit (>= 6 samples).
  bool ready() const;

  /// Reserved for compatibility with older callers. Bounce detection is owned
  /// by the trajectory state machines, not this estimator.
  bool bounceDetected() const;

  /// Compute smoothed ball position and velocity at the latest timestamp.
  /// Throws std::runtime_error if not ready.
  struct Estimate {
    Eigen::Vector3d p;
    Eigen::Vector3d v;
    double t;
  };
  Estimate estimate() const;

 private:
  common::PlannerConfig config_;
  std::vector<double> t_buffer_;
  std::vector<Eigen::Vector3d> p_buffer_;
  std::vector<double> z_hist_;  // 3-sample ring buffer, NaN=empty
  bool bounce_detected_ = false;
};

}  // namespace trajectory

#endif  // TRAJECTORY_BALL_STATE_ESTIMATOR_H
