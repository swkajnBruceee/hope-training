#ifndef TRAJECTORY_BALL_TRAJECTORY_PREDICTOR_H
#define TRAJECTORY_BALL_TRAJECTORY_PREDICTOR_H

#include <Eigen/Dense>
#include <vector>

#include "constants.h"

namespace trajectory {

/**
 * Output of Stage 2: predicted ball state at the hitting plane.
 */
struct StrikeTarget {
  Eigen::Vector3d p_ball;        // predicted ball position at strike
  Eigen::Vector3d v_ball;        // predicted ball velocity at strike
  double t_strike = 0.0;
  int num_bounces = 0;
  bool valid = false;
};

/**
 * Stage 2 - Ball trajectory prediction.
 *
 * Forward-integrate the ball trajectory with explicit Euler at 1 kHz using a
 * hybrid flight (quadratic drag + gravity) / bounce (diagonal restitution)
 * model, and return the predicted ball state at the virtual hitting plane.
 *
 * See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 4.
 */
class BallTrajectoryPredictor {
 public:
  BallTrajectoryPredictor(
    const common::BallPhysics & physics,
    const common::PlannerConfig & config,
    const common::TableParams & table);

  /// Forward-integrate and find the hitting-plane crossing.
  StrikeTarget predict(const Eigen::Vector3d & p0, const Eigen::Vector3d & v0, double t0);

  /// Forward-integrate and return sampled future ball positions for visualization.
  std::vector<Eigen::Vector3d> sampleFuture(
    const Eigen::Vector3d & p0,
    const Eigen::Vector3d & v0,
    double horizon_s,
    int sample_stride) const;

  /// Apply table bounce restitution: v+ = diag(C_h, C_h, -C_v) @ v-.
  Eigen::Vector3d applyBounce(const Eigen::Vector3d & v) const;

  /// Sample the *incoming* (pre-bounce) trajectory until the first
  /// table-surface contact on the P1 half.
  ///
  /// Stops as soon as the ball center crosses z = physics_.radius while
  /// descending; if the contact x/y falls inside the P1 half
  /// (``[0, net_x] x [-table_width, 0]``), the contact point is appended
  /// and integration stops.  If the contact falls outside P1, the
  /// integration also stops -- the caller decides whether to send the
  /// packet.
  ///
  /// This function NEVER simulates a bounce; it exists so the pre-bounce
  /// overlay draws a single incoming arc ending at the predicted contact,
  /// instead of bouncing like sampleFuture().
  std::vector<Eigen::Vector3d> sampleIncomingUntilFirstP1Bounce(
    const Eigen::Vector3d & p0,
    const Eigen::Vector3d & v0,
    double horizon_s,
    int sample_stride) const;

  /// Check if ball could contact the table surface (expanded by ball radius).
  bool isOnTable(const Eigen::Vector3d & p) const;

 private:
  Eigen::Vector3d flightAcceleration(const Eigen::Vector3d & v) const;

  common::BallPhysics physics_;
  common::PlannerConfig config_;
  common::TableParams table_;
};

}  // namespace trajectory

#endif  // TRAJECTORY_BALL_TRAJECTORY_PREDICTOR_H
