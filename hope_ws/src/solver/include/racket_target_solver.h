#ifndef SOLVER_RACKET_TARGET_SOLVER_H
#define SOLVER_RACKET_TARGET_SOLVER_H

#include <Eigen/Dense>

#include "constants.h"

namespace trajectory {
struct StrikeTarget;
}

namespace solver {

/**
 * Output of Stage 3: desired racket state at strike time.
 */
struct RacketCommand {
  Eigen::Vector3d p_intercept;     // desired racket center position at interception
  Eigen::Vector3d v_racket;        // desired racket velocity vector [vx, vy, vz]
  Eigen::Vector3d n_racket;        // desired racket face normal (unit vector)
  double t_strike = 0.0;           // predicted time of strike
  Eigen::Vector3d v_ball_outgoing; // expected outgoing ball velocity
  Eigen::Vector3d target_land;     // intended landing point
  bool clears_net = false;         // True if return trajectory clears the net
  bool bypasses_net_posts = false; // True if ball passes outside net Y extent
  bool valid = false;              // True if all computations succeeded
  int num_bounces = 0;             // bounces predicted before the strike (from Stage 2)
};

/**
 * Stage 3 - Racket target solver.
 *
 * Given the predicted ball state at the hitting plane (Stage 2), compute the
 * desired racket velocity and face orientation to return the ball to the
 * opponent's half center, with a net-clearance check and flight-time fallback.
 *
 * See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 5.
 */
class RacketTargetSolver {
 public:
  RacketTargetSolver(
    const common::BallPhysics & physics,
    const common::PlannerConfig & config,
    const common::TableParams & table);

  /// Compute desired racket state for a valid return.
  RacketCommand plan(const trajectory::StrikeTarget & strike);

  /// Solve post-strike velocity so the drag model lands near p_land.
  Eigen::Vector3d computeOutgoingVelocity(
    const Eigen::Vector3d & p_strike,
    const Eigen::Vector3d & p_land,
    double delta_t) const;

  /// Integrate free flight with the same drag model used by Stage 2.
  std::pair<Eigen::Vector3d, Eigen::Vector3d> integrateFlight(
    const Eigen::Vector3d & p0,
    const Eigen::Vector3d & v0,
    double duration) const;

  /// Return the interpolated flight position where x crosses x_target.
  bool positionAtX(
    const Eigen::Vector3d & p0,
    const Eigen::Vector3d & v0,
    double x_target,
    double max_time,
    Eigen::Vector3d & out) const;

  /// Orient a racket normal toward the opponent side (+x).
  Eigen::Vector3d faceOpponent(const Eigen::Vector3d & n) const;

  /// Compute desired racket velocity and face normal from impact model.
  std::pair<Eigen::Vector3d, Eigen::Vector3d> computeRacketVelocity(
    const Eigen::Vector3d & v_incoming,
    const Eigen::Vector3d & v_outgoing,
    double C_r) const;

  /// Check height clearance and Y-axis net extent. Returns (clears_net, bypasses_posts).
  std::pair<bool, bool> checkNetClearance(
    const Eigen::Vector3d & p_strike,
    const Eigen::Vector3d & v_outgoing,
    double margin = 0.03) const;

 private:
  Eigen::Vector3d flightAcceleration(const Eigen::Vector3d & v) const;

  common::BallPhysics physics_;
  common::PlannerConfig config_;
  common::TableParams table_;
};

}  // namespace solver

#endif  // SOLVER_RACKET_TARGET_SOLVER_H
