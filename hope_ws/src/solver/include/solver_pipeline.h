#ifndef SOLVER_SOLVER_PIPELINE_H
#define SOLVER_SOLVER_PIPELINE_H

#include <optional>

#include <Eigen/Dense>

#include "constants.h"
#include "hit_plan.h"
#include "hit_plan_solver.h"
#include "ball_state_estimator.h"
#include "ball_trajectory_predictor.h"

namespace solver {

/**
 * Top-level solver pipeline combining Stages 1-3.
 *
 * Mirrors the original Python HOPESolverPipeline:
 *   - BallStateEstimator (Stage 1)
 *   - BallTrajectoryPredictor (Stage 2)
 *   - HitPlanSolver (Stage 3 adapter around RacketTargetSolver)
 */
class HOPESolverPipeline {
 public:
  HOPESolverPipeline(
    const common::BallPhysics & physics = common::BallPhysics(),
    const common::PlannerConfig & config = common::PlannerConfig(),
    const common::TableParams & table = common::TableParams());

  /// Process a new ball position measurement. Returns std::nullopt until a strike can be evaluated.
  std::optional<HitPlan> update(
    double t,
    const Eigen::Vector3d & p_ball,
    const SolveTarget & target);

  /// Latest computed hit plan (may be invalid).
  const HitPlan & hitPlan() const { return latest_plan_; }

  /// Time-to-strike (seconds remaining) of the latest valid command, or std::nullopt.
  std::optional<double> timeToStrike() const;

  /// Latest valid StrikeTarget (for diagnostics/tests).
  const std::optional<trajectory::StrikeTarget> & latestStrike() const { return latest_strike_; }

  /// Latest sample timestamp.
  double latestSampleTime() const { return latest_t_; }

 private:
  common::BallPhysics physics_;
  common::PlannerConfig config_;
  common::TableParams table_;
  trajectory::BallStateEstimator estimator_;
  trajectory::BallTrajectoryPredictor predictor_;
  HitPlanSolver solver_;
  HitPlan latest_plan_;
  std::optional<trajectory::StrikeTarget> latest_strike_;
  double latest_t_ = 0.0;
};

}  // namespace solver

#endif  // SOLVER_SOLVER_PIPELINE_H
