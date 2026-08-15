#include "solver_pipeline.h"

namespace solver {

HOPESolverPipeline::HOPESolverPipeline(
  const common::BallPhysics & physics,
  const common::PlannerConfig & config,
  const common::TableParams & table)
: physics_(physics),
  config_(config),
  table_(table),
  estimator_(config_),
  predictor_(physics_, config_, table_),
  solver_(physics_, config_, table_)
{
}

std::optional<HitPlan> HOPESolverPipeline::update(
  double t,
  const Eigen::Vector3d & p_ball,
  const SolveTarget & target)
{
  estimator_.push(t, p_ball);

  if (!estimator_.ready()) {
    latest_plan_ = HitPlan{};
    latest_plan_.reason = "estimator_not_ready";
    return std::nullopt;
  }

  auto est = estimator_.estimate();
  latest_t_ = est.t;

  // Only predict if the ball is moving toward P1 (v_x < 0).
  if (est.v.x() >= 0.0) {
    latest_plan_ = HitPlan{};
    latest_plan_.reason = "ball_not_incoming";
    latest_strike_.reset();
    return std::nullopt;
  }

  trajectory::StrikeTarget strike = predictor_.predict(est.p, est.v, est.t);
  latest_strike_ = strike;

  latest_plan_ = solver_.solve(strike, target);
  return latest_plan_;
}

std::optional<double> HOPESolverPipeline::timeToStrike() const {
  if (!latest_plan_.valid) {
    return std::nullopt;
  }
  if (!latest_strike_.has_value()) {
    return std::nullopt;
  }
  return latest_strike_->t_strike - latest_t_;
}

}  // namespace solver
