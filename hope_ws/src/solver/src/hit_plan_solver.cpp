#include "hit_plan_solver.h"

#include <cmath>

#include <Eigen/Geometry>

#include "ball_trajectory_predictor.h"

namespace solver {

namespace {

bool finiteVector(const Eigen::Vector3d & v) {
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

}  // namespace

HitPlanSolver::HitPlanSolver(
  const common::BallPhysics & physics,
  const common::PlannerConfig & config,
  const common::TableParams & table)
: physics_(physics),
  config_(config),
  table_(table),
  racket_solver_(physics_, config_, table_)
{
}

HitPlan HitPlanSolver::solve(
  const trajectory::StrikeTarget & strike,
  const SolveTarget & target) const
{
  HitPlan plan;
  plan.p_hit = strike.p_ball;
  plan.t_hit = strike.t_strike;
  plan.v_in = strike.v_ball;
  plan.target_land = target.target_land;
  plan.flight_time = target.delta_t_flight;

  if (!strike.valid) {
    plan.reason = "invalid_strike";
    return plan;
  }
  if (!target.valid) {
    plan.reason = "invalid_target";
    return plan;
  }
  if (target.delta_t_flight <= 0.0) {
    plan.reason = "non_positive_flight_time";
    return plan;
  }

  plan.v_out = racket_solver_.computeOutgoingVelocity(
    strike.p_ball, target.target_land, target.delta_t_flight);
  auto [racket_velocity, racket_normal] =
    racket_solver_.computeRacketVelocity(strike.v_ball, plan.v_out, config_.C_r);
  plan.racket_velocity = racket_velocity;
  plan.racket_normal = racket_normal;
  plan.racket_orientation = normalToQuaternion(racket_normal);

  auto [clears_net, bypasses_net_posts] =
    racket_solver_.checkNetClearance(
      strike.p_ball, plan.v_out, target.net_clearance_margin);
  plan.clears_net = clears_net;
  plan.bypasses_net_posts = bypasses_net_posts;

  if (!finiteVector(plan.v_out) ||
      !finiteVector(plan.racket_velocity) ||
      !finiteVector(plan.racket_normal)) {
    plan.reason = "non_finite_solution";
    return plan;
  }
  if (!plan.clears_net) {
    plan.reason = "net_not_clear";
    return plan;
  }
  if (target.max_ball_out_speed > 0.0 && plan.v_out.norm() > target.max_ball_out_speed) {
    plan.reason = "ball_speed_limit";
    return plan;
  }
  if (target.max_racket_speed > 0.0 && plan.racket_velocity.norm() > target.max_racket_speed) {
    plan.reason = "racket_speed_limit";
    return plan;
  }

  plan.valid = true;
  plan.reason = "ok";
  return plan;
}

geometry_msgs::msg::Quaternion HitPlanSolver::normalToQuaternion(const Eigen::Vector3d & normal) const {
  Eigen::Vector3d n = normal;
  if (n.norm() < 1e-9) {
    n = Eigen::Vector3d::UnitX();
  } else {
    n.normalize();
  }

  const Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitX(), n).normalized();
  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}

}  // namespace solver
