#include <gtest/gtest.h>

#include <cmath>

#include "ball_trajectory_predictor.h"
#include "hit_plan_solver.h"

namespace {

trajectory::StrikeTarget makeStrike() {
  trajectory::StrikeTarget strike;
  strike.p_ball = Eigen::Vector3d(0.0, -0.7625, 0.3);
  strike.v_ball = Eigen::Vector3d(-3.0, 0.0, -0.5);
  strike.t_strike = 0.5;
  strike.num_bounces = 1;
  strike.valid = true;
  return strike;
}

solver::SolveTarget makeTarget() {
  solver::SolveTarget target;
  target.target_land = Eigen::Vector3d(2.055, -0.7625, 0.0);
  target.delta_t_flight = 0.5;
  target.desired_ball_speed = -1.0;
  target.max_ball_out_speed = -1.0;
  target.max_racket_speed = 6.0;
  target.net_clearance_margin = 0.03;
  target.valid = true;
  target.mode = "fixed_center";
  return target;
}

solver::HitPlanSolver makeSolver() {
  return solver::HitPlanSolver{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
}

void expectFiniteVector(const Eigen::Vector3d & v) {
  EXPECT_TRUE(std::isfinite(v.x()));
  EXPECT_TRUE(std::isfinite(v.y()));
  EXPECT_TRUE(std::isfinite(v.z()));
}

}  // namespace

TEST(HitPlanSolverTest, FixedTargetSolveProducesFinitePlan) {
  auto solver = makeSolver();
  const auto plan = solver.solve(makeStrike(), makeTarget());

  if (plan.valid) {
    EXPECT_EQ(plan.reason, "ok");
  } else {
    EXPECT_FALSE(plan.reason.empty());
  }
  expectFiniteVector(plan.v_out);
  expectFiniteVector(plan.v_in);
  expectFiniteVector(plan.racket_velocity);
  expectFiniteVector(plan.racket_normal);
  EXPECT_NEAR(plan.racket_normal.norm(), 1.0, 1e-9);
}

TEST(HitPlanSolverTest, InvalidStrikeReportsReason) {
  auto solver = makeSolver();
  auto strike = makeStrike();
  strike.valid = false;

  const auto plan = solver.solve(strike, makeTarget());
  EXPECT_FALSE(plan.valid);
  EXPECT_EQ(plan.reason, "invalid_strike");
}

TEST(HitPlanSolverTest, InvalidTargetReportsReason) {
  auto solver = makeSolver();
  auto target = makeTarget();
  target.valid = false;

  const auto plan = solver.solve(makeStrike(), target);
  EXPECT_FALSE(plan.valid);
  EXPECT_EQ(plan.reason, "invalid_target");
}

TEST(HitPlanSolverTest, NonPositiveFlightTimeReportsReason) {
  auto solver = makeSolver();
  auto target = makeTarget();
  target.delta_t_flight = 0.0;

  const auto plan = solver.solve(makeStrike(), target);
  EXPECT_FALSE(plan.valid);
  EXPECT_EQ(plan.reason, "non_positive_flight_time");
}

TEST(HitPlanSolverTest, RacketSpeedLimitReportsReason) {
  auto solver = makeSolver();
  auto target = makeTarget();
  target.max_racket_speed = 0.01;

  const auto plan = solver.solve(makeStrike(), target);
  EXPECT_FALSE(plan.valid);
  EXPECT_EQ(plan.reason, "racket_speed_limit");
}
