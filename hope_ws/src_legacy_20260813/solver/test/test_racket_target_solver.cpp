#include <gtest/gtest.h>

#include <cmath>

#include "racket_target_solver.h"
#include "ball_trajectory_predictor.h"

namespace {

trajectory::StrikeTarget makeIncoming() {
  trajectory::StrikeTarget s;
  s.p_ball = Eigen::Vector3d(0.0, -0.7625, 0.3);
  s.v_ball = Eigen::Vector3d(-3.0, 0.0, -0.5);
  s.t_strike = 0.4;
  s.num_bounces = 1;
  s.valid = true;
  return s;
}

}  // namespace

TEST(RacketTargetSolverTest, NormalIncomingBallProducesValidCommand) {
  solver::RacketTargetSolver s{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
  auto cmd = s.plan(makeIncoming());
  EXPECT_TRUE(cmd.valid);
  EXPECT_EQ(cmd.num_bounces, 1);
}

TEST(RacketTargetSolverTest, NormalVectorIsUnitLength) {
  solver::RacketTargetSolver s{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
  auto cmd = s.plan(makeIncoming());
  EXPECT_NEAR(cmd.n_racket.norm(), 1.0, 1e-9);
}

TEST(RacketTargetSolverTest, RacketNormalFacesOpponentSide) {
  solver::RacketTargetSolver s{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
  auto cmd = s.plan(makeIncoming());
  EXPECT_GT(cmd.n_racket.x(), 0.0);
}

TEST(RacketTargetSolverTest, OutgoingVelocityLandsUnderDragModel) {
  solver::RacketTargetSolver s{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
  common::PlannerConfig cfg;
  auto strike = makeIncoming();
  Eigen::Vector3d v_out = s.computeOutgoingVelocity(strike.p_ball, cfg.target_land, cfg.delta_t_flight);
  auto [p_end, v_end] = s.integrateFlight(strike.p_ball, v_out, cfg.delta_t_flight);
  EXPECT_NEAR((p_end - cfg.target_land).norm(), 0.0, 2e-3);
}

TEST(RacketTargetSolverTest, NetClearanceReportedCorrectly) {
  solver::RacketTargetSolver s{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
  Eigen::Vector3d p_strike(0.0, -0.7625, 0.3);
  auto [clears_hi, bypass_hi] = s.checkNetClearance(p_strike, Eigen::Vector3d(5.0, 0.0, 2.0));
  auto [clears_lo, bypass_lo] = s.checkNetClearance(p_strike, Eigen::Vector3d(5.0, 0.0, -2.0));
  EXPECT_TRUE(clears_hi);
  EXPECT_FALSE(bypass_hi);
  EXPECT_FALSE(clears_lo);
}

TEST(RacketTargetSolverTest, InvalidStrikeProducesValidFalse) {
  solver::RacketTargetSolver s{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
  trajectory::StrikeTarget strike;
  strike.p_ball = Eigen::Vector3d(0.0, -0.7625, 0.3);
  strike.v_ball = Eigen::Vector3d(1.0, 0.0, 0.0);
  strike.t_strike = 0.0;
  strike.num_bounces = 0;
  strike.valid = false;
  auto cmd = s.plan(strike);
  EXPECT_FALSE(cmd.valid);
}
