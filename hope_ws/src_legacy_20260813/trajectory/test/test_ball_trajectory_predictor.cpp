#include <gtest/gtest.h>

#include "ball_trajectory_predictor.h"

TEST(BallTrajectoryPredictorTest, IncomingTrajectoryCrossesHitPlane) {
  trajectory::BallTrajectoryPredictor pred{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
  Eigen::Vector3d p0(0.5, -0.7625, 0.5);
  Eigen::Vector3d v0(-4.0, 0.0, 2.0);
  auto strike = pred.predict(p0, v0, 0.0);
  EXPECT_TRUE(strike.valid);
  EXPECT_NEAR(strike.p_ball.x(), 0.0, 1e-6);
  EXPECT_EQ(strike.num_bounces, 0);
  EXPECT_GT(strike.t_strike, 0.05);
  EXPECT_LT(strike.t_strike, 0.4);
}

TEST(BallTrajectoryPredictorTest, BallMovingAwayProducesNoValidCommand) {
  trajectory::BallTrajectoryPredictor pred{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
  Eigen::Vector3d p0(0.5, -0.7625, 0.5);
  Eigen::Vector3d v0(3.0, 0.0, 1.0);
  auto strike = pred.predict(p0, v0, 0.0);
  EXPECT_FALSE(strike.valid);
}

TEST(BallTrajectoryPredictorTest, TableBounceReversesZVelocity) {
  trajectory::BallTrajectoryPredictor pred{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
  Eigen::Vector3d v_minus(2.0, 0.0, -3.0);
  Eigen::Vector3d v_plus = pred.applyBounce(v_minus);
  EXPECT_GT(v_plus.z(), 0.0);
  EXPECT_NEAR(v_plus.z(), common::BallPhysics().C_v * 3.0, 1e-9);
  EXPECT_NEAR(v_plus.x(), common::BallPhysics().C_h * 2.0, 1e-9);
}

TEST(BallTrajectoryPredictorTest, BounceThenCrossesHitPlane) {
  trajectory::BallTrajectoryPredictor pred{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
  Eigen::Vector3d p0(0.5, -0.7625, 0.2);
  Eigen::Vector3d v0(-2.0, 0.0, -2.0);
  auto strike = pred.predict(p0, v0, 0.0);
  EXPECT_TRUE(strike.valid);
  EXPECT_EQ(strike.num_bounces, 1);
  EXPECT_NEAR(strike.p_ball.x(), 0.0, 1e-6);
  EXPECT_LT(strike.v_ball.x(), 0.0);
  EXPECT_GT(strike.p_ball.z(), 0.0);
}

TEST(BallTrajectoryPredictorTest, BounceOutsideTableBoundsNotValid) {
  trajectory::BallTrajectoryPredictor pred{
    common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
  Eigen::Vector3d on_table(1.0, -0.7625, -0.01);
  Eigen::Vector3d off_table(3.0, -0.7625, -0.01);
  EXPECT_TRUE(pred.isOnTable(on_table));
  EXPECT_FALSE(pred.isOnTable(off_table));
}
