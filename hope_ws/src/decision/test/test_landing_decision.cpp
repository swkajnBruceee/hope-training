#include <gtest/gtest.h>

#include "landing_decision.h"

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

decision::LandingDecisionConfig makeDecisionConfig() {
  decision::LandingDecisionConfig cfg;
  cfg.racket_speed_planning_cap = 5.4;
  cfg.max_ball_out_speed = -1.0;
  return cfg;
}

decision::HardConstraintFilter makeFilter() {
  return decision::HardConstraintFilter{
    makeDecisionConfig(), common::BallPhysics(), common::PlannerConfig(), common::TableParams()};
}

decision::LandingCandidate makeCandidate(double x, double y, double dt) {
  decision::LandingCandidate candidate;
  candidate.target_land = Eigen::Vector3d(x, y, 0.0);
  candidate.delta_t_flight = dt;
  candidate.mode = "test";
  return candidate;
}

}  // namespace

TEST(LandingCandidateGeneratorTest, GeneratesFixedLandingAndFlightTimeGrid) {
  decision::LandingCandidateGenerator generator;
  const auto candidates = generator.generate();

  EXPECT_EQ(candidates.size(), 28u);
  EXPECT_EQ(candidates.front().mode, "dynamic_center_mid");
  EXPECT_NEAR(candidates.front().target_land.x(), 1.95, 1e-12);
  EXPECT_NEAR(candidates.front().target_land.y(), -0.7625, 1e-12);
  EXPECT_NEAR(candidates.front().delta_t_flight, 0.45, 1e-12);
}

TEST(HardConstraintFilterTest, RejectsLandingOutsideOpponentTable) {
  auto filter = makeFilter();
  const auto result = filter.evaluate(makeStrike(), 0.30, makeCandidate(1.0, -0.7625, 0.5));

  EXPECT_FALSE(result.hard_valid);
  EXPECT_EQ(result.hard_reason, "outside_opponent_table");
}

TEST(HardConstraintFilterTest, RejectsLateStrike) {
  auto filter = makeFilter();
  const auto result = filter.evaluate(makeStrike(), 0.05, makeCandidate(2.055, -0.7625, 0.5));

  EXPECT_FALSE(result.hard_valid);
  EXPECT_EQ(result.hard_reason, "too_late_to_execute");
}

TEST(HardConstraintFilterTest, AcceptsReachableCenterCandidate) {
  auto filter = makeFilter();
  const auto result = filter.evaluate(makeStrike(), 0.30, makeCandidate(2.055, -0.7625, 0.5));

  EXPECT_TRUE(result.hard_valid) << result.hard_reason;
  EXPECT_EQ(result.hard_reason, "ok");
  EXPECT_TRUE(result.plan.valid);
  EXPECT_TRUE(result.plan.clears_net);
  EXPECT_FALSE(result.plan.bypasses_net_posts);
  EXPECT_LE(result.plan.racket_velocity.norm(), 5.4);
}

TEST(SoftConstraintScorerTest, DoesNotRewardBallSpeedBeingBelowComfortRange) {
  decision::SoftConstraintScorer scorer{
    makeDecisionConfig(), common::BallPhysics(), common::PlannerConfig(), common::TableParams()};

  auto slow = makeCandidate(2.055, -0.7625, 0.5);
  slow.hard_valid = true;
  slow.plan.v_out = Eigen::Vector3d(1.0, 0.0, 0.0);
  slow.plan.racket_velocity = Eigen::Vector3d(2.0, 0.0, 0.0);
  slow.plan.p_hit = Eigen::Vector3d(0.0, -0.7625, 0.3);

  auto comfort = slow;
  comfort.plan.v_out = Eigen::Vector3d(6.0, 0.0, 1.0);

  const auto slow_scored = scorer.score(slow);
  const auto comfort_scored = scorer.score(comfort);

  EXPECT_LT(slow_scored.ball_speed_score, comfort_scored.ball_speed_score);
}

TEST(LandingDecisionPlannerTest, SelectsValidDynamicTargetForNormalStrike) {
  decision::LandingDecisionPlanner planner{
    makeDecisionConfig(), common::BallPhysics(), common::PlannerConfig(), common::TableParams()};

  const auto result = planner.select(makeStrike(), 0.30);

  EXPECT_TRUE(result.target.valid);
  EXPECT_FALSE(result.target.mode.empty());
  EXPECT_NE(result.target.mode, "fixed_center");
  EXPECT_GT(result.candidate_count, 0);
  EXPECT_GT(result.hard_valid_count, 0);
  EXPECT_GE(result.selected.total_score, 0.0);
  EXPECT_LE(result.target.max_racket_speed, 5.4);
  EXPECT_LT(result.target.max_ball_out_speed, 0.0);
}

TEST(LandingDecisionPlannerTest, ReportsNoFeasibleLandingWhenStrikeIsTooLate) {
  decision::LandingDecisionPlanner planner{
    makeDecisionConfig(), common::BallPhysics(), common::PlannerConfig(), common::TableParams()};

  const auto result = planner.select(makeStrike(), 0.05);

  EXPECT_FALSE(result.target.valid);
  EXPECT_EQ(result.target.mode, "no_feasible_landing");
  EXPECT_GT(result.reject_reasons.at("too_late_to_execute"), 0);
}
