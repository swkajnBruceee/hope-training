#include <gtest/gtest.h>

#include <cmath>

#include "ball_state_estimator.h"

namespace {
constexpr double kDt = 1.0 / 360.0;
}

TEST(BallStateEstimatorTest, ConstantVelocityReturnsCorrectVelocity) {
  common::PlannerConfig cfg;
  trajectory::BallStateEstimator est(cfg);
  Eigen::Vector3d v_true(1.0, -0.5, 0.0);
  Eigen::Vector3d p0(0.2, -0.3, 0.5);
  for (int i = 0; i < 20; ++i) {
    est.push(i * kDt, p0 + v_true * (i * kDt));
  }
  ASSERT_TRUE(est.ready());
  auto e = est.estimate();
  EXPECT_NEAR(e.v.x(), v_true.x(), 1e-3);
  EXPECT_NEAR(e.v.y(), v_true.y(), 1e-3);
  EXPECT_NEAR(e.v.z(), v_true.z(), 1e-3);
}

TEST(BallStateEstimatorTest, ParabolicZReturnsCorrectVerticalVelocity) {
  common::PlannerConfig cfg;
  trajectory::BallStateEstimator est(cfg);
  double z0 = 1.0, vz0 = 0.0, g = -9.81;
  int n = 20;
  for (int i = 0; i < n; ++i) {
    double t = i * kDt;
    Eigen::Vector3d p(0.0, 0.0, z0 + vz0 * t + 0.5 * g * t * t);
    est.push(t, p);
  }
  auto e = est.estimate();
  double expected_vz = vz0 + g * e.t;
  EXPECT_NEAR(e.v.z(), expected_vz, 1e-3);
}

TEST(BallStateEstimatorTest, ManualResetClearsBuffer) {
  common::PlannerConfig cfg;
  trajectory::BallStateEstimator est(cfg);
  for (int i = 0; i < 7; ++i) {
    est.push(i * kDt, Eigen::Vector3d(0.0, 0.0, 0.1));
  }
  ASSERT_TRUE(est.ready());
  est.reset();
  EXPECT_FALSE(est.ready());
}

TEST(BallStateEstimatorTest, FewerThanSixSamplesNotReady) {
  common::PlannerConfig cfg;
  trajectory::BallStateEstimator est(cfg);
  for (int i = 0; i < 5; ++i) {
    est.push(i * kDt, Eigen::Vector3d(0.0, 0.0, 0.5));
  }
  EXPECT_FALSE(est.ready());
}
