#include <gtest/gtest.h>

#include "landing_error_corrector.h"

namespace {

solver::LandingFeedbackSample makeSample(const Eigen::Vector3d & error) {
  solver::LandingFeedbackSample sample;
  sample.hit_position = Eigen::Vector3d(0.0, -0.75, 0.25);
  sample.ball_velocity_incoming = Eigen::Vector3d(-3.0, 0.1, -0.5);
  sample.ball_velocity_outgoing = Eigen::Vector3d(5.0, 0.0, 2.5);
  sample.target_land = Eigen::Vector3d(2.3, -0.75, 0.0);
  sample.actual_landing = sample.target_land + error;
  sample.return_flight_time = 0.5;
  return sample;
}

}  // namespace

TEST(LandingErrorCorrectorTest, WaitsForMinimumHistory) {
  solver::LandingErrorCorrectionConfig config;
  config.min_samples = 3;
  solver::LandingErrorCorrector corrector(config);
  corrector.addSample(makeSample(Eigen::Vector3d(0.1, 0.0, 0.0)));

  const auto correction = corrector.correctionFor(
    Eigen::Vector3d(0.0, -0.75, 0.25),
    Eigen::Vector3d(-3.0, 0.1, -0.5),
    Eigen::Vector3d(2.3, -0.75, 0.0),
    0.5);

  EXPECT_FALSE(correction.active);
  EXPECT_EQ(correction.sample_count, 1u);
}

TEST(LandingErrorCorrectorTest, ConvertsEstimatedLandingErrorToOppositeVelocityCorrection) {
  solver::LandingErrorCorrectionConfig config;
  config.min_samples = 2;
  config.correction_gain = 0.5;
  config.max_delta_v = 10.0;
  solver::LandingErrorCorrector corrector(config);
  corrector.addSample(makeSample(Eigen::Vector3d(0.10, -0.20, 0.0)));
  corrector.addSample(makeSample(Eigen::Vector3d(0.10, -0.20, 0.0)));

  const auto correction = corrector.correctionFor(
    Eigen::Vector3d(0.0, -0.75, 0.25),
    Eigen::Vector3d(-3.0, 0.1, -0.5),
    Eigen::Vector3d(2.3, -0.75, 0.0),
    0.5);

  ASSERT_TRUE(correction.active);
  EXPECT_NEAR(correction.estimated_landing_error.x(), 0.10, 1e-9);
  EXPECT_NEAR(correction.estimated_landing_error.y(), -0.20, 1e-9);
  EXPECT_NEAR(correction.delta_v_out.x(), -0.10, 1e-9);
  EXPECT_NEAR(correction.delta_v_out.y(), 0.20, 1e-9);
  EXPECT_NEAR(correction.delta_v_out.z(), 0.0, 1e-12);
}

TEST(LandingErrorCorrectorTest, ClampsLargeVelocityCorrection) {
  solver::LandingErrorCorrectionConfig config;
  config.min_samples = 1;
  config.correction_gain = 1.0;
  config.max_delta_v = 0.25;
  solver::LandingErrorCorrector corrector(config);
  corrector.addSample(makeSample(Eigen::Vector3d(2.0, 0.0, 0.0)));

  const auto correction = corrector.correctionFor(
    Eigen::Vector3d(0.0, -0.75, 0.25),
    Eigen::Vector3d(-3.0, 0.1, -0.5),
    Eigen::Vector3d(2.3, -0.75, 0.0),
    0.5);

  EXPECT_TRUE(correction.active);
  EXPECT_NEAR(correction.delta_v_out.norm(), 0.25, 1e-9);
}
