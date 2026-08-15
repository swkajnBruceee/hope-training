#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

#include "split_calibration_csv.h"

TEST(SplitCalibrationCsvTest, InferScaleDetectsMillimeters) {
  std::vector<Eigen::Vector3d> pos = {
    {1000.0, -500.0, -680.0},
    {1200.0, -450.0, -640.0},
  };
  EXPECT_DOUBLE_EQ(calibration::inferScale(pos), 0.001);
}

TEST(SplitCalibrationCsvTest, SplitRowsBreaksOnLargeTimeGap) {
  std::vector<double> t = {0.0, 0.02, 2.50};
  std::vector<Eigen::Vector3d> p = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.1},
    {2.0, 0.0, 0.2},
  };
  auto chunks = calibration::splitRows(t, p, 1.0);
  EXPECT_EQ(chunks.size(), 2u);
  EXPECT_EQ(chunks[0].size(), 2u);
  EXPECT_EQ(chunks[1].size(), 1u);
}

TEST(SplitCalibrationCsvTest, NormalizeChunkRezerosTimeAndLocalTableHeight) {
  std::vector<double> t = {10.0, 10.1, 10.2, 10.3};
  std::vector<Eigen::Vector3d> p = {
    {0.0, 0.0, -680.0},
    {100.0, 0.0, -680.0},
    {200.0, 0.0, 120.0},
    {300.0, 0.0, -675.0},
  };
  auto norm = calibration::normalizeChunk(t, p, 0.001);
  EXPECT_NEAR(norm.times.front(), 0.0, 1e-12);
  EXPECT_NEAR(norm.times.back(), 0.3, 1e-9);
  EXPECT_LT(norm.z_offset_m, -0.60);
  double min_z = norm.pos.front().z();
  for (const auto & pp : norm.pos) {
    min_z = std::min(min_z, pp.z());
  }
  EXPECT_GT(min_z, -0.05);
}
