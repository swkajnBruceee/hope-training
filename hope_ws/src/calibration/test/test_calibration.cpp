#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

#include "calibration.h"

namespace {

std::vector<std::pair<double, Eigen::Vector3d>> freeFlight(
  double dt = 1.0 / 360.0, int n = 160)
{
  std::vector<std::pair<double, Eigen::Vector3d>> out;
  double t = 0.0;
  Eigen::Vector3d v(3.0, 0.5, 4.0);
  Eigen::Vector3d p(0.0, 0.0, 0.5);
  for (int i = 0; i < n; ++i) {
    out.emplace_back(t, p);
    v = v + Eigen::Vector3d(0.0, 0.0, -9.81) * dt;
    p = p + v * dt;
    t += dt;
  }
  return out;
}

std::vector<std::pair<double, Eigen::Vector3d>> cleanBounce(
  double c_v, double c_h,
  double vz1 = -10.0, double vx1 = 4.0,
  double dt = 1.0 / 360.0, int n_pre = 12, int n_post = 12)
{
  std::vector<std::pair<double, Eigen::Vector3d>> out;
  double v_post_z = -c_v * vz1;
  double v_post_x = c_h * vx1;
  double t = 0.0;
  for (int k = n_pre; k > 0; --k) {
    double ts = out.size() * dt;
    Eigen::Vector3d p(vx1 * (-k) * dt, 0.0, vz1 * (-k) * dt);
    out.emplace_back(ts, p);
    t = ts;
  }
  double tc = out.size() * dt;
  out.emplace_back(tc, Eigen::Vector3d(0.0, 0.0, 0.0));
  for (int k = 1; k <= n_post; ++k) {
    double ts = out.size() * dt;
    Eigen::Vector3d p(v_post_x * k * dt, 0.0, v_post_z * k * dt);
    out.emplace_back(ts, p);
    t = ts;
  }
  return out;
}

}  // namespace

TEST(CalibrationTest, DragFreeFlightFitsNearZeroK) {
  auto rows = freeFlight();
  std::vector<std::vector<double>> timestamps;
  std::vector<std::vector<Eigen::Vector3d>> trajectories;
  std::vector<double> t;
  std::vector<Eigen::Vector3d> p;
  for (const auto & [tt, pp] : rows) {
    t.push_back(tt);
    p.push_back(pp);
  }
  timestamps.push_back(t);
  trajectories.push_back(p);
  auto phys = calibration::calibrateBallPhysics(timestamps, trajectories);
  EXPECT_LT(phys.k, 0.05);
}

TEST(CalibrationTest, RestitutionRecoveredFromSyntheticBounce) {
  auto rows = cleanBounce(0.60, 0.55);
  std::vector<std::vector<double>> timestamps;
  std::vector<std::vector<Eigen::Vector3d>> trajectories;
  std::vector<double> t;
  std::vector<Eigen::Vector3d> p;
  for (const auto & [tt, pp] : rows) {
    t.push_back(tt);
    p.push_back(pp);
  }
  timestamps.push_back(t);
  trajectories.push_back(p);
  auto phys = calibration::calibrateBallPhysics(timestamps, trajectories);
  EXPECT_NEAR(phys.C_v, 0.60, 0.02);
  EXPECT_NEAR(phys.C_h, 0.55, 0.02);
}

TEST(CalibrationTest, EmptyInputReturnsDefaults) {
  std::vector<std::vector<double>> timestamps;
  std::vector<std::vector<Eigen::Vector3d>> trajectories;
  auto phys = calibration::calibrateBallPhysics(timestamps, trajectories);
  EXPECT_DOUBLE_EQ(phys.k, 0.09375);
  EXPECT_DOUBLE_EQ(phys.C_h, 0.649);
  EXPECT_DOUBLE_EQ(phys.C_v, 0.906);
}
