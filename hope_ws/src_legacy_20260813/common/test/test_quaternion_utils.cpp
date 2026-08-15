#include <gtest/gtest.h>

#include <cmath>

#include "quaternion_utils.h"

namespace {

Eigen::Matrix3d quatToMatrix(const Eigen::Quaterniond & q) {
  Eigen::Matrix3d R;
  const double x = q.x(), y = q.y(), z = q.z(), w = q.w();
  R << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
       2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
       2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
  return R;
}

}  // namespace

TEST(QuaternionUtilsTest, IsNormalized) {
  Eigen::Vector3d normals[] = {
    {1.0, 0.0, 0.0},
    {-1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {1.0, 1.0, 1.0},
    {0.2, -0.7, 0.4}
  };
  for (const auto & n : normals) {
    auto q = common::normalToQuaternion(n, false);
    EXPECT_NEAR(q.norm(), 1.0, 1e-9);
  }
}

TEST(QuaternionUtilsTest, RotatesLocalXToNormal) {
  Eigen::Vector3d normals[] = {
    {1.0, 0.0, 0.0},
    {-1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.2, -0.7, 0.4}
  };
  Eigen::Vector3d local_x(1.0, 0.0, 0.0);
  for (const auto & n : normals) {
    Eigen::Vector3d target = n.normalized();
    auto q = common::normalToQuaternion(n, false);
    Eigen::Vector3d rotated = quatToMatrix(q) * local_x;
    EXPECT_NEAR((rotated - target).norm(), 0.0, 1e-9);
  }
}

TEST(QuaternionUtilsTest, ConstrainUpNormalized) {
  Eigen::Vector3d normals[] = {
    {1.0, 0.2, 0.3},
    {0.5, -0.5, 0.7}
  };
  for (const auto & n : normals) {
    auto q = common::normalToQuaternion(n, true);
    EXPECT_NEAR(q.norm(), 1.0, 1e-6);
  }
}
