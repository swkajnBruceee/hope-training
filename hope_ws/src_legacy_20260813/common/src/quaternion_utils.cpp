#include "quaternion_utils.h"

#include <cmath>

namespace common {

namespace {

Eigen::Quaterniond quatFromAxisAngle(const Eigen::Vector3d & axis, double angle) {
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
}

Eigen::Matrix3d quatToMatrix(const Eigen::Quaterniond & q) {
  return q.toRotationMatrix();
}

Eigen::Quaterniond quatMultiply(const Eigen::Quaterniond & a, const Eigen::Quaterniond & b) {
  return a * b;
}

}  // namespace

Eigen::Quaterniond normalToQuaternion(
  const Eigen::Vector3d & normal_in,
  bool constrain_up)
{
  Eigen::Vector3d n = normal_in.normalized();
  Eigen::Vector3d ref = Eigen::Vector3d(1.0, 0.0, 0.0);

  Eigen::Vector3d axis = ref.cross(n);
  double sin_angle = axis.norm();
  double cos_angle = ref.dot(n);

  Eigen::Quaterniond q;
  if (sin_angle < 1e-8) {
    if (cos_angle > 0.0) {
      q = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    } else {
      q = Eigen::Quaterniond(0.0, 0.0, 1.0, 0.0);
    }
  } else {
    axis /= sin_angle;
    double angle = std::atan2(sin_angle, cos_angle);
    q = quatFromAxisAngle(axis, angle);
  }

  if (!constrain_up) {
    return q;
  }

  Eigen::Matrix3d R = quatToMatrix(q);
  Eigen::Vector3d current_y = R.col(1);
  Eigen::Vector3d down = Eigen::Vector3d(0.0, 0.0, -1.0);
  Eigen::Vector3d desired_y = down - down.dot(n) * n;
  double desired_y_norm = desired_y.norm();
  if (desired_y_norm < 1e-6) {
    return q;
  }
  desired_y /= desired_y_norm;

  double cos_roll = std::clamp(current_y.dot(desired_y), -1.0, 1.0);
  Eigen::Vector3d cross_y = current_y.cross(desired_y);
  double sin_roll = cross_y.dot(n);
  double roll_angle = std::atan2(sin_roll, cos_roll);

  Eigen::Quaterniond q_roll(Eigen::AngleAxisd(roll_angle, n));
  Eigen::Quaterniond q_combined = quatMultiply(q_roll, q);
  return q_combined;
}

}  // namespace common
