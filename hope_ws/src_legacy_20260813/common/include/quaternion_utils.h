#ifndef COMMON_QUATERNION_UTILS_H
#define COMMON_QUATERNION_UTILS_H

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace common {

/**
 * Convert a racket face normal to an orientation quaternion [x, y, z, w].
 *
 * The face normal defines paddle tilt (2 DOF) but not roll about the face axis.
 * The constrain_up option aligns the paddle handle (local Y) with world -Z.
 *
 * Returns an Eigen::Quaterniond (w, x, y, z). Callers that need a ROS
 * geometry_msgs::msg::Quaternion can convert via q.x()/q.y()/q.z()/q.w().
 */
Eigen::Quaterniond normalToQuaternion(
  const Eigen::Vector3d & normal,
  bool constrain_up = false);

}  // namespace common

#endif  // COMMON_QUATERNION_UTILS_H
