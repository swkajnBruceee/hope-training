#ifndef CALIBRATION_CALIBRATION_H
#define CALIBRATION_CALIBRATION_H

#include <vector>

#include "constants.h"

namespace calibration {

/**
 * Fit the drag coefficient k and the horizontal/vertical restitution
 * coefficients C_h, C_v from recorded ball trajectories, following
 * HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 2.1.
 */
common::BallPhysics calibrateBallPhysics(
  const std::vector<std::vector<double>> & timestamps,
  const std::vector<std::vector<Eigen::Vector3d>> & trajectories,
  const Eigen::Vector3d & g = Eigen::Vector3d(0.0, 0.0, -9.81));

/**
 * Load a single trajectory CSV with columns t,x,y,z (header optional).
 * Returns timestamps and positions.
 */
struct CsvTrajectory {
  std::vector<double> t;
  std::vector<Eigen::Vector3d> p;
};
CsvTrajectory loadTrajectoryCsv(const std::string & path);

}  // namespace calibration

#endif  // CALIBRATION_CALIBRATION_H
