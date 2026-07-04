#ifndef SOLVER_HIT_PLAN_H
#define SOLVER_HIT_PLAN_H

#include <string>

#include <Eigen/Dense>
#include <geometry_msgs/msg/quaternion.hpp>

#include "constants.h"

namespace solver {

struct SolveTarget {
  Eigen::Vector3d target_land = Eigen::Vector3d::Zero();
  double delta_t_flight = 0.0;
  double desired_ball_speed = -1.0;
  double max_ball_out_speed = -1.0;
  double max_racket_speed = -1.0;
  double net_clearance_margin = 0.03;
  bool valid = false;
  std::string mode;
};

inline SolveTarget makeDefaultSolveTarget(const common::PlannerConfig & config) {
  SolveTarget target;
  target.target_land = config.target_land;
  target.delta_t_flight = config.delta_t_flight;
  target.desired_ball_speed = -1.0;
  target.max_ball_out_speed = -1.0;
  target.max_racket_speed = -1.0;
  target.net_clearance_margin = 0.03;
  target.valid = true;
  target.mode = "config_default";
  return target;
}

struct HitPlan {
  HitPlan() {
    racket_orientation.x = 0.0;
    racket_orientation.y = 0.0;
    racket_orientation.z = 0.0;
    racket_orientation.w = 1.0;
  }

  Eigen::Vector3d p_hit = Eigen::Vector3d::Zero();
  double t_hit = 0.0;
  Eigen::Vector3d v_in = Eigen::Vector3d::Zero();

  Eigen::Vector3d target_land = Eigen::Vector3d::Zero();
  double flight_time = 0.0;

  Eigen::Vector3d v_out = Eigen::Vector3d::Zero();
  Eigen::Vector3d racket_normal = Eigen::Vector3d::UnitX();
  Eigen::Vector3d racket_velocity = Eigen::Vector3d::Zero();
  geometry_msgs::msg::Quaternion racket_orientation;

  bool clears_net = false;
  bool bypasses_net_posts = false;
  bool valid = false;

  double score = 0.0;
  std::string reason;
};

}  // namespace solver

#endif  // SOLVER_HIT_PLAN_H
