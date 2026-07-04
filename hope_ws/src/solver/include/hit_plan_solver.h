#ifndef SOLVER_HIT_PLAN_SOLVER_H
#define SOLVER_HIT_PLAN_SOLVER_H

#include "constants.h"
#include "hit_plan.h"
#include "racket_target_solver.h"

namespace trajectory {
struct StrikeTarget;
}

namespace solver {

class HitPlanSolver {
 public:
  HitPlanSolver(
    const common::BallPhysics & physics,
    const common::PlannerConfig & config,
    const common::TableParams & table);

  HitPlan solve(
    const trajectory::StrikeTarget & strike,
    const SolveTarget & target) const;

 private:
  geometry_msgs::msg::Quaternion normalToQuaternion(const Eigen::Vector3d & normal) const;

  common::BallPhysics physics_;
  common::PlannerConfig config_;
  common::TableParams table_;
  RacketTargetSolver racket_solver_;
};

}  // namespace solver

#endif  // SOLVER_HIT_PLAN_SOLVER_H
