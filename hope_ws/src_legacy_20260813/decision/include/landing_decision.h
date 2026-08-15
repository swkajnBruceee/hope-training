#ifndef DECISION_LANDING_DECISION_H
#define DECISION_LANDING_DECISION_H

#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "ball_trajectory_predictor.h"
#include "constants.h"
#include "hit_plan.h"
#include "hit_plan_solver.h"
#include "racket_target_solver.h"
#include "target_selector.h"

namespace decision {

struct LandingDecisionConfig {
  double min_time_to_strike = 0.12;
  double min_flight_time = 0.40;
  double max_flight_time = 0.70;
  double net_clearance_margin = 0.03;
  double net_clearance_comfort = 0.05;
  double racket_speed_rule_cap = 6.0;
  double racket_speed_planning_cap = 5.4;
  double ball_speed_comfort_min = 4.5;
  double ball_speed_comfort_max = 8.5;
  double flight_time_comfort_min = 0.48;
  double flight_time_comfort_max = 0.60;
  double competitiveness_level = 0.25;
  double edge_margin_comfort = 0.20;
  double desired_ball_speed = -1.0;
  double max_ball_out_speed = -1.0;
};

struct LandingCandidate {
  Eigen::Vector3d target_land = Eigen::Vector3d::Zero();
  double delta_t_flight = 0.0;
  std::string mode;

  solver::HitPlan plan;
  bool hard_valid = false;
  bool selected = false;
  std::string hard_reason;

  double total_score = 0.0;
  double edge_score = 0.0;
  double net_score = 0.0;
  double racket_speed_score = 0.0;
  double ball_speed_score = 0.0;
  double flight_time_score = 0.0;
  double competitiveness_score = 0.0;
  double tactical_score = 0.0;
};

struct LandingDecisionResult {
  TargetDecisionData target;
  bool used_fallback = false;
  int candidate_count = 0;
  int hard_valid_count = 0;
  LandingCandidate selected;
  std::vector<LandingCandidate> candidates;
  std::map<std::string, int> reject_reasons;
};

class LandingCandidateGenerator {
 public:
  std::vector<LandingCandidate> generate() const;
};

class HardConstraintFilter {
 public:
  HardConstraintFilter(
    const LandingDecisionConfig & decision_config,
    const common::BallPhysics & physics,
    const common::PlannerConfig & planner_config,
    const common::TableParams & table);

  LandingCandidate evaluate(
    const trajectory::StrikeTarget & strike,
    double time_to_strike,
    const LandingCandidate & candidate) const;

 private:
  bool finiteVector(const Eigen::Vector3d & v) const;
  bool isOpponentLandingPoint(const Eigen::Vector3d & p) const;
  bool touchesTableBeforeTarget(
    const Eigen::Vector3d & p0,
    const Eigen::Vector3d & v0,
    double duration) const;

  LandingDecisionConfig decision_config_;
  common::BallPhysics physics_;
  common::PlannerConfig planner_config_;
  common::TableParams table_;
  solver::HitPlanSolver hit_solver_;
  solver::RacketTargetSolver racket_solver_;
};

class SoftConstraintScorer {
 public:
  SoftConstraintScorer(
    const LandingDecisionConfig & decision_config,
    const common::BallPhysics & physics,
    const common::PlannerConfig & planner_config,
    const common::TableParams & table);

  LandingCandidate score(
    const LandingCandidate & candidate,
    const trajectory::StrikeTarget & strike) const;

 private:
  double clamp01(double value) const;
  double rangePenalty(double value, double min_value, double max_value, double scale) const;
  double edgeMarginScore(const Eigen::Vector3d & p) const;
  double netClearanceScore(const LandingCandidate & candidate) const;
  double competitivenessScore(const Eigen::Vector3d & p) const;
  double tacticalPlacementScore(
    const Eigen::Vector3d & p,
    const trajectory::StrikeTarget & strike) const;

  LandingDecisionConfig decision_config_;
  common::BallPhysics physics_;
  common::PlannerConfig planner_config_;
  common::TableParams table_;
  Eigen::Vector3d comfort_point_;
};

class LandingDecisionPlanner {
 public:
  LandingDecisionPlanner(
    const LandingDecisionConfig & decision_config,
    const common::BallPhysics & physics,
    const common::PlannerConfig & planner_config,
    const common::TableParams & table);

  LandingDecisionResult select(
    const trajectory::StrikeTarget & strike,
    double time_to_strike) const;

  TargetDecisionData makeNoFeasibleTarget() const;

 private:
  TargetDecisionData candidateToTarget(const LandingCandidate & candidate, const std::string & mode) const;

  LandingDecisionConfig decision_config_;
  common::PlannerConfig planner_config_;
  LandingCandidateGenerator generator_;
  HardConstraintFilter hard_filter_;
  SoftConstraintScorer scorer_;
};

}  // namespace decision

#endif  // DECISION_LANDING_DECISION_H
