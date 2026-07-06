#include "landing_decision.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace decision {

namespace {

constexpr double kEpsilon = 1e-9;

double safeRatio(double numerator, double denominator) {
  if (std::abs(denominator) < kEpsilon) {
    return 0.0;
  }
  return numerator / denominator;
}

}  // namespace

std::vector<LandingCandidate> LandingCandidateGenerator::generate() const {
  const struct NamedPoint {
    const char * name;
    double x;
    double y;
  } points[] = {
    {"center_mid", 1.95, -0.7625},
    {"center_deep", 2.30, -0.7625},
    {"side_neg_mid", 1.95, -1.10},
    {"side_pos_mid", 1.95, -0.425},
    {"side_neg_deep", 2.30, -1.10},
    {"side_pos_deep", 2.30, -0.425},
    {"short_center", 1.65, -0.7625},
  };
  const double flight_times[] = {0.45, 0.50, 0.55, 0.60};

  std::vector<LandingCandidate> out;
  out.reserve(28);
  for (const auto & point : points) {
    for (double dt : flight_times) {
      LandingCandidate candidate;
      candidate.target_land = Eigen::Vector3d(point.x, point.y, 0.0);
      candidate.delta_t_flight = dt;
      candidate.mode = std::string("dynamic_") + point.name;
      out.push_back(candidate);
    }
  }
  return out;
}

HardConstraintFilter::HardConstraintFilter(
  const LandingDecisionConfig & decision_config,
  const common::BallPhysics & physics,
  const common::PlannerConfig & planner_config,
  const common::TableParams & table)
: decision_config_(decision_config),
  physics_(physics),
  planner_config_(planner_config),
  table_(table),
  hit_solver_(physics_, planner_config_, table_),
  racket_solver_(physics_, planner_config_, table_)
{
}

LandingCandidate HardConstraintFilter::evaluate(
  const trajectory::StrikeTarget & strike,
  double time_to_strike,
  const LandingCandidate & candidate) const
{
  LandingCandidate out = candidate;
  out.hard_valid = false;

  if (!strike.valid) {
    out.hard_reason = "invalid_strike";
    return out;
  }
  if (!finiteVector(strike.p_ball) || !finiteVector(strike.v_ball) ||
      !std::isfinite(strike.t_strike) || !std::isfinite(time_to_strike)) {
    out.hard_reason = "non_finite_strike";
    return out;
  }
  if (time_to_strike < decision_config_.min_time_to_strike) {
    out.hard_reason = "too_late_to_execute";
    return out;
  }
  if (!finiteVector(candidate.target_land) || !std::isfinite(candidate.delta_t_flight)) {
    out.hard_reason = "non_finite_target";
    return out;
  }
  if (!isOpponentLandingPoint(candidate.target_land)) {
    out.hard_reason = "outside_opponent_table";
    return out;
  }
  if (candidate.delta_t_flight < decision_config_.min_flight_time ||
      candidate.delta_t_flight > decision_config_.max_flight_time) {
    out.hard_reason = "flight_time_limit";
    return out;
  }

  solver::SolveTarget target;
  target.target_land = candidate.target_land;
  target.delta_t_flight = candidate.delta_t_flight;
  target.desired_ball_speed = decision_config_.desired_ball_speed;
  target.max_ball_out_speed = decision_config_.max_ball_out_speed;
  target.max_racket_speed = decision_config_.racket_speed_planning_cap;
  target.net_clearance_margin = decision_config_.net_clearance_margin;
  target.valid = true;
  target.mode = candidate.mode;

  out.plan = hit_solver_.solve(strike, target);
  if (!out.plan.valid) {
    out.hard_reason = out.plan.reason.empty() ? "invalid_plan" : out.plan.reason;
    return out;
  }
  if (!out.plan.clears_net) {
    out.hard_reason = "net_not_clear";
    return out;
  }
  if (out.plan.bypasses_net_posts) {
    out.hard_reason = "net_post_bypass";
    return out;
  }
  if (out.plan.racket_velocity.norm() > decision_config_.racket_speed_planning_cap) {
    out.hard_reason = "racket_speed_limit";
    return out;
  }
  if (touchesTableBeforeTarget(strike.p_ball, out.plan.v_out, candidate.delta_t_flight)) {
    out.hard_reason = "early_table_contact";
    return out;
  }

  out.hard_valid = true;
  out.hard_reason = "ok";
  return out;
}

bool HardConstraintFilter::finiteVector(const Eigen::Vector3d & v) const {
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

bool HardConstraintFilter::isOpponentLandingPoint(const Eigen::Vector3d & p) const {
  return p.x() >= table_.net_x && p.x() <= table_.length &&
         p.y() >= -table_.width && p.y() <= 0.0 &&
         std::abs(p.z()) <= 1e-6;
}

bool HardConstraintFilter::touchesTableBeforeTarget(
  const Eigen::Vector3d & p0,
  const Eigen::Vector3d & v0,
  double duration) const
{
  double elapsed = 0.0;
  Eigen::Vector3d last_p = p0;
  while (elapsed < duration - 1e-12) {
    const double next_elapsed = std::min(elapsed + planner_config_.dt_integrate, duration);
    auto [p, v] = racket_solver_.integrateFlight(p0, v0, next_elapsed);
    (void)v;
    if (next_elapsed < duration - 2.0 * planner_config_.dt_integrate &&
        last_p.z() > 0.0 && p.z() <= 0.0 &&
        p.x() >= table_.net_x && p.x() <= table_.length &&
        p.y() >= -table_.width && p.y() <= 0.0) {
      return true;
    }
    last_p = p;
    elapsed = next_elapsed;
  }
  return false;
}

SoftConstraintScorer::SoftConstraintScorer(
  const LandingDecisionConfig & decision_config,
  const common::BallPhysics & physics,
  const common::PlannerConfig & planner_config,
  const common::TableParams & table)
: decision_config_(decision_config),
  physics_(physics),
  planner_config_(planner_config),
  table_(table),
  comfort_point_(2.055, -0.7625, 0.0)
{
}

LandingCandidate SoftConstraintScorer::score(const LandingCandidate & candidate) const {
  LandingCandidate out = candidate;

  out.edge_score = edgeMarginScore(out.target_land);
  out.net_score = netClearanceScore(out);

  const double racket_speed = out.plan.racket_velocity.norm();
  const double racket_penalty =
    rangePenalty(racket_speed, 0.0, 3.0, decision_config_.racket_speed_planning_cap - 3.0);
  out.racket_speed_score = 1.0 - racket_penalty;

  const double ball_speed = out.plan.v_out.norm();
  const double ball_penalty =
    rangePenalty(
      ball_speed,
      decision_config_.ball_speed_comfort_min,
      decision_config_.ball_speed_comfort_max,
      3.0);
  out.ball_speed_score = 1.0 - ball_penalty;

  const double flight_penalty =
    rangePenalty(
      out.delta_t_flight,
      decision_config_.flight_time_comfort_min,
      decision_config_.flight_time_comfort_max,
      0.10);
  out.flight_time_score = 1.0 - flight_penalty;

  out.competitiveness_score = competitivenessScore(out.target_land);

  out.total_score =
    0.28 * out.edge_score +
    0.18 * out.net_score +
    0.18 * out.racket_speed_score +
    0.14 * out.ball_speed_score +
    0.12 * out.flight_time_score +
    0.10 * out.competitiveness_score;
  return out;
}

double SoftConstraintScorer::clamp01(double value) const {
  return std::max(0.0, std::min(1.0, value));
}

double SoftConstraintScorer::rangePenalty(
  double value,
  double min_value,
  double max_value,
  double scale) const
{
  if (value >= min_value && value <= max_value) {
    return 0.0;
  }
  const double distance = value < min_value ? (min_value - value) : (value - max_value);
  return clamp01(safeRatio(distance, scale));
}

double SoftConstraintScorer::edgeMarginScore(const Eigen::Vector3d & p) const {
  const double margin_to_net = p.x() - table_.net_x;
  const double margin_to_end = table_.length - p.x();
  const double margin_to_left = p.y() - (-table_.width);
  const double margin_to_right = 0.0 - p.y();
  const double min_margin =
    std::min(std::min(margin_to_net, margin_to_end), std::min(margin_to_left, margin_to_right));
  return clamp01(safeRatio(min_margin, decision_config_.edge_margin_comfort));
}

double SoftConstraintScorer::netClearanceScore(const LandingCandidate & candidate) const {
  const double margin = decision_config_.net_clearance_margin;
  const double comfort = decision_config_.net_clearance_comfort;
  if (comfort <= margin + kEpsilon) {
    return 1.0;
  }

  solver::RacketTargetSolver racket_solver{
    physics_, planner_config_, table_};
  Eigen::Vector3d p_net;
  if (!racket_solver.positionAtX(
      candidate.plan.p_hit,
      candidate.plan.v_out,
      table_.net_x,
      2.0,
      p_net)) {
    return 0.0;
  }
  const double clearance = p_net.z() - table_.net_height;
  return clamp01(safeRatio(clearance - margin, comfort - margin));
}

double SoftConstraintScorer::competitivenessScore(const Eigen::Vector3d & p) const {
  const double half_x = table_.length - table_.net_x;
  const double diagonal = std::sqrt(half_x * half_x + table_.width * table_.width);
  const double competitiveness = clamp01((p - comfort_point_).norm() / (0.5 * diagonal));
  const double target = clamp01(decision_config_.competitiveness_level);
  return 1.0 - clamp01(std::abs(competitiveness - target));
}

LandingDecisionPlanner::LandingDecisionPlanner(
  const LandingDecisionConfig & decision_config,
  const common::BallPhysics & physics,
  const common::PlannerConfig & planner_config,
  const common::TableParams & table)
: decision_config_(decision_config),
  planner_config_(planner_config),
  hard_filter_(decision_config, physics, planner_config, table),
  scorer_(decision_config, physics, planner_config, table)
{
}

LandingDecisionResult LandingDecisionPlanner::select(
  const trajectory::StrikeTarget & strike,
  double time_to_strike) const
{
  LandingDecisionResult result;
  result.target = makeNoFeasibleTarget();

  const auto candidates = generator_.generate();
  result.candidate_count = static_cast<int>(candidates.size());

  bool has_best = false;
  LandingCandidate best;
  for (const auto & candidate : candidates) {
    auto evaluated = hard_filter_.evaluate(strike, time_to_strike, candidate);
    if (!evaluated.hard_valid) {
      result.reject_reasons[evaluated.hard_reason]++;
      result.candidates.push_back(evaluated);
      continue;
    }
    result.hard_valid_count++;
    auto scored = scorer_.score(evaluated);
    result.candidates.push_back(scored);
    if (!has_best || scored.total_score > best.total_score) {
      best = scored;
      has_best = true;
    }
  }

  if (has_best) {
    result.selected = best;
    for (auto & candidate : result.candidates) {
      candidate.selected = candidate.mode == best.mode &&
                           std::abs(candidate.delta_t_flight - best.delta_t_flight) < 1e-9;
    }
    result.target = candidateToTarget(best, best.mode);
    return result;
  }

  LandingCandidate fallback;
  fallback.target_land = planner_config_.target_land;
  fallback.delta_t_flight = planner_config_.delta_t_flight;
  fallback.mode = "fallback_fixed_center";
  auto evaluated = hard_filter_.evaluate(strike, time_to_strike, fallback);
  if (evaluated.hard_valid) {
    result.used_fallback = true;
    result.hard_valid_count++;
    result.selected = scorer_.score(evaluated);
    result.target = candidateToTarget(result.selected, "fallback_fixed_center");
    return result;
  }

  result.reject_reasons[evaluated.hard_reason]++;
  result.target = makeNoFeasibleTarget();
  return result;
}

TargetDecisionData LandingDecisionPlanner::makeNoFeasibleTarget() const {
  TargetDecisionData out;
  out.target_land = planner_config_.target_land;
  out.delta_t_flight = planner_config_.delta_t_flight;
  out.desired_ball_speed = decision_config_.desired_ball_speed;
  out.max_ball_out_speed = decision_config_.max_ball_out_speed;
  out.max_racket_speed = decision_config_.racket_speed_planning_cap;
  out.net_clearance_margin = decision_config_.net_clearance_margin;
  out.valid = false;
  out.mode = "no_feasible_landing";
  return out;
}

TargetDecisionData LandingDecisionPlanner::candidateToTarget(
  const LandingCandidate & candidate,
  const std::string & mode) const
{
  TargetDecisionData out;
  out.target_land = candidate.target_land;
  out.delta_t_flight = candidate.delta_t_flight;
  out.desired_ball_speed = decision_config_.desired_ball_speed;
  out.max_ball_out_speed = decision_config_.max_ball_out_speed;
  out.max_racket_speed = decision_config_.racket_speed_planning_cap;
  out.net_clearance_margin = decision_config_.net_clearance_margin;
  out.valid = true;
  out.mode = mode;
  return out;
}

}  // namespace decision
