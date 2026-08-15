#include "racket_target_solver.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "ball_trajectory_predictor.h"

namespace solver {

RacketTargetSolver::RacketTargetSolver(
  const common::BallPhysics & physics,
  const common::PlannerConfig & config,
  const common::TableParams & table)
: physics_(physics), config_(config), table_(table)
{
}

Eigen::Vector3d RacketTargetSolver::flightAcceleration(const Eigen::Vector3d & v) const {
  double speed = v.norm();
  return -physics_.k * speed * v + physics_.g;
}

Eigen::Vector3d RacketTargetSolver::computeOutgoingVelocity(
  const Eigen::Vector3d & p_strike,
  const Eigen::Vector3d & p_land,
  double delta_t) const
{
  if (delta_t <= 1e-6) {
    throw std::invalid_argument("delta_t must be positive");
  }
  Eigen::Vector3d v = (p_land - p_strike) / delta_t - 0.5 * physics_.g * delta_t;
  for (int iter = 0; iter < 24; ++iter) {
    auto [p_end, v_end] = integrateFlight(p_strike, v, delta_t);
    Eigen::Vector3d error = p_land - p_end;
    if (error.norm() < 1e-4) {
      break;
    }
    v = v + error / delta_t;
  }
  return v;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> RacketTargetSolver::integrateFlight(
  const Eigen::Vector3d & p0,
  const Eigen::Vector3d & v0,
  double duration) const
{
  double dt_nominal = config_.dt_integrate;
  Eigen::Vector3d p = p0;
  Eigen::Vector3d v = v0;
  double elapsed = 0.0;
  while (elapsed < duration - 1e-12) {
    double dt = std::min(dt_nominal, duration - elapsed);
    Eigen::Vector3d a = flightAcceleration(v);
    p = p + v * dt + 0.5 * a * dt * dt;
    v = v + a * dt;
    elapsed += dt;
  }
  return {p, v};
}

bool RacketTargetSolver::positionAtX(
  const Eigen::Vector3d & p0,
  const Eigen::Vector3d & v0,
  double x_target,
  double max_time,
  Eigen::Vector3d & out) const
{
  if (p0.x() >= x_target) {
    out = p0;
    return true;
  }
  if (v0.x() <= 0.0) {
    return false;
  }
  double dt_nominal = config_.dt_integrate;
  Eigen::Vector3d p = p0;
  Eigen::Vector3d v = v0;
  double elapsed = 0.0;
  while (elapsed < max_time - 1e-12) {
    double dt = std::min(dt_nominal, max_time - elapsed);
    Eigen::Vector3d a = flightAcceleration(v);
    Eigen::Vector3d p_next = p + v * dt + 0.5 * a * dt * dt;
    Eigen::Vector3d v_next = v + a * dt;
    if (p.x() <= x_target && x_target <= p_next.x()) {
      double dx = p_next.x() - p.x();
      double frac = (std::abs(dx) > 1e-9) ? ((x_target - p.x()) / dx) : 0.0;
      if (frac < 0.0) {
        frac = 0.0;
      }
      if (frac > 1.0) {
        frac = 1.0;
      }
      out = p + frac * (p_next - p);
      return true;
    }
    p = p_next;
    v = v_next;
    elapsed += dt;
  }
  return false;
}

Eigen::Vector3d RacketTargetSolver::faceOpponent(const Eigen::Vector3d & n_in) const {
  Eigen::Vector3d n = n_in;
  double norm = n.norm();
  if (norm < 1e-9) {
    return Eigen::Vector3d(1.0, 0.0, 0.0);
  }
  n /= norm;
  if (n.x() < 0.0) {
    n = -n;
  }
  if (n.x() <= 1e-6) {
    n = n + Eigen::Vector3d(1e-6, 0.0, 0.0);
    n /= n.norm();
  }
  return n;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> RacketTargetSolver::computeRacketVelocity(
  const Eigen::Vector3d & v_incoming,
  const Eigen::Vector3d & v_outgoing,
  double C_r) const
{
  Eigen::Vector3d delta_v = v_outgoing - v_incoming;
  double delta_v_norm = delta_v.norm();
  if (delta_v_norm < 1e-6) {
    Eigen::Vector3d n = faceOpponent(-v_incoming);
    return {Eigen::Vector3d::Zero(), n};
  }
  Eigen::Vector3d u_hat = faceOpponent(delta_v);
  double v_o_n = v_outgoing.dot(u_hat);
  double v_i_n = v_incoming.dot(u_hat);
  double v_r_n = (v_o_n + C_r * v_i_n) / (1.0 + C_r);
  return {v_r_n * u_hat, u_hat};
}

std::pair<bool, bool> RacketTargetSolver::checkNetClearance(
  const Eigen::Vector3d & p_strike,
  const Eigen::Vector3d & v_outgoing,
  double margin) const
{
  double x_net = table_.net_x;
  double z_net = table_.net_height;
  if (v_outgoing.x() <= 0.0) {
    return {false, false};
  }
  Eigen::Vector3d p_net;
  if (!positionAtX(p_strike, v_outgoing, x_net, config_.max_predict_time, p_net)) {
    return {false, false};
  }
  double z_at_net = p_net.z();
  double y_at_net = p_net.y();
  double y_net_min = -table_.width - table_.net_overhang;
  double y_net_max = table_.net_overhang;
  bool bypasses = (y_at_net < y_net_min) || (y_at_net > y_net_max);
  if (bypasses) {
    return {false, true};
  }
  return {z_at_net > (z_net + margin), false};
}

RacketCommand RacketTargetSolver::plan(const trajectory::StrikeTarget & strike) {
  RacketCommand cmd;
  cmd.target_land = config_.target_land;
  cmd.num_bounces = strike.num_bounces;

  if (!strike.valid) {
    cmd.p_intercept = strike.p_ball;
    cmd.n_racket = Eigen::Vector3d(1.0, 0.0, 0.0);
    cmd.t_strike = strike.t_strike;
    cmd.valid = false;
    return cmd;
  }

  Eigen::Vector3d p_strike = strike.p_ball;
  Eigen::Vector3d v_incoming = strike.v_ball;
  Eigen::Vector3d p_land = config_.target_land;

  Eigen::Vector3d v_outgoing = computeOutgoingVelocity(p_strike, p_land, config_.delta_t_flight);
  auto [v_racket, n_racket] = computeRacketVelocity(v_incoming, v_outgoing, config_.C_r);
  auto [clears, bypasses] = checkNetClearance(p_strike, v_outgoing);

  // Auto-adjust flight time if net clearance fails
  if (!clears) {
    const double candidates[] = {0.4, 0.6, 0.35, 0.7, 0.3};
    for (double dt_adj : candidates) {
      Eigen::Vector3d v_out_adj = computeOutgoingVelocity(p_strike, p_land, dt_adj);
      auto [clears_adj, bypasses_adj] = checkNetClearance(p_strike, v_out_adj);
      if (clears_adj) {
        v_outgoing = v_out_adj;
        auto vr_n = computeRacketVelocity(v_incoming, v_outgoing, config_.C_r);
        v_racket = vr_n.first;
        n_racket = vr_n.second;
        clears = true;
        bypasses = bypasses_adj;
        break;
      }
    }
  }

  cmd.p_intercept = p_strike;
  cmd.v_racket = v_racket;
  cmd.n_racket = n_racket;
  cmd.t_strike = strike.t_strike;
  cmd.v_ball_outgoing = v_outgoing;
  cmd.clears_net = clears;
  cmd.bypasses_net_posts = bypasses;
  cmd.valid = true;
  return cmd;
}

}  // namespace solver
