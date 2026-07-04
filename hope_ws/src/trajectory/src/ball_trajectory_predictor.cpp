#include "ball_trajectory_predictor.h"

#include <algorithm>
#include <cmath>

namespace trajectory {

BallTrajectoryPredictor::BallTrajectoryPredictor(
  const common::BallPhysics & physics,
  const common::PlannerConfig & config,
  const common::TableParams & table)
: physics_(physics), config_(config), table_(table)
{
}

Eigen::Vector3d BallTrajectoryPredictor::flightAcceleration(const Eigen::Vector3d & v) const {
  double speed = v.norm();
  return -physics_.k * speed * v + physics_.g;
}

Eigen::Vector3d BallTrajectoryPredictor::applyBounce(const Eigen::Vector3d & v) const {
  Eigen::Vector3d out;
  out.x() = physics_.C_h * v.x();
  out.y() = physics_.C_h * v.y();
  out.z() = -physics_.C_v * v.z();
  return out;
}

bool BallTrajectoryPredictor::isOnTable(const Eigen::Vector3d & p) const {
  double r = physics_.radius;
  return (-r <= p.x() && p.x() <= table_.length + r &&
          -table_.width - r <= p.y() && p.y() <= r);
}

StrikeTarget BallTrajectoryPredictor::predict(
  const Eigen::Vector3d & p0, const Eigen::Vector3d & v0, double t0)
{
  double dt = config_.dt_integrate;
  int max_steps = static_cast<int>(config_.max_predict_time / dt);
  double x_hit = config_.x_hit;

  Eigen::Vector3d p = p0;
  Eigen::Vector3d v = v0;
  double t = t0;
  int bounces = 0;

  Eigen::Vector3d p_bounce = p;
  Eigen::Vector3d v_post = v;
  double remaining_dt = dt;

  for (int step = 0; step < max_steps; ++step) {
    double p_prev_x = p.x();

    // Euler integration step
    Eigen::Vector3d a = flightAcceleration(v);
    Eigen::Vector3d v_new = v + a * dt;
    Eigen::Vector3d p_new = p + v * dt + 0.5 * a * dt * dt;
    t += dt;
    bool bounce_this_step = false;

    // Bounce detection
    if (p_new.z() < 0.0 && v_new.z() < 0.0) {
      if (isOnTable(p_new)) {
        double dz = p.z() - p_new.z();
        double frac = (dz > 1e-9) ? (p.z() / dz) : 0.5;
        if (frac < 0.0) {
          frac = 0.0;
        }
        if (frac > 1.0) {
          frac = 1.0;
        }

        p_bounce = p + frac * (p_new - p);
        p_bounce.z() = 0.0;
        Eigen::Vector3d v_at_bounce = v + a * (frac * dt);
        v_post = applyBounce(v_at_bounce);

        // Continue from bounce with second-order correction
        remaining_dt = (1.0 - frac) * dt;
        Eigen::Vector3d a_post = flightAcceleration(v_post);
        p_new = p_bounce + v_post * remaining_dt + 0.5 * a_post * remaining_dt * remaining_dt;
        v_new = v_post + a_post * remaining_dt;
        bounces += 1;
        bounce_this_step = true;
      } else {
        p_new.z() = std::max(p_new.z(), 0.0);
      }
    }

    // Hitting plane crossing detection
    if (p_prev_x > x_hit && p_new.x() <= x_hit && v_new.x() < 0.0) {
      Eigen::Vector3d p_cross;
      Eigen::Vector3d v_cross;
      double t_cross;
      if (bounce_this_step) {
        double dx_arc = p_bounce.x() - p_new.x();
        double frac_cross;
        if (std::abs(dx_arc) > 1e-9) {
          frac_cross = (p_bounce.x() - x_hit) / dx_arc;
        } else {
          frac_cross = 0.5;
        }
        if (frac_cross < 0.0) {
          frac_cross = 0.0;
        }
        if (frac_cross > 1.0) {
          frac_cross = 1.0;
        }
        p_cross = p_bounce + frac_cross * (p_new - p_bounce);
        v_cross = v_post + frac_cross * (v_new - v_post);
        t_cross = (t - remaining_dt) + frac_cross * remaining_dt;
      } else {
        double dx_step = p.x() - p_new.x();
        double frac_cross;
        if (std::abs(dx_step) > 1e-9) {
          frac_cross = (p.x() - x_hit) / dx_step;
        } else {
          frac_cross = 0.5;
        }
        if (frac_cross < 0.0) {
          frac_cross = 0.0;
        }
        if (frac_cross > 1.0) {
          frac_cross = 1.0;
        }
        p_cross = p + frac_cross * (p_new - p);
        v_cross = v + frac_cross * (v_new - v);
        t_cross = t - dt + frac_cross * dt;
      }
      p_cross.x() = x_hit;

      StrikeTarget out;
      out.p_ball = p_cross;
      out.v_ball = v_cross;
      out.t_strike = t_cross;
      out.num_bounces = bounces;
      out.valid = true;
      return out;
    }

    p = p_new;
    v = v_new;
  }

  StrikeTarget out;
  out.p_ball = p;
  out.v_ball = v;
  out.t_strike = t;
  out.num_bounces = bounces;
  out.valid = false;
  return out;
}

std::vector<Eigen::Vector3d> BallTrajectoryPredictor::sampleFuture(
  const Eigen::Vector3d & p0,
  const Eigen::Vector3d & v0,
  double horizon_s,
  int sample_stride) const
{
  const double dt = config_.dt_integrate;
  const int max_steps = static_cast<int>(std::max(0.0, horizon_s) / dt);
  const int stride = std::max(1, sample_stride);

  Eigen::Vector3d p = p0;
  Eigen::Vector3d v = v0;
  std::vector<Eigen::Vector3d> points;
  points.push_back(p);

  for (int step = 0; step < max_steps; ++step) {
    const Eigen::Vector3d a = flightAcceleration(v);
    Eigen::Vector3d p_new = p + v * dt + 0.5 * a * dt * dt;
    Eigen::Vector3d v_new = v + a * dt;

    if (p_new.z() < physics_.radius && v_new.z() < 0.0) {
      if (isOnTable(p_new)) {
        const double dz = p.z() - p_new.z();
        double frac = (dz > 1e-9) ? ((p.z() - physics_.radius) / dz) : 0.5;
        frac = std::max(0.0, std::min(1.0, frac));

        Eigen::Vector3d p_bounce = p + frac * (p_new - p);
        p_bounce.z() = physics_.radius;
        const Eigen::Vector3d v_at_bounce = v + a * (frac * dt);
        const Eigen::Vector3d v_post = applyBounce(v_at_bounce);

        const double remaining_dt = (1.0 - frac) * dt;
        const Eigen::Vector3d a_post = flightAcceleration(v_post);
        p_new = p_bounce + v_post * remaining_dt + 0.5 * a_post * remaining_dt * remaining_dt;
        v_new = v_post + a_post * remaining_dt;
      } else {
        break;
      }
    }

    p = p_new;
    v = v_new;

    if ((step + 1) % stride == 0) {
      points.push_back(p);
    }

    if (p.z() < -0.1 || p.x() < -0.6 || p.x() > table_.length + 0.6) {
      break;
    }
  }

  if ((points.back() - p).norm() > 1e-9) {
    points.push_back(p);
  }
  return points;
}

std::vector<Eigen::Vector3d> BallTrajectoryPredictor::sampleIncomingUntilFirstP1Bounce(
  const Eigen::Vector3d & p0,
  const Eigen::Vector3d & v0,
  double horizon_s,
  int sample_stride) const
{
  // Pre-bounce trajectory: a SINGLE smooth incoming arc, stopping at the
  // first contact with the table surface on the P1 half.  No bounce
  // simulation, no second arc.

  const double dt = config_.dt_integrate;
  const int max_steps = static_cast<int>(std::max(0.0, horizon_s) / dt);
  const int stride = std::max(1, sample_stride);

  Eigen::Vector3d p = p0;
  Eigen::Vector3d v = v0;
  std::vector<Eigen::Vector3d> points;
  points.push_back(p);

  for (int step = 0; step < max_steps; ++step) {
    const Eigen::Vector3d a = flightAcceleration(v);
    Eigen::Vector3d p_new = p + v * dt + 0.5 * a * dt * dt;
    Eigen::Vector3d v_new = v + a * dt;

    // Hard out-of-scene guards: stop before sending absurd points.
    if (p_new.z() < -0.05 || p_new.z() > 1.5) {
      break;
    }
    if (p_new.x() < -0.7 || p_new.x() > 3.5) {
      break;
    }
    if (p_new.y() < -1.7 || p_new.y() > 0.2) {
      break;
    }

    // First-contact detection: ball center crosses z = physics_.radius
    // while descending.  We do NOT simulate a bounce -- we just locate
    // the contact and stop.
    if (p_new.z() <= physics_.radius && v_new.z() < 0.0) {
      const double dz = p.z() - p_new.z();
      double frac = (dz > 1e-9) ? ((p.z() - physics_.radius) / dz) : 0.5;
      frac = std::max(0.0, std::min(1.0, frac));
      Eigen::Vector3d p_contact = p + frac * (p_new - p);
      p_contact.z() = physics_.radius;

      // Append the contact point (still subject to step-end bounds checks).
      if (p_contact.z() >= -0.05 && p_contact.z() <= 1.5 &&
          p_contact.x() >= -0.7 && p_contact.x() <= 3.5 &&
          p_contact.y() >= -1.7 && p_contact.y() <= 0.2) {
        points.push_back(p_contact);
      }
      // Stop either way -- we hit the table.
      break;
    }

    p = p_new;
    v = v_new;

    if ((step + 1) % stride == 0) {
      points.push_back(p);
    }
  }

  return points;
}

}  // namespace trajectory
