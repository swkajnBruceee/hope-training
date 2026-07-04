#include "incoming_fitter.h"

#include <algorithm>
#include <cmath>

namespace trajectory {

namespace {

inline Eigen::Vector3d accel(const common::BallPhysics & phys, const Eigen::Vector3d & v) {
  const double speed = v.norm();
  return -phys.k * speed * v + phys.g;
}

// Integrate (p, v) forward by dt (single explicit Euler step, no bounce).
inline void eulerStep(
  const common::BallPhysics & phys, const double dt,
  Eigen::Vector3d & p, Eigen::Vector3d & v)
{
  const Eigen::Vector3d a = accel(phys, v);
  p = p + v * dt + 0.5 * a * dt * dt;
  v = v + a * dt;
}

// Reusable forward integrator that records the full state every step.
// Used to score the fixed physical prediction against observed samples.
struct IntegratedPoint {
  double t;
  Eigen::Vector3d p;
  Eigen::Vector3d v;
};

std::vector<IntegratedPoint> integrateFine(
  const common::BallPhysics & phys, const double dt,
  const Eigen::Vector3d & p0, const Eigen::Vector3d & v0,
  const double T)
{
  std::vector<IntegratedPoint> out;
  if (T <= 0.0) {
    out.push_back({0.0, p0, v0});
    return out;
  }
  Eigen::Vector3d p = p0;
  Eigen::Vector3d v = v0;
  double t = 0.0;
  out.push_back({t, p, v});
  const int n_steps = std::max(1, static_cast<int>(std::ceil(T / dt)));
  for (int i = 0; i < n_steps; ++i) {
    eulerStep(phys, dt, p, v);
    t += dt;
    out.push_back({t, p, v});
  }
  return out;
}

// Initial velocity guess:
//   - prefer a central difference over the first 3..5 samples;
//   - fall back to a linear least-squares slope over the whole list.
Eigen::Vector3d initialVelocityGuess(const std::vector<TimedBallSample> & s) {
  const std::size_t n = s.size();
  if (n >= 3) {
    const std::size_t i0 = 0;
    const std::size_t i2 = std::min<std::size_t>(n - 1, 4);
    const double dt = s[i2].t - s[i0].t;
    if (std::abs(dt) > 1e-6) {
      return (s[i2].p - s[i0].p) / dt;
    }
  }
  // Linear regression: p(axis) = a * t + b, return a per axis.
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  if (n < 2) return v;
  double sum_t = 0.0, sum_tt = 0.0;
  for (const auto & si : s) { sum_t += si.t; sum_tt += si.t * si.t; }
  const double n_d = static_cast<double>(n);
  const double denom = n_d * sum_tt - sum_t * sum_t;
  if (std::abs(denom) < 1e-12) return v;
  for (int axis = 0; axis < 3; ++axis) {
    double sum_y = 0.0, sum_ty = 0.0;
    for (const auto & si : s) {
      const double y = si.p(axis);
      sum_y += y;
      sum_ty += si.t * y;
    }
    v(axis) = (n_d * sum_ty - sum_t * sum_y) / denom;
  }
  return v;
}

Eigen::Vector3d recentVelocityGuess(const std::vector<TimedBallSample> & s) {
  const std::size_t n = s.size();
  if (n < 2) return Eigen::Vector3d::Zero();

  const std::size_t i0 = (n > 8) ? (n - 8) : 0;
  const double t0 = s[i0].t;
  double sum_t = 0.0;
  double sum_tt = 0.0;
  const double n_d = static_cast<double>(n - i0);
  for (std::size_t i = i0; i < n; ++i) {
    const double t = s[i].t - t0;
    sum_t += t;
    sum_tt += t * t;
  }
  const double denom = n_d * sum_tt - sum_t * sum_t;
  if (std::abs(denom) < 1e-12) {
    const double dt = s.back().t - s[n - 2].t;
    if (std::abs(dt) > 1e-6) {
      return (s.back().p - s[n - 2].p) / dt;
    }
    return Eigen::Vector3d::Zero();
  }

  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  for (int axis = 0; axis < 3; ++axis) {
    double sum_y = 0.0;
    double sum_ty = 0.0;
    for (std::size_t i = i0; i < n; ++i) {
      const double t = s[i].t - t0;
      const double y = s[i].p(axis);
      sum_y += y;
      sum_ty += t * y;
    }
    v(axis) = (n_d * sum_ty - sum_t * sum_y) / denom;
  }
  return v;
}

}  // namespace

IncomingFitter::IncomingFitter(
  const common::BallPhysics & physics,
  const common::PlannerConfig & config,
  const common::TableParams & table)
: physics_(physics), config_(config), table_(table)
{
}

IncomingFitResult IncomingFitter::fitAndPredict(
  const std::vector<TimedBallSample> & samples,
  double horizon_s,
  int sample_stride) const
{
  IncomingFitResult out;

  const std::size_t n = samples.size();
  if (n < 3) {
    out.reason = "too_few_samples";
    return out;
  }

  // Reject any obviously bad samples (NaN/Inf or non-monotonic time).
  std::vector<TimedBallSample> good;
  good.reserve(n);
  good.push_back(samples.front());
  for (std::size_t i = 1; i < n; ++i) {
    const auto & s = samples[i];
    if (!std::isfinite(s.t) ||
        !std::isfinite(s.p.x()) || !std::isfinite(s.p.y()) || !std::isfinite(s.p.z())) {
      continue;
    }
    if (s.t <= good.back().t) {
      continue;
    }
    good.push_back(s);
  }
  if (good.size() < 3) {
    out.reason = "good_samples_below_3";
    return out;
  }

  const Eigen::Vector3d p_ref = good.front().p;
  const double t_ref = good.front().t;

  const Eigen::Vector3d v_ref = initialVelocityGuess(good);

  const double dt = config_.dt_integrate;
  const double t_total = good.back().t - t_ref;
  if (t_total < 1e-4) {
    out.reason = "t_span_too_small";
    return out;
  }

  if (!std::isfinite(v_ref.x()) || !std::isfinite(v_ref.y()) || !std::isfinite(v_ref.z()) ||
      v_ref.norm() > 15.0) {
    out.reason = "initial_velocity_invalid";
    return out;
  }

  // Score the fixed physical prediction against the samples for diagnostics
  // only.  Do not optimize v_ref here: the incoming overlay should be a
  // forward simulation from the early measured state, not a curve that gets
  // re-fitted and visibly reshaped every frame.
  double rms_error = 0.0;
  const double huber_k = 0.05;  // robust threshold (m)
  {
    const auto traj = integrateFine(physics_, dt, p_ref, v_ref, t_total);
    double sse = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 1; i < good.size(); ++i) {
      const double t_q = good[i].t - t_ref;
      if (t_q <= 0.0) continue;
      std::size_t k = 1;
      while (k < traj.size() && traj[k].t < t_q) ++k;
      if (k >= traj.size()) k = traj.size() - 1;
      Eigen::Vector3d p_pred = traj[k - 1].p;
      if (k > 0 && traj[k].t > traj[k - 1].t) {
        const double a = (t_q - traj[k - 1].t) / (traj[k].t - traj[k - 1].t);
        const double ac = std::max(0.0, std::min(1.0, a));
        p_pred = traj[k - 1].p + ac * (traj[k].p - traj[k - 1].p);
      }
      // Huber-style weight to keep one bad sample from dominating.
      Eigen::Vector3d r = p_pred - good[i].p;
      const double rn = r.norm();
      const double w = (rn < huber_k) ? 1.0 : (huber_k / std::max(rn, 1e-9));
      sse += w * r.squaredNorm();
      count++;
    }
    rms_error = (count > 0) ? std::sqrt(sse / static_cast<double>(count)) : 0.0;
  }

  // --- Build observed_points (highest -> latest) and predicted_points ----.
  // The observed red segment must be the real measured ball path.  Replaying
  // the fitted model here can visibly bend the already-observed line above
  // the recorded highest point when the fit is still under-constrained.
  const int stride = std::max(1, sample_stride);
  out.observed_points.reserve(good.size());
  for (const auto & sample : good) {
    out.observed_points.push_back(sample.p);
  }

  // --- Forward continuation from latest measured point. ------------------
  Eigen::Vector3d p_start = good.back().p;
  Eigen::Vector3d v_start = recentVelocityGuess(good);
  if (!std::isfinite(v_start.x()) || !std::isfinite(v_start.y()) || !std::isfinite(v_start.z()) ||
      v_start.norm() > 15.0) {
    v_start = v_ref;
  }

  // Forward continuation up to horizon_s, stopping at first P1 contact.
  // If horizon_s <= 0, fall back to a sane 1.2 s.
  const double T_pred = (horizon_s > 0.0) ? horizon_s : 1.2;
  const int max_steps = static_cast<int>(std::ceil(T_pred / dt));
  Eigen::Vector3d p = p_start;
  Eigen::Vector3d vv = v_start;
  double t_cur = 0.0;
  bool hit_contact = false;
  Eigen::Vector3d p_contact = Eigen::Vector3d::Zero();
  out.predicted_points.reserve(max_steps / stride + 2);
  out.predicted_points.push_back(p);

  for (int step = 0; step < max_steps; ++step) {
    const Eigen::Vector3d p_prev = p;
    eulerStep(physics_, dt, p, vv);
    t_cur += dt;

    // First-contact detection (ball center crosses z = physics_.radius while
    // descending).  Same criterion as BallTrajectoryPredictor.
    if (p.z() <= physics_.radius && vv.z() < 0.0) {
      // Sub-step linear interpolation to find exact z = physics_.radius crossing.
      const double dz = p_prev.z() - p.z();
      double frac = (std::abs(dz) > 1e-9) ? ((p_prev.z() - physics_.radius) / dz) : 0.5;
      frac = std::max(0.0, std::min(1.0, frac));
      p_contact = p_prev + frac * (p - p_prev);
      p_contact.z() = physics_.radius;

      // Accept the contact only if it lies on the P1 table surface.
      const bool on_p1 =
        p_contact.x() >= -physics_.radius && p_contact.x() <= table_.net_x + physics_.radius &&
        p_contact.y() >= -table_.width - physics_.radius && p_contact.y() <= physics_.radius;
      if (on_p1) {
        out.predicted_points.push_back(p_contact);
        out.contact = p_contact;
        out.contact_predicted = true;
        hit_contact = true;
      }
      break;
    }

    if ((step + 1) % stride == 0) {
      out.predicted_points.push_back(p);
    }

    // Stop if we wander far off-scene (sanity bound).
    if (p.z() < -0.05 || p.z() > 1.5 ||
        p.x() < -0.7 || p.x() > 3.5 ||
        p.y() < -1.7 || p.y() > 0.2) {
      break;
    }
  }

  if (!hit_contact) {
    // Make sure predicted_points is at least observed_points' tail.
    if (out.predicted_points.size() < 2) {
      out.predicted_points.push_back(good.back().p);
    }
  }

  out.p_ref = p_ref;
  out.v_ref = v_ref;
  out.t_ref = t_ref;
  out.rms_error = rms_error;
  out.num_used = good.size();
  out.ok = true;
  return out;
}

}  // namespace trajectory
