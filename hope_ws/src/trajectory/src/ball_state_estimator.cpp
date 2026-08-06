#include "ball_state_estimator.h"

#include <stdexcept>

namespace trajectory {

namespace {

double polyfitValue(const std::vector<double> & t, const std::vector<double> & y, double t_eval, int order) {
  // Mimic np.polyfit (least squares) and evaluate at t_eval.
  // We solve a Vandermonde system: A * c = y, with A[i][k] = t[i]^(order-k).
  const int n = static_cast<int>(t.size());
  const int m = order + 1;

  // Build augmented matrix [A | y]
  std::vector<std::vector<double>> aug(m, std::vector<double>(m + 1, 0.0));
  for (int i = 0; i < n; ++i) {
    double ti = t[i];
    double ti_pow = 1.0;
    for (int k = 0; k < m; ++k) {
      aug[k][m] += ti_pow * y[i];
      for (int j = 0; j < m; ++j) {
        aug[k][j] += ti_pow * std::pow(ti, m - 1 - j);  // placeholder, unused directly
      }
      // Simplify: build A^T A and A^T y directly
    }
  }
  // Re-build A^T A explicitly (cleaner):
  std::vector<std::vector<double>> ATA(m, std::vector<double>(m, 0.0));
  std::vector<double> ATy(m, 0.0);
  for (int i = 0; i < n; ++i) {
    double ti = t[i];
    std::vector<double> row(m);
    for (int k = 0; k < m; ++k) {
      row[k] = std::pow(ti, order - k);
    }
    for (int j = 0; j < m; ++j) {
      for (int k = 0; k < m; ++k) {
        ATA[j][k] += row[j] * row[k];
      }
      ATy[j] += row[j] * y[i];
    }
  }
  // Solve via Gaussian elimination.
  for (int i = 0; i < m; ++i) {
    int pivot = i;
    for (int r = i + 1; r < m; ++r) {
      if (std::abs(ATA[r][i]) > std::abs(ATA[pivot][i])) {
        pivot = r;
      }
    }
    if (pivot != i) {
      std::swap(ATA[i], ATA[pivot]);
      std::swap(ATy[i], ATy[pivot]);
    }
    for (int r = i + 1; r < m; ++r) {
      double factor = ATA[r][i] / ATA[i][i];
      for (int c = i; c < m; ++c) {
        ATA[r][c] -= factor * ATA[i][c];
      }
      ATy[r] -= factor * ATy[i];
    }
  }
  std::vector<double> c(m);
  for (int i = m - 1; i >= 0; --i) {
    double s = ATy[i];
    for (int j = i + 1; j < m; ++j) {
      s -= ATA[i][j] * c[j];
    }
    c[i] = s / ATA[i][i];
  }
  // Evaluate at t_eval: y = sum_k c[k] * t_eval^(order-k)
  double out = 0.0;
  for (int k = 0; k < m; ++k) {
    out += c[k] * std::pow(t_eval, order - k);
  }
  return out;
}

}  // namespace

BallStateEstimator::BallStateEstimator(const common::PlannerConfig & config)
: config_(config), z_hist_(3, std::numeric_limits<double>::quiet_NaN())
{
}

void BallStateEstimator::push(double t, const Eigen::Vector3d & p) {
  // Update z history ring buffer (still useful for callers that want
  // bounceDetected() / inspect the recent z profile).  We deliberately do
  // NOT auto-reset on a generic "three-point bounce" pattern here: the
  // overlay node owns the state machine, and an unconditional internal
  // reset on z crossing bounce_z_tol would wipe the buffer during
  // legitimate low flights over the net.
  z_hist_[0] = z_hist_[1];
  z_hist_[1] = z_hist_[2];
  z_hist_[2] = p.z();
  bounce_detected_ = false;

  t_buffer_.push_back(t);
  p_buffer_.push_back(p);

  while (static_cast<int>(t_buffer_.size()) > config_.fit_window) {
    t_buffer_.erase(t_buffer_.begin());
    p_buffer_.erase(p_buffer_.begin());
  }
  while (t_buffer_.size() >= 2 && (t_buffer_.back() - t_buffer_.front()) > config_.fit_window_s) {
    t_buffer_.erase(t_buffer_.begin());
    p_buffer_.erase(p_buffer_.begin());
  }
}

void BallStateEstimator::reset() {
  t_buffer_.clear();
  p_buffer_.clear();
}

bool BallStateEstimator::ready() const {
  return static_cast<int>(t_buffer_.size()) >= 6;
}

bool BallStateEstimator::bounceDetected() const {
  return bounce_detected_;
}

BallStateEstimator::Estimate BallStateEstimator::estimate() const {
  if (!ready()) {
    throw std::runtime_error("Need >= 6 samples, have " + std::to_string(t_buffer_.size()));
  }
  double t_ref = t_buffer_.back();
  std::vector<double> t_norm(t_buffer_.size());
  for (size_t i = 0; i < t_buffer_.size(); ++i) {
    t_norm[i] = t_buffer_[i] - t_ref;
  }
  Estimate out;
  out.p = Eigen::Vector3d::Zero();
  out.v = Eigen::Vector3d::Zero();
  out.t = t_ref;
  for (int axis = 0; axis < 3; ++axis) {
    std::vector<double> y(t_buffer_.size());
    for (size_t i = 0; i < t_buffer_.size(); ++i) {
      y[i] = p_buffer_[i](axis);
    }
    out.p(axis) = polyfitValue(t_norm, y, 0.0, config_.poly_order);
    out.v(axis) = polyfitValue(t_norm, y, 1e-6, config_.poly_order);
    // Linearize: dv = (y(t+h) - y(t)) / h for numerical derivative w.r.t. t_norm.
    // However, np.polyfit[1] (a1 in [a2,a1,a0]) is exactly derivative at t=0 for the polynomial.
    // For better fidelity match Python, recompute using the polynomial derivative:
    // y(t) = a2*t^2 + a1*t + a0; dy/dt(t=0) = a1
    // np.polyfit returns [a2, a1, a0] in descending power order; a1 is index 1.
    // We recover a1 by fitting again and reading coefficient 1.
    // For simplicity (and matching the original 1e-6 finite-diff used above), use:
    // Reconstruct: fit returns y(t), so v = y(eps) - y(0) / eps.
    // That's what we already did above.
    out.v(axis) = (polyfitValue(t_norm, y, 1e-6, config_.poly_order) -
                    polyfitValue(t_norm, y, 0.0, config_.poly_order)) / 1e-6;
  }
  return out;
}

}  // namespace trajectory
