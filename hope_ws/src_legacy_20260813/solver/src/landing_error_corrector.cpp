#include "landing_error_corrector.h"

#include <algorithm>
#include <cmath>

namespace solver {

LandingErrorCorrector::LandingErrorCorrector(const LandingErrorCorrectionConfig & config)
: config_(config)
{
}

void LandingErrorCorrector::addSample(const LandingFeedbackSample & sample)
{
  if (!config_.enabled) {
    return;
  }
  if (!finiteVector(sample.hit_position) ||
      !finiteVector(sample.ball_velocity_incoming) ||
      !finiteVector(sample.ball_velocity_outgoing) ||
      !finiteVector(sample.target_land) ||
      !finiteVector(sample.actual_landing) ||
      !std::isfinite(sample.return_flight_time) ||
      sample.return_flight_time <= 1e-6) {
    return;
  }
  history_.push_back(sample);
  if (history_.size() > config_.max_history) {
    const auto overflow = static_cast<std::vector<LandingFeedbackSample>::difference_type>(
      history_.size() - config_.max_history);
    history_.erase(history_.begin(), history_.begin() + overflow);
  }
}

LandingErrorCorrection LandingErrorCorrector::correctionFor(
  const Eigen::Vector3d & hit_position,
  const Eigen::Vector3d & ball_velocity_incoming,
  const Eigen::Vector3d & target_land,
  double return_flight_time) const
{
  LandingErrorCorrection out;
  out.sample_count = history_.size();
  if (!config_.enabled ||
      history_.size() < config_.min_samples ||
      !finiteVector(hit_position) ||
      !finiteVector(ball_velocity_incoming) ||
      !finiteVector(target_land) ||
      !std::isfinite(return_flight_time) ||
      return_flight_time <= 1e-6) {
    return out;
  }

  Eigen::Vector3d weighted_error = Eigen::Vector3d::Zero();
  double weight_sum = 0.0;
  const double sigma = std::max(1e-6, config_.distance_sigma);
  for (const auto & sample : history_) {
    const double distance = featureDistance(sample, hit_position, ball_velocity_incoming, target_land);
    const double weight = std::exp(-0.5 * distance * distance / (sigma * sigma));
    const Eigen::Vector3d landing_error = sample.actual_landing - sample.target_land;
    weighted_error += weight * landing_error;
    weight_sum += weight;
  }

  out.weight_sum = weight_sum;
  if (weight_sum < config_.min_weight_sum) {
    return out;
  }

  out.estimated_landing_error = weighted_error / weight_sum;
  out.delta_v_out = -config_.correction_gain * out.estimated_landing_error / return_flight_time;
  out.delta_v_out.z() = 0.0;

  const double norm = out.delta_v_out.norm();
  if (norm > config_.max_delta_v && norm > 1e-9) {
    out.delta_v_out *= config_.max_delta_v / norm;
  }
  out.active = out.delta_v_out.norm() > 1e-9;
  return out;
}

bool LandingErrorCorrector::finiteVector(const Eigen::Vector3d & v)
{
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

double LandingErrorCorrector::featureDistance(
  const LandingFeedbackSample & sample,
  const Eigen::Vector3d & hit_position,
  const Eigen::Vector3d & ball_velocity_incoming,
  const Eigen::Vector3d & target_land) const
{
  const double hit_scale = 0.35;
  const double velocity_scale = 4.0;
  const double target_scale = 0.55;
  const double d_hit = (sample.hit_position - hit_position).norm() / hit_scale;
  const double d_velocity = (sample.ball_velocity_incoming - ball_velocity_incoming).norm() / velocity_scale;
  const double d_target = (sample.target_land - target_land).norm() / target_scale;
  return std::sqrt(d_hit * d_hit + d_velocity * d_velocity + d_target * d_target);
}

}  // namespace solver
