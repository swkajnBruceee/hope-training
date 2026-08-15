#ifndef SOLVER_LANDING_ERROR_CORRECTOR_H
#define SOLVER_LANDING_ERROR_CORRECTOR_H

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

namespace solver {

struct LandingFeedbackSample {
  Eigen::Vector3d hit_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d ball_velocity_incoming = Eigen::Vector3d::Zero();
  Eigen::Vector3d ball_velocity_outgoing = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_land = Eigen::Vector3d::Zero();
  Eigen::Vector3d actual_landing = Eigen::Vector3d::Zero();
  double return_flight_time = 0.0;
};

struct LandingErrorCorrectionConfig {
  bool enabled = true;
  std::size_t max_history = 200;
  std::size_t min_samples = 3;
  double distance_sigma = 1.0;
  double correction_gain = 0.65;
  double max_delta_v = 1.5;
  double min_weight_sum = 1e-6;
};

struct LandingErrorCorrection {
  Eigen::Vector3d estimated_landing_error = Eigen::Vector3d::Zero();
  Eigen::Vector3d delta_v_out = Eigen::Vector3d::Zero();
  double weight_sum = 0.0;
  std::size_t sample_count = 0;
  bool active = false;
};

class LandingErrorCorrector {
 public:
  explicit LandingErrorCorrector(const LandingErrorCorrectionConfig & config = {});

  void addSample(const LandingFeedbackSample & sample);

  LandingErrorCorrection correctionFor(
    const Eigen::Vector3d & hit_position,
    const Eigen::Vector3d & ball_velocity_incoming,
    const Eigen::Vector3d & target_land,
    double return_flight_time) const;

  std::size_t historySize() const { return history_.size(); }

 private:
  static bool finiteVector(const Eigen::Vector3d & v);
  double featureDistance(
    const LandingFeedbackSample & sample,
    const Eigen::Vector3d & hit_position,
    const Eigen::Vector3d & ball_velocity_incoming,
    const Eigen::Vector3d & target_land) const;

  LandingErrorCorrectionConfig config_;
  std::vector<LandingFeedbackSample> history_;
};

}  // namespace solver

#endif  // SOLVER_LANDING_ERROR_CORRECTOR_H
