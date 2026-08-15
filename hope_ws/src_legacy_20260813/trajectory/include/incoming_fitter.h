#ifndef TRAJECTORY_INCOMING_FITTER_H
#define TRAJECTORY_INCOMING_FITTER_H

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "constants.h"

namespace trajectory {

/**
 * One raw /ball/point observation, tagged with its ROS timestamp (seconds).
 *
 * We keep these from the moment we decide the highest point has passed
 * until a stable physical prediction is generated or a real P1 bounce is
 * observed.
 */
struct TimedBallSample {
  double t = 0.0;
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
};

/**
 * Result of predicting an incoming arc from TimedBallSample observations.
 *
 * The model is the HOPE flight ODE:
 *
 *     a(v) = -k * |v| * v + g
 *
 * with p0 fixed at the earliest/highest sample and v0 estimated from the
 * early measured samples.  We forward-integrate the model from the latest
 * observation's timestamp to
 * the predicted first P1 contact so that ``predicted_points`` always
 * continues the observed history seamlessly.
 */
struct IncomingFitResult {
  bool ok = false;
  std::string reason;

  // Reference state (apex in the current schema: p_ref == apex position).
  Eigen::Vector3d p_ref{Eigen::Vector3d::Zero()};
  Eigen::Vector3d v_ref{Eigen::Vector3d::Zero()};
  double t_ref = 0.0;

  // Real observed history (highest -> latest observed).
  std::vector<Eigen::Vector3d> observed_points;

  // Forward continuation from latest observation to the predicted first
  // P1 contact (still part of the SAME incoming arc; no bounce).
  std::vector<Eigen::Vector3d> predicted_points;

  // True if a contact with the P1 table surface was found.
  bool contact_predicted = false;
  Eigen::Vector3d contact{Eigen::Vector3d::Zero()};

  // Root-mean-square residual across all observed samples (m).
  double rms_error = 0.0;

  // Number of samples that participated in the prediction.
  std::size_t num_used = 0;
};

/**
 * Predict a single incoming arc (no rotation, HOPE flight model) from a list
 * of highest-after /ball/point samples.
 *
 * Algorithm:
 *   1. Reference state is fixed at the *front* of ``samples`` (the apex).
 *      p_ref == samples.front().p  (held exactly).
 *      Estimate v_ref == (vx0, vy0, vz0) from the first few samples.
 *   2. Initial guess: central difference over the first 3-5 samples
 *      (or, if not enough, the linear regression slope of the whole
 *      list for each axis independently).
 *   3. Score the fixed physical prediction against observed samples for
 *      diagnostics only; do not optimize or reshape the incoming curve.
 *   4. Forward-integrate the same ODE from the *latest* sample to the
 *      first P1 contact to produce ``predicted_points``.
 *
 * Inputs that are too few, NaN/Inf, or whose initial velocity is invalid produce
 * ``ok == false`` with a ``reason``.  The caller is expected to keep
 * the previous frame's red trajectory in that case.
 */
class IncomingFitter {
 public:
  IncomingFitter(
    const common::BallPhysics & physics,
    const common::PlannerConfig & config,
    const common::TableParams & table);

  IncomingFitResult fitAndPredict(
    const std::vector<TimedBallSample> & samples,
    double horizon_s,
    int sample_stride) const;

 private:
  common::BallPhysics physics_;
  common::PlannerConfig config_;
  common::TableParams table_;
};

}  // namespace trajectory

#endif  // TRAJECTORY_INCOMING_FITTER_H
