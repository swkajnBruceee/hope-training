#include "trajectory_arm_source.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace a3_deploy::control {

bool TrajectoryArmSource::Load(const std::string& path, std::string& error) {
  if (!trajectory_.Load(path, error)) return false;
  std::array<double, 14> first_dq{};
  std::array<double, 14> final_dq{};
  if (!trajectory_.Sample(0.0, first_q_, first_dq)) {
    error = "failed to read the first trajectory frame";
    return false;
  }
  if (!trajectory_.Sample(trajectory_.DurationS(), final_q_, final_dq)) {
    error = "failed to read the final trajectory frame";
    return false;
  }
  return true;
}

bool TrajectoryArmSource::RequestReplay() noexcept {
  // Keep at most one request. A request received during playback is consumed
  // immediately after the current cycle completes.
  return !replay_requested_.exchange(true, std::memory_order_acq_rel);
}

void TrajectoryArmSource::MinimumJerkToFirst(
    const std::array<double, 14>& from,
    const std::array<double, 14>& first,
    double elapsed_s, double duration_s,
    std::array<double, 14>& q,
    std::array<double, 14>& qd) noexcept {
  const double u = std::clamp(elapsed_s / duration_s, 0.0, 1.0);
  // Quintic minimum-jerk blend. Position, velocity and acceleration are all
  // continuous at both ends, avoiding the torque pulse produced by the old
  // cubic blend when a phase changed.
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;
  const double s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
  const double ds_dt =
      (30.0 * u2 - 60.0 * u3 + 30.0 * u4) / duration_s;
  for (std::size_t i = 0; i < 14; ++i) {
    q[i] = (1.0 - s) * from[i] + s * first[i];
    qd[i] = ds_dt * (first[i] - from[i]);
  }
}

bool TrajectoryArmSource::Update(const robot_io::RobotState& state, double time_s,
                                 control::ArmTarget& target) noexcept {
  if (trajectory_.Empty() || !std::isfinite(time_s) ||
      state.q.size() < 5 + 14) {
    return false;
  }

  if (!timeline_started_) {
    timeline_started_ = true;
    timeline_origin_s_ = time_s;
    for (std::size_t i = 0; i < 14; ++i) {
      startup_from_q_[i] = state.q[5 + static_cast<int>(i)];
    }
  }

  if (!playback_active_.load(std::memory_order_acquire) &&
      replay_requested_.exchange(false, std::memory_order_acq_rel)) {
    // Replay starts from the ready-pose hold; it does not repeat the initial
    // measured-pose approach.
    timeline_origin_s_ = time_s - kStartupTransitionS;
    playback_active_.store(true, std::memory_order_release);
    std::cout << "model3396: replay started" << std::endl;
  }

  const double elapsed_s = std::max(0.0, time_s - timeline_origin_s_);

  if (elapsed_s < kStartupTransitionS) {
    MinimumJerkToFirst(startup_from_q_, first_q_, elapsed_s,
                       kStartupTransitionS, target.q, target.dq);
    return true;
  }

  const double motion_elapsed_s = elapsed_s - kStartupTransitionS;
  const double trajectory_s = trajectory_.DurationS();
  const double cycle_s =
      kHoldS + trajectory_s + kReturnTransitionS + kHoldS;
  if (!std::isfinite(trajectory_s) || trajectory_s <= 0.0 ||
      !std::isfinite(cycle_s) || cycle_s <= 0.0) {
    return false;
  }

  if (!playback_active_.load(std::memory_order_acquire)) {
    target.q = first_q_;
    target.dq.fill(0.0);
    return true;
  }

  if (motion_elapsed_s >= cycle_s) {
    playback_active_.store(false, std::memory_order_release);
    target.q = first_q_;
    target.dq.fill(0.0);
    std::cout << "model3396: replay completed; waiting for SPACE"
              << std::endl;
    return true;
  }

  const double phase_s = motion_elapsed_s;

  if (phase_s < kHoldS) {
    target.q = first_q_;
    target.dq.fill(0.0);
    return true;
  }

  const double trajectory_start_s = kHoldS;
  const double return_start_s = kHoldS + trajectory_s;
  const double hold_end_s = return_start_s + kReturnTransitionS;
  if (phase_s < return_start_s) {
    return trajectory_.Sample(
        phase_s - trajectory_start_s, target.q, target.dq);
  }

  if (phase_s < hold_end_s) {
    // Never restart the return segment from the measured pose. At the end of
    // a fast swing the real arm can lag the commanded trajectory slightly.
    // Replacing q_des with that measured pose for one frame unloads all arm
    // joints at once and is perceived as a whole-arm shake. Continue from the
    // immutable final command instead, so q_des and dq_des remain continuous.
    MinimumJerkToFirst(final_q_, first_q_, phase_s - return_start_s,
                       kReturnTransitionS, target.q, target.dq);
    return true;
  }

  target.q = first_q_;
  target.dq.fill(0.0);
  return true;
}

}  // namespace a3_deploy::control
