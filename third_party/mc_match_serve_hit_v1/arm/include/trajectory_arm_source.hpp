#pragma once

#include "control_source.hpp"
#include "robot_io/robot_io_backend.hpp"
#include "upper_trajectory.hpp"

#include <array>
#include <atomic>
#include <string>

namespace a3_deploy::control {

class TrajectoryArmSource final {
 public:
  bool Load(const std::string& path, std::string& error);
  bool RequestReplay() noexcept;
  bool Update(const robot_io::RobotState& state, double time_s,
              control::ArmTarget& target) noexcept;

 private:
  static constexpr double kStartupTransitionS = 2.0;
  static constexpr double kHoldS = 0.5;
  static constexpr double kReturnTransitionS = 2.0;

  static void MinimumJerkToFirst(const std::array<double, 14>& from,
                                 const std::array<double, 14>& first,
                                 double elapsed_s, double duration_s,
                                 std::array<double, 14>& q,
                                 std::array<double, 14>& qd) noexcept;

  UpperTrajectory trajectory_;
  bool timeline_started_{false};
  double timeline_origin_s_{0.0};
  std::array<double, 14> startup_from_q_{};
  std::array<double, 14> first_q_{};
  std::array<double, 14> final_q_{};
  std::atomic<bool> playback_active_{true};
  std::atomic<bool> replay_requested_{false};
};

}  // namespace a3_deploy::control
