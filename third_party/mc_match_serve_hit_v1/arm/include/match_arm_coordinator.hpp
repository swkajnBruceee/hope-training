#pragma once

#include "control_source.hpp"
#include "ik_point_arm_source.hpp"
#include "upper_trajectory.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace a3_deploy::control {

// Competition arm state machine:
// automatic serve -> right-arm idle-attractor transition -> external 10-D
// strike -> optional recovery. A valid strike may preempt either the initial
// ready transition or a post-strike recovery; the ready pose is never a gate.
// SPACE queues exactly one new serve and never interrupts a committed strike.
//
// The left arm follows the complete serve trajectory, then independently
// returns to the seven joint positions captured from the first valid robot
// state. It remains enabled and held there during rally play. Right-arm rally
// readiness and strike admission never wait for the left-arm return to finish.
class MatchArmCoordinator final {
 public:
  enum class State : std::uint8_t {
    kInitialServeEntry,
    kServeEntryHold,
    kServing,
    kServeToRally,
    kRallyReady,
    kStriking,
    kRecoverToRally,
  };

  bool Load(const std::string& serve_path,
            const std::string& rally_pose_path,
            const std::string& hit_config_path,
            const std::string& robot_xml,
            std::string& error);

  // Thread-safe terminal request. At most one future serve is queued.
  bool RequestServe() noexcept;
  bool CanAcceptStrikeGoal() const noexcept;
  bool SetStrikeGoal(const ArmGoal& goal) noexcept;
  bool Update(const robot_io::RobotState& state, double time_s,
              ArmTarget& target) noexcept;

  State CurrentState() const noexcept { return state_; }
  const char* StateName() const noexcept;
  IkPointArmSource* HitSource() noexcept { return hit_source_.get(); }

 private:
  static void MinimumJerk(const std::array<double, 14>& q0,
                          const std::array<double, 14>& q1,
                          double elapsed_s, double duration_s,
                          std::array<double, 14>& q,
                          std::array<double, 14>& dq) noexcept;
  static void MinimumJerkRight(const std::array<double, 14>& q0,
                               const std::array<double, 7>& q1_right,
                               double elapsed_s, double duration_s,
                               std::array<double, 14>& q,
                               std::array<double, 14>& dq) noexcept;
  static void MinimumJerkLeft(const std::array<double, 7>& q0,
                              const std::array<double, 7>& q1_left,
                              double elapsed_s, double duration_s,
                              std::array<double, 14>& q,
                              std::array<double, 14>& dq) noexcept;
  bool LoadRallyPose(const std::string& path, std::string& error);
  void BeginServeEntry(const std::array<double, 14>& from,
                       double time_s, bool initial) noexcept;
  void BeginRightTransition(State next,
                            const std::array<double, 14>& from,
                            double time_s,
                            double nominal_duration_s) noexcept;
  double RightTransitionDuration(const std::array<double, 14>& from,
                                 double nominal_duration_s) const noexcept;
  void BeginLeftReturn(const std::array<double, 14>& from,
                       double time_s) noexcept;
  void ApplyLeftReturn(double time_s, ArmTarget& target) noexcept;
  double LeftReturnDuration(const std::array<double, 14>& from) const noexcept;
  void FillRightReady(ArmTarget& target) const noexcept;
  static std::array<double, 14> MeasuredArms(
      const robot_io::RobotState& state) noexcept;

  UpperTrajectory serve_trajectory_;
  std::unique_ptr<IkPointArmSource> hit_source_;
  std::array<double, 14> serve_first_q_{};
  std::array<double, 14> serve_final_q_{};
  std::array<double, 7> rally_right_q_{};
  std::array<double, 7> startup_left_q_{};
  std::array<double, 7> left_return_from_q_{};
  std::array<double, 14> transition_from_q_{};
  std::array<double, 14> last_command_q_{};
  std::array<double, 14> last_command_dq_{};
  bool initialized_{false};
  bool left_return_active_{false};
  bool left_return_complete_logged_{false};
  State state_{State::kInitialServeEntry};
  double phase_origin_s_{0.0};
  double transition_duration_s_{0.0};
  double left_return_origin_s_{0.0};
  double left_return_duration_s_{0.60};
  // Nominal right-arm transition is deliberately below 0.5 s. It is extended
  // only when the minimum-jerk peak speed would exceed the configured bound.
  double serve_to_rally_s_{0.45};
  double hit_to_rally_s_{0.45};
  double rally_to_serve_s_{0.65};
  double initial_serve_entry_s_{2.0};
  double serve_entry_hold_s_{0.15};
  double right_ready_max_velocity_rad_s_{4.5};
  double left_return_nominal_s_{0.60};
  double left_return_max_s_{0.70};
  double left_return_max_velocity_rad_s_{6.0};
  std::atomic<bool> serve_requested_{false};
  std::uint64_t strike_count_at_start_{0};
  std::uint64_t reject_count_at_start_{0};
};

}  // namespace a3_deploy::control
