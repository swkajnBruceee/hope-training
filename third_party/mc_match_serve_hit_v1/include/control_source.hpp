#pragma once

#include "robot_io/robot_io_backend.hpp"

#include <array>
#include <cstdint>

namespace a3_deploy::control {

struct LegTarget {
  std::array<double, 12> q{};
};

struct ArmTarget {
  std::array<double, 14> q{};
  std::array<double, 14> dq{};
  // Optional source-specific gains. Existing trajectory mode leaves this
  // false, preserving mc(1)'s command composition exactly.
  bool has_arm_gains{false};
  std::array<double, 14> kp{};
  std::array<double, 14> kd{};
};

// Exact low-level representation of the high-level 10-D racket state:
// position[3], linear velocity[3], face normal[3], time-to-strike[1].
// Metadata is kept outside those ten task dimensions.
struct ArmGoal {
  bool valid{false};
  bool has_cartesian_position{false};
  std::array<double, 3> position_m{};
  bool has_cartesian_linear_velocity{false};
  std::array<double, 3> linear_velocity_mps{};
  bool has_racket_normal{false};
  std::array<double, 3> racket_normal{1.0, 0.0, 0.0};
  bool has_time_to_strike{false};
  double time_to_strike_s{0.0};

  // Optional compatibility fields retained only for the IK implementation.
  bool has_orientation{false};
  std::array<double, 4> orientation_wxyz{1.0, 0.0, 0.0, 0.0};
  bool has_cartesian_angular_velocity{false};
  std::array<double, 3> angular_velocity_rps{};
  double source_time_to_strike_s{0.0};
  double transport_compensation_s{0.0};
  double actuation_lead_s{0.0};
  int swing_type{0};
  std::int64_t source_stamp_ns{0};
  std::int64_t source_deadline_ns{0};
  double local_receipt_age_s{0.0};
  std::uint64_t sequence{0};
  float confidence{1.0F};
};

class IArmControlSource {
 public:
  virtual ~IArmControlSource() = default;
  virtual bool SetGoal(const ArmGoal&) noexcept { return false; }
  virtual void Reset() noexcept = 0;
  virtual bool Update(const robot_io::RobotState& state, double time_s,
                      ArmTarget& target) noexcept = 0;
  virtual double UpdateHz() const noexcept = 0;
};

}  // namespace a3_deploy::control
