#include "ik_point_arm_source.hpp"

#include "math_utils.hpp"

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace a3_deploy::control {
namespace {

constexpr std::array<int, 7> kRightArmSdk = {12, 13, 14, 15, 16, 17, 18};
constexpr std::array<double, 7> kRightArmMin = {
    -2.87979, -2.61799, -2.79253, -0.959931,
    -2.79253, -1.62316, -1.62316};
constexpr std::array<double, 7> kRightArmMax = {
    2.87979, 0.0872665, 2.79253, 1.74533,
    2.79253, 1.62316, 1.62316};
constexpr std::array<double, 7> kRightArmKp = {
    // Match the A3's already-proven PD_STAND arm envelope.  The previous
    // learned-model gains (40/40/30/...) left a gravity-loaded shoulder over
    // 0.2 rad behind even after the trajectory had completed.
    // The shoulder-pitch actuator (joint 0) carries the racket and retained
    // about 0.05 rad static gravity error at the standard 200 gain.  Its
    // 400 gain remains below the documented 60 Nm actuator envelope at the
    // observed 0.106 rad peak error; the remaining joints stay at PD_STAND.
    400.0, 200.0, 100.0, 200.0, 100.0, 50.0, 50.0};
constexpr std::array<double, 7> kRightArmKd = {
    4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0};

// Accepted fixed-base strikes from the same A3 asset. Shoulder-yaw is the
// decisive branch separator: forehand is negative, backhand is positive.
constexpr std::array<double, 7> kForehandSeed = {
    -1.245926, -0.354023, -1.696661, 0.875103,
    -0.583579, 0.167723, -0.166605};
constexpr std::array<double, 7> kBackhandSeed = {
    -0.978463, -0.685783, 0.890102, 0.612371,
    0.732525, -0.319086, 0.255459};

constexpr std::array<double, 3> kRacketMountOffset = {
    0.210211399202899, 0.0320784994676765, 0.0320358706296689};
constexpr std::array<double, 3> kRacketMountNormal = {0.0, 1.0, 0.0};

constexpr double kJointLimitMarginRad = 0.005;
constexpr double kFiniteDifferenceRad = 1.0e-5;
constexpr double kOrientationWeightMPerRad = 0.25;
constexpr double kIkDamping = 0.015;
constexpr double kVelocityDamping = 0.025;
constexpr double kMaxIterationStepRad = 0.15;
constexpr int kMaxIterations = 140;
constexpr double kSolvedPositionToleranceM = 0.005;
constexpr double kSolvedNormalToleranceDeg = 2.0;
constexpr double kStrikeJointMarginRad = 0.04;
constexpr double kForehandMaxShoulderYaw = -0.20;
constexpr double kBackhandMinShoulderYaw = 0.20;
constexpr int kMaxRandomAttempts = 48;
// The base +X axis points from the robot toward the table. Keep the racket
// centre in front of this conservative torso protection plane throughout a
// live stroke, not merely at its endpoints. The old unchecked interpolation
// was observed at x ~= 0.135 m even though both endpoints were near 0.40 m.
// Conservative inflated torso front plane. The underlying MJCF torso shell
// is not parsed by RobotFK as collision geometry; this is its runtime
// equivalent with an additional body/racket margin.
constexpr double kRacketBodyProtectionPlaneX = 0.24;
// The impact phase must give the wrist enough time to reach the strike
// velocity with the configured 6 rad/s joint envelope and the
// hardware-side ~100 rad/s^2 acceleration limit. A 0.16 s window produced a
// 5.86 rad/s peak and ~293 rad/s^2 peak acceleration for typical 0.5 rad
// motions, which collided with both envelopes. 0.32 s keeps the same impact
// pose but cuts the peak acceleration by 4x and the peak speed by 2x.
constexpr double kMaximumImpactTransitionS = 0.32;
// Approximate maximum acceleration tolerated at the robot-side policy
// driver. The impact-transition sample check enforces this so the trajectory
// never asks the controller to deliver 200+ rad/s^2 peaks.
constexpr double kMaximumImpactAccelRadSSq = 95.0;
// A quintic rest-to-rest path reaches 1.875 * distance / duration at its
// maximum.  1.5 rad/s is deliberately below the raw 6 rad/s mathematical
// limit so the diagnostic reflects the loaded physical arm, not only IK.
constexpr double kConservativeTrackingJointSpeedRadS = 1.5;

bool Finite(double value) noexcept { return std::isfinite(value); }

template <std::size_t N>
bool ReadFiniteArray(const YAML::Node& node, std::array<double, N>& out,
                     const char* name, std::string& error) {
  if (!node || !node.IsSequence() || node.size() != N) {
    error = std::string(name) + " must contain exactly " +
            std::to_string(N) + " values";
    return false;
  }
  for (std::size_t i = 0; i < N; ++i) {
    out[i] = node[i].as<double>();
    if (!Finite(out[i])) {
      error = std::string(name) + " contains NaN or infinity";
      return false;
    }
  }
  return true;
}

bool Normalize3(std::array<double, 3>& value) noexcept {
  const double norm = std::sqrt(
      value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
  if (!Finite(norm) || norm < 1.0e-8) return false;
  for (double& component : value) component /= norm;
  return true;
}

double Norm3(const std::array<double, 3>& value) noexcept {
  return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                   value[2] * value[2]);
}

std::array<double, 3> Cross(const std::array<double, 3>& a,
                            const std::array<double, 3>& b) noexcept {
  return {
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
  };
}

double Dot(const std::array<double, 3>& a,
           const std::array<double, 3>& b) noexcept {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double PositionError(const std::array<double, 3>& a,
                     const std::array<double, 3>& b) noexcept {
  const std::array<double, 3> delta{
      a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  return Norm3(delta);
}

double ComputeNormalErrorDeg(const std::array<double, 3>& current,
                             const std::array<double, 3>& target) noexcept {
  return std::acos(std::clamp(Dot(current, target), -1.0, 1.0)) *
         180.0 / M_PI;
}

bool BranchCompatible(const std::array<double, 7>& q,
                      bool forehand) noexcept {
  return forehand ? q[2] <= kForehandMaxShoulderYaw
                  : q[2] >= kBackhandMinShoulderYaw;
}

double MinimumJointMargin(const std::array<double, 7>& q) noexcept {
  double margin = std::numeric_limits<double>::infinity();
  for (std::size_t joint = 0; joint < q.size(); ++joint) {
    margin = std::min(
        margin, std::min(q[joint] - kRightArmMin[joint],
                         kRightArmMax[joint] - q[joint]));
  }
  return margin;
}

struct JointSample {
  double q{0.0};
  double dq{0.0};
};

JointSample QuinticBoundary(double q0, double dq0, double q1, double dq1,
                            double duration_s, double elapsed_s) noexcept {
  const double duration = std::max(duration_s, 1.0e-6);
  const double u = std::clamp(elapsed_s / duration, 0.0, 1.0);
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;
  const double h00 = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5;
  const double h10 = u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5;
  const double h01 = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
  const double h11 = -4.0 * u3 + 7.0 * u4 - 3.0 * u5;
  const double dh00 = -30.0 * u2 + 60.0 * u3 - 30.0 * u4;
  const double dh10 = 1.0 - 18.0 * u2 + 32.0 * u3 - 15.0 * u4;
  const double dh01 = 30.0 * u2 - 60.0 * u3 + 30.0 * u4;
  const double dh11 = -12.0 * u2 + 28.0 * u3 - 15.0 * u4;
  return {
      h00 * q0 + h10 * duration * dq0 + h01 * q1 +
          h11 * duration * dq1,
      dh00 * q0 / duration + dh10 * dq0 + dh01 * q1 / duration +
          dh11 * dq1,
  };
}

}  // namespace

IkPointArmSource::IkPointArmSource(double update_hz) noexcept
    : update_hz_(Finite(update_hz) && update_hz > 0.0 ? update_hz : 50.0) {}

bool IkPointArmSource::LoadGoal(const std::string& path, std::string& error) {
  try {
    const YAML::Node root = YAML::LoadFile(path);
    const YAML::Node position =
        root["position_b_m"] ? root["position_b_m"] : root["position_m"];
    if (!ReadFiniteArray(position, target_position_b_m_, "position_b_m",
                         error)) {
      return false;
    }
    if (root["normal_b"]) {
      if (!ReadFiniteArray(root["normal_b"], target_normal_b_, "normal_b",
                           error) ||
          !Normalize3(target_normal_b_)) {
        error = "normal_b must be a non-zero finite vector";
        return false;
      }
    } else if (root["orientation_wxyz"]) {
      std::array<double, 4> orientation{};
      if (!ReadFiniteArray(root["orientation_wxyz"], orientation,
                           "orientation_wxyz", error)) {
        return false;
      }
      double norm = 0.0;
      for (double value : orientation) norm += value * value;
      norm = std::sqrt(norm);
      if (!Finite(norm) || norm < 1.0e-8) {
        error = "orientation_wxyz must be a non-zero finite quaternion";
        return false;
      }
      for (double& component : orientation) component /= norm;
      target_normal_b_ = quat_rotate_d(orientation, kRacketMountNormal);
      if (!Normalize3(target_normal_b_)) {
        error = "orientation_wxyz produced an invalid racket normal";
        return false;
      }
    } else {
      error = "one of normal_b or orientation_wxyz is required";
      return false;
    }

    if (root["linear_velocity_b_mps"] &&
        !ReadFiniteArray(root["linear_velocity_b_mps"],
                         target_velocity_b_mps_,
                         "linear_velocity_b_mps", error)) {
      return false;
    }
    if (root["strike_time_s"]) {
      strike_time_s_ = root["strike_time_s"].as<double>();
    } else if (root["time_to_strike_s"]) {
      strike_time_s_ = root["time_to_strike_s"].as<double>();
    } else if (root["transition_s"]) {
      // Backward-compatible name from the old static-pose planner.
      strike_time_s_ = root["transition_s"].as<double>();
    }
    requested_strike_time_s_ = strike_time_s_;
    if (root["follow_through_s"])
      follow_through_s_ = root["follow_through_s"].as<double>();
    if (root["follow_through_distance_m"])
      follow_through_distance_m_ =
          root["follow_through_distance_m"].as<double>();
    if (root["max_joint_velocity_rad_s"])
      max_joint_velocity_rad_s_ =
          root["max_joint_velocity_rad_s"].as<double>();
    if (root["external_min_strike_time_s"])
      external_min_strike_time_s_ =
          root["external_min_strike_time_s"].as<double>();
    if (root["external_ready_time_s"])
      external_ready_time_s_ = root["external_ready_time_s"].as<double>();
    if (root["external_ready_position_b_m"] &&
        !ReadFiniteArray(root["external_ready_position_b_m"],
                         external_ready_position_b_m_,
                         "external_ready_position_b_m", error)) {
      return false;
    }
    if (root["external_ready_normal_b"] &&
        (!ReadFiniteArray(root["external_ready_normal_b"],
                          external_ready_normal_b_,
                          "external_ready_normal_b", error) ||
         !Normalize3(external_ready_normal_b_))) {
      error = "external_ready_normal_b must be a non-zero finite vector";
      return false;
    }
    if (!Finite(strike_time_s_) || strike_time_s_ < 0.60 ||
        !Finite(external_min_strike_time_s_) ||
        external_min_strike_time_s_ < 0.0 ||
        external_min_strike_time_s_ > 3.0 ||
        !Finite(external_ready_time_s_) ||
        external_ready_time_s_ < 0.60 ||
        external_ready_time_s_ > 4.0 ||
        !Finite(follow_through_s_) || follow_through_s_ < 0.15 ||
        !Finite(follow_through_distance_m_) ||
        follow_through_distance_m_ < 0.03 ||
        !Finite(max_joint_velocity_rad_s_) ||
        max_joint_velocity_rad_s_ < 1.0 ||
        max_joint_velocity_rad_s_ > 10.0) {
      error =
          "strike/external/follow-through timing or velocity limit is invalid";
      return false;
    }

    if (root["random_targets"])
      random_targets_ = root["random_targets"].as<bool>();
    if (root["alternate_swings"])
      alternate_swings_ = root["alternate_swings"].as<bool>();
    if (root["target_interval_s"])
      target_interval_s_ = root["target_interval_s"].as<double>();
    if (!Finite(target_interval_s_) ||
        target_interval_s_ < strike_time_s_ + follow_through_s_ + 0.5) {
      error = "target_interval_s does not leave time to complete the strike";
      return false;
    }
    if (root["random_seed"]) {
      const auto seed = root["random_seed"].as<std::uint64_t>();
      if (seed != 0) rng_.seed(seed);
    }

    if (root["forehand_position_min_b_m"] &&
        !ReadFiniteArray(root["forehand_position_min_b_m"],
                         forehand_position_min_,
                         "forehand_position_min_b_m", error)) {
      return false;
    }
    if (root["forehand_position_max_b_m"] &&
        !ReadFiniteArray(root["forehand_position_max_b_m"],
                         forehand_position_max_,
                         "forehand_position_max_b_m", error)) {
      return false;
    }
    if (root["backhand_position_min_b_m"] &&
        !ReadFiniteArray(root["backhand_position_min_b_m"],
                         backhand_position_min_,
                         "backhand_position_min_b_m", error)) {
      return false;
    }
    if (root["backhand_position_max_b_m"] &&
        !ReadFiniteArray(root["backhand_position_max_b_m"],
                         backhand_position_max_,
                         "backhand_position_max_b_m", error)) {
      return false;
    }
    if (root["forehand_velocity_b_mps"] &&
        !ReadFiniteArray(root["forehand_velocity_b_mps"],
                         forehand_velocity_b_mps_,
                         "forehand_velocity_b_mps", error)) {
      return false;
    }
    if (root["backhand_velocity_b_mps"] &&
        !ReadFiniteArray(root["backhand_velocity_b_mps"],
                         backhand_velocity_b_mps_,
                         "backhand_velocity_b_mps", error)) {
      return false;
    }
    if (root["random_velocity_jitter_mps"] &&
        !ReadFiniteArray(root["random_velocity_jitter_mps"],
                         random_velocity_jitter_mps_,
                         "random_velocity_jitter_mps", error)) {
      return false;
    }
    if (root["random_normal_yaw_deg"])
      random_normal_yaw_rad_ =
          root["random_normal_yaw_deg"].as<double>() * M_PI / 180.0;
    if (root["random_normal_pitch_deg"]) {
      const double symmetric_pitch =
          root["random_normal_pitch_deg"].as<double>() * M_PI / 180.0;
      forehand_normal_pitch_min_rad_ = -symmetric_pitch;
      forehand_normal_pitch_max_rad_ = symmetric_pitch;
      backhand_normal_pitch_min_rad_ = -symmetric_pitch;
      backhand_normal_pitch_max_rad_ = symmetric_pitch;
    }
    if (root["forehand_normal_pitch_min_deg"])
      forehand_normal_pitch_min_rad_ =
          root["forehand_normal_pitch_min_deg"].as<double>() * M_PI / 180.0;
    if (root["forehand_normal_pitch_max_deg"])
      forehand_normal_pitch_max_rad_ =
          root["forehand_normal_pitch_max_deg"].as<double>() * M_PI / 180.0;
    if (root["backhand_normal_pitch_min_deg"])
      backhand_normal_pitch_min_rad_ =
          root["backhand_normal_pitch_min_deg"].as<double>() * M_PI / 180.0;
    if (root["backhand_normal_pitch_max_deg"])
      backhand_normal_pitch_max_rad_ =
          root["backhand_normal_pitch_max_deg"].as<double>() * M_PI / 180.0;

    for (int axis = 0; axis < 3; ++axis) {
      if (!(forehand_position_min_[axis] <
            forehand_position_max_[axis]) ||
          !(backhand_position_min_[axis] <
            backhand_position_max_[axis]) ||
          !Finite(random_velocity_jitter_mps_[axis]) ||
          random_velocity_jitter_mps_[axis] < 0.0) {
        error = "forehand/backhand random target bounds are invalid";
        return false;
      }
    }
    if (!Finite(random_normal_yaw_rad_) || random_normal_yaw_rad_ < 0.0 ||
        !Finite(forehand_normal_pitch_min_rad_) ||
        !Finite(forehand_normal_pitch_max_rad_) ||
        !Finite(backhand_normal_pitch_min_rad_) ||
        !Finite(backhand_normal_pitch_max_rad_) ||
        forehand_normal_pitch_min_rad_ >=
            forehand_normal_pitch_max_rad_ ||
        backhand_normal_pitch_min_rad_ >=
            backhand_normal_pitch_max_rad_) {
      error = "random normal bounds are invalid";
      return false;
    }

    const std::string swing =
        root["swing_type"] ? root["swing_type"].as<std::string>() : "auto";
    if (swing == "forehand") {
      requested_swing_ = SwingType::kForehand;
      requested_swing_auto_ = false;
    } else if (swing == "backhand") {
      requested_swing_ = SwingType::kBackhand;
      requested_swing_auto_ = false;
    } else if (swing == "auto") {
      requested_swing_ = SelectSwingForPosition(target_position_b_m_);
      requested_swing_auto_ = true;
    } else {
      error = "swing_type must be auto, forehand or backhand";
      return false;
    }
    const std::string first =
        root["first_swing"] ? root["first_swing"].as<std::string>()
                            : "forehand";
    if (first == "forehand") {
      next_random_swing_ = SwingType::kForehand;
    } else if (first == "backhand") {
      next_random_swing_ = SwingType::kBackhand;
    } else {
      error = "first_swing must be forehand or backhand";
      return false;
    }
    return true;
  } catch (const std::exception& e) {
    error = "IK strike config: " + std::string(e.what());
    return false;
  }
}

bool IkPointArmSource::Load(const std::string& goal_path,
                            const std::string& robot_xml,
                            std::string& error) {
  if (!LoadGoal(goal_path, error)) return false;
  try {
    fk_ = std::make_unique<RobotFK>(robot_xml);
    wrist_body_index_ = fk_->FindBodyIndex("right_wrist_yaw_Link");
    if (wrist_body_index_ < 0) {
      error = "right_wrist_yaw_Link is absent from robot XML";
      fk_.reset();
      return false;
    }
  } catch (const std::exception& e) {
    error = "IK FK: " + std::string(e.what());
    return false;
  }
  Reset();
  return true;
}

bool IkPointArmSource::SetGoal(const ArmGoal& goal) noexcept {
  if (!goal.valid || !goal.has_cartesian_position) return false;
  auto position = goal.position_m;
  for (double value : position) {
    if (!Finite(value)) return false;
  }

  auto normal = target_normal_b_;
  if (goal.has_racket_normal) {
    normal = goal.racket_normal;
    if (!Normalize3(normal)) return false;
  } else if (goal.has_orientation) {
    std::array<double, 4> orientation = goal.orientation_wxyz;
    double norm = 0.0;
    for (double value : orientation) {
      if (!Finite(value)) return false;
      norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm < 1.0e-8) return false;
    for (double& value : orientation) value /= norm;
    normal = quat_rotate_d(orientation, kRacketMountNormal);
    if (!Normalize3(normal)) return false;
  }

  std::array<double, 3> velocity{};
  if (goal.has_cartesian_linear_velocity) {
    for (double value : goal.linear_velocity_mps) {
      if (!Finite(value)) return false;
    }
    velocity = goal.linear_velocity_mps;
  }

  double strike_time = requested_strike_time_s_;
  if (goal.has_time_to_strike) {
    if (!Finite(goal.time_to_strike_s) || goal.time_to_strike_s < 0.0) {
      return false;
    }
    strike_time = goal.time_to_strike_s;
  }

  SwingType swing = SwingType::kForehand;
  if (goal.swing_type > 0) {
    swing = SwingType::kForehand;
  } else if (goal.swing_type < 0) {
    swing = SwingType::kBackhand;
  } else {
    swing = SelectSwingForPosition(position);
  }
  // The face normal is part of the high-level 10-D strike contract. Never
  // silently negate it to fit an IK branch: doing so changes which rubber face
  // is commanded and hides a frame/semantics error from the planner.

  // Commit only after the entire request has passed validation.  A rejected
  // packet must never leave a mixture of old and new goal fields behind.
  // Reject a SetGoal that arrives while a previous live strike is still
  // committed (Approach/Strike/FollowThrough). The new request becomes the
  // candidate for the NEXT strike once the lock lifts; the model3396 driver
  // already keeps the goal around by calling SetGoal every tick.
  if (committed_strike_active_) {
    return false;
  }
  target_position_b_m_ = position;
  target_normal_b_ = normal;
  target_velocity_b_mps_ = velocity;
  strike_time_s_ = strike_time;
  requested_strike_time_s_ = strike_time;
  requested_swing_ = swing;
  requested_swing_auto_ = goal.swing_type == 0;
  last_source_deadline_ns_ = goal.source_deadline_ns;
  pending_external_goal_sequence_ = goal.sequence;
  have_external_goal_ = true;
  preparation_plan_ = false;
  prepared_ = false;
  goal_dirty_ = true;
  return true;
}

void IkPointArmSource::SeedCommandState(
    const std::array<double, 7>& q,
    const std::array<double, 7>& dq) noexcept {
  for (std::size_t i = 0; i < q.size(); ++i) {
    if (!Finite(q[i]) || !Finite(dq[i])) return;
  }
  last_command_q_ = q;
  last_command_dq_ = dq;
  have_last_command_ = true;
  // Planning failure must hold the same continuous command boundary. Without
  // this initialization, the first rejected live goal would emit one frame of
  // the default zero-filled safe_hold_q_ before the fallback learned the
  // measured pose, producing a visible right-arm jerk.
  safe_hold_q_ = q;
  safe_hold_valid_ = true;
}

void IkPointArmSource::RequireExternalGoals(bool required, bool prepare_ready) noexcept {
  require_external_goals_ = required;
  if (required) {
    random_targets_ = false;
    initialized_ = false;
    prepared_ = false;
    phase_ = Phase::kSolve;
    if (prepare_ready) {
      QueueReadyPlan();
    } else {
      have_external_goal_ = false;
      goal_dirty_ = false;
      preparation_plan_ = false;
    }
  }
}

bool IkPointArmSource::CanAcceptExternalGoal() const noexcept {
  // A real ball always preempts a pending/recovering ready plan. This keeps
  // the controller responsive even if a ball appears during startup or
  // immediately after the preceding follow-through.
  if (!require_external_goals_) return false;
  // A committed live strike is locked: the planner has already cleared IK,
  // trajectory, velocity, body-clearance and timing gates. New rolling
  // predictions must wait for Strike+FollowThrough to finish, otherwise the
  // next-frame SetGoal() would re-plan the approach and Approach would never
  // terminate. The lock releases when the arm returns to kHold.
  if (committed_strike_active_) {
    return false;
  }
  return (!goal_dirty_ || preparation_plan_) &&
      (!initialized_ || phase_ == Phase::kApproach ||
       phase_ == Phase::kHold);
}

void IkPointArmSource::QueueReadyPlan() noexcept {
  // Raise and extend the arm while there is no ball. Ball-flight time is then
  // reserved entirely for the next real stroke.
  target_position_b_m_ = external_ready_position_b_m_;
  target_normal_b_ = external_ready_normal_b_;
  target_velocity_b_mps_.fill(0.0);
  strike_time_s_ = external_ready_time_s_;
  requested_strike_time_s_ = external_ready_time_s_;
  requested_swing_ = SelectSwingForPosition(target_position_b_m_);
  requested_swing_auto_ = true;
  have_external_goal_ = true;
  goal_dirty_ = true;
  preparation_plan_ = true;
  prepared_ = false;
}

void IkPointArmSource::Reset() noexcept {
  phase_ = Phase::kSolve;
  initialized_ = false;
  goal_dirty_ = true;
  strike_reported_ = false;
  preparation_plan_ = false;
  prepared_ = false;
  committed_strike_active_ = false;
  pending_external_goal_sequence_ = 0;
  committed_external_goal_sequence_ = 0;
  plan_origin_s_ = 0.0;
  time_to_strike_s_ = 0.0;
  tracking_minimum_strike_time_s_ = 0.0;
  approach_from_q_.fill(0.0);
  approach_from_dq_.fill(0.0);
  trajectory_base_q_.fill(0.0);
  last_command_q_.fill(0.0);
  last_command_dq_.fill(0.0);
  have_last_command_ = false;
  launch_q_.fill(0.0);
  impact_transition_s_ = follow_through_s_;
  solved_q_.fill(0.0);
  solved_dq_.fill(0.0);
  follow_q_.fill(0.0);
  last_racket_position_b_m_.fill(0.0);
  achieved_strike_velocity_b_mps_.fill(0.0);
  last_position_error_m_ = 0.0;
  last_normal_error_deg_ = 0.0;
  solved_position_error_m_ = 0.0;
  solved_normal_error_deg_ = 0.0;
  solve_iterations_ = 0;
  next_target_time_s_ = 0.0;
  next_plan_retry_time_s_ = 0.0;
  strike_count_ = 0;
  rejected_target_count_ = 0;
  solve_reject_count_ = 0;
  velocity_reject_count_ = 0;
  follow_reject_count_ = 0;
  trajectory_reject_count_ = 0;
  last_velocity_solve_error_mps_ = 0.0;
  last_normal_rate_rad_s_ = 0.0;
  impact_measured_velocity_b_mps_.fill(0.0);
  impact_position_error_m_ = 0.0;
  impact_normal_error_deg_ = 0.0;
  impact_timing_error_s_ = 0.0;
  impact_source_deadline_error_s_ = 0.0;
  impact_max_joint_error_rad_ = 0.0;
  last_source_deadline_ns_ = 0;
  last_planning_duration_ms_ = 0.0;
  planned_min_racket_body_clearance_m_ =
      std::numeric_limits<double>::infinity();
  last_trajectory_reject_reason_ = "none";
  planned_min_clearance_segment_ = -1;
  planned_min_racket_position_b_m_.fill(0.0);
  have_external_goal_ = false;
}

void IkPointArmSource::GenerateRandomTarget(SwingType swing) noexcept {
  const auto& minimum = swing == SwingType::kForehand
                            ? forehand_position_min_
                            : backhand_position_min_;
  const auto& maximum = swing == SwingType::kForehand
                            ? forehand_position_max_
                            : backhand_position_max_;
  for (int axis = 0; axis < 3; ++axis) {
    std::uniform_real_distribution<double> distribution(
        minimum[axis], maximum[axis]);
    target_position_b_m_[axis] = distribution(rng_);
  }

  std::uniform_real_distribution<double> yaw_dist(
      -random_normal_yaw_rad_, random_normal_yaw_rad_);
  const double pitch_min =
      swing == SwingType::kForehand
          ? forehand_normal_pitch_min_rad_
          : backhand_normal_pitch_min_rad_;
  const double pitch_max =
      swing == SwingType::kForehand
          ? forehand_normal_pitch_max_rad_
          : backhand_normal_pitch_max_rad_;
  std::uniform_real_distribution<double> pitch_dist(pitch_min, pitch_max);
  const double yaw = yaw_dist(rng_) +
      (swing == SwingType::kForehand ? 0.0 : M_PI);
  const double pitch = pitch_dist(rng_);
  target_normal_b_ = {
      std::cos(yaw) * std::cos(pitch),
      std::sin(yaw) * std::cos(pitch),
      std::sin(pitch)};
  Normalize3(target_normal_b_);

  const auto& nominal_velocity = swing == SwingType::kForehand
                                     ? forehand_velocity_b_mps_
                                     : backhand_velocity_b_mps_;
  for (int axis = 0; axis < 3; ++axis) {
    std::uniform_real_distribution<double> jitter(
        -random_velocity_jitter_mps_[axis],
        random_velocity_jitter_mps_[axis]);
    target_velocity_b_mps_[axis] = nominal_velocity[axis] + jitter(rng_);
  }
  target_velocity_b_mps_[0] =
      std::max(0.5, target_velocity_b_mps_[0]);
}

const char* IkPointArmSource::PhaseName() const noexcept {
  switch (phase_) {
    case Phase::kSolve:
      return "solve";
    case Phase::kApproach:
      return "approach";
    case Phase::kStrike:
      return "strike";
    case Phase::kFollowThrough:
      return "follow_through";
    case Phase::kHold:
      return "hold";
  }
  return "unknown";
}

const char* IkPointArmSource::SwingName() const noexcept {
  return active_swing_ == SwingType::kForehand ? "forehand" : "backhand";
}

IkPointArmSource::SwingType IkPointArmSource::SelectSwingForPosition(
    const std::array<double, 3>& position) const noexcept {
  // The right-handed forehand covers robot-right (-Y); backhand covers the
  // centre and robot-left (+Y).
  return position[1] < -0.12
             ? SwingType::kForehand
             : SwingType::kBackhand;
}

bool IkPointArmSource::ComputePose(const std::array<double, 31>& q_sdk,
                                   RacketPose& pose) const {
  if (!fk_ || wrist_body_index_ < 0 ||
      wrist_body_index_ >= fk_->NumJoints()) {
    return false;
  }
  const int body_count = fk_->NumJoints();
  std::vector<std::array<double, 3>> positions(
      static_cast<std::size_t>(body_count));
  std::vector<std::array<double, 4>> rotations(
      static_cast<std::size_t>(body_count));
  fk_->DoFKA3(positions.data(), rotations.data(), {0.0, 0.0, 0.0},
              {1.0, 0.0, 0.0, 0.0}, q_sdk.data());
  const auto& wrist_position =
      positions[static_cast<std::size_t>(wrist_body_index_)];
  const auto& wrist_rotation =
      rotations[static_cast<std::size_t>(wrist_body_index_)];
  const auto offset = quat_rotate_d(wrist_rotation, kRacketMountOffset);
  for (int axis = 0; axis < 3; ++axis) {
    pose.position_b_m[axis] = wrist_position[axis] + offset[axis];
  }
  pose.normal_b = quat_rotate_d(wrist_rotation, kRacketMountNormal);
  return Normalize3(pose.normal_b);
}

bool IkPointArmSource::EvaluateRacketPose(
    const std::array<double, 31>& q_sdk,
    std::array<double, 3>& position_b_m,
    std::array<double, 3>& normal_b) const {
  RacketPose pose{};
  if (!ComputePose(q_sdk, pose)) return false;
  position_b_m = pose.position_b_m;
  normal_b = pose.normal_b;
  return true;
}

bool IkPointArmSource::ComputeJacobian(
    const std::array<double, 31>& q_sdk,
    Eigen::Matrix<double, 6, 7>& jacobian) const {
  RacketPose current{};
  if (!ComputePose(q_sdk, current)) return false;
  for (std::size_t joint = 0; joint < kRightArmSdk.size(); ++joint) {
    std::array<double, 31> perturbed_q = q_sdk;
    const int sdk = kRightArmSdk[joint];
    const double room =
        kRightArmMax[joint] - kJointLimitMarginRad - perturbed_q[sdk];
    const double epsilon =
        room >= kFiniteDifferenceRad ? kFiniteDifferenceRad
                                    : -kFiniteDifferenceRad;
    perturbed_q[sdk] += epsilon;
    RacketPose perturbed{};
    if (!ComputePose(perturbed_q, perturbed)) return false;
    for (int axis = 0; axis < 3; ++axis) {
      jacobian(axis, static_cast<Eigen::Index>(joint)) =
          (perturbed.position_b_m[axis] - current.position_b_m[axis]) /
          epsilon;
    }
    const auto angular_delta = Cross(current.normal_b, perturbed.normal_b);
    for (int axis = 0; axis < 3; ++axis) {
      jacobian(3 + axis, static_cast<Eigen::Index>(joint)) =
          kOrientationWeightMPerRad * angular_delta[axis] / epsilon;
    }
  }
  return jacobian.allFinite();
}

bool IkPointArmSource::SolveFromSeed(
    const std::array<double, 31>& base_q, const std::array<double, 7>& seed,
    const std::array<double, 3>& target_position,
    const std::array<double, 3>& target_normal,
    std::array<double, 7>& solution, double& position_error_m,
    double& normal_error_deg, int& iterations) const {
  std::array<double, 31> q = base_q;
  for (int axis = 0; axis < 3; ++axis) q[axis] = 0.0;
  for (std::size_t joint = 0; joint < seed.size(); ++joint) {
    q[kRightArmSdk[joint]] = std::clamp(
        seed[joint], kRightArmMin[joint] + kJointLimitMarginRad,
        kRightArmMax[joint] - kJointLimitMarginRad);
  }

  RacketPose current{};
  iterations = 0;
  for (; iterations < kMaxIterations; ++iterations) {
    if (!ComputePose(q, current)) return false;
    position_error_m = PositionError(current.position_b_m, target_position);
    normal_error_deg =
        ComputeNormalErrorDeg(current.normal_b, target_normal);
    if (position_error_m <= 0.001 && normal_error_deg <= 0.5) break;

    Eigen::Matrix<double, 6, 1> error;
    for (int axis = 0; axis < 3; ++axis) {
      error[axis] = target_position[axis] - current.position_b_m[axis];
    }
    const auto normal_error = Cross(current.normal_b, target_normal);
    for (int axis = 0; axis < 3; ++axis) {
      error[3 + axis] = kOrientationWeightMPerRad * normal_error[axis];
    }

    Eigen::Matrix<double, 6, 7> jacobian;
    if (!ComputeJacobian(q, jacobian)) return false;
    const Eigen::Matrix<double, 6, 6> regularized =
        jacobian * jacobian.transpose() +
        (kIkDamping * kIkDamping) *
            Eigen::Matrix<double, 6, 6>::Identity();
    const Eigen::Matrix<double, 7, 6> jacobian_pinv =
        jacobian.transpose() * regularized.inverse();
    Eigen::Matrix<double, 7, 1> delta = jacobian_pinv * error;
    // Position + one face normal constrain five independent task-space DOFs,
    // leaving two arm posture DOFs. Keep those close to the selected
    // forehand/backhand seed instead of letting the least-norm solve drift to
    // a shoulder/wrist limit and create an unnatural elbow branch.
    Eigen::Matrix<double, 7, 1> posture_error;
    for (std::size_t joint = 0; joint < seed.size(); ++joint) {
      posture_error[static_cast<Eigen::Index>(joint)] =
          seed[joint] - q[kRightArmSdk[joint]];
    }
    const Eigen::Matrix<double, 7, 7> nullspace =
        Eigen::Matrix<double, 7, 7>::Identity() -
        jacobian_pinv * jacobian;
    delta += 0.02 * nullspace * posture_error;
    if (!delta.allFinite()) return false;
    const double max_abs = delta.cwiseAbs().maxCoeff();
    if (max_abs > kMaxIterationStepRad) {
      delta *= kMaxIterationStepRad / max_abs;
    }
    for (std::size_t joint = 0; joint < seed.size(); ++joint) {
      const int sdk = kRightArmSdk[joint];
      q[sdk] = std::clamp(
          q[sdk] + delta[static_cast<Eigen::Index>(joint)],
          kRightArmMin[joint] + kJointLimitMarginRad,
          kRightArmMax[joint] - kJointLimitMarginRad);
    }
  }

  if (!ComputePose(q, current)) return false;
  position_error_m = PositionError(current.position_b_m, target_position);
  normal_error_deg =
      ComputeNormalErrorDeg(current.normal_b, target_normal);
  for (std::size_t joint = 0; joint < solution.size(); ++joint) {
    solution[joint] = q[kRightArmSdk[joint]];
  }
  return true;
}

bool IkPointArmSource::Solve(const robot_io::RobotState& state,
                             SwingType swing) noexcept {
  if (!fk_ || state.q.size() < 31) return false;
  try {
    std::array<double, 31> base_q{};
    for (int sdk = 0; sdk < 31; ++sdk) {
      if (!Finite(state.q[sdk])) return false;
      base_q[sdk] = state.q[sdk];
    }

    // The 10-D interface intentionally carries no mandatory forehand/backhand
    // label. In automatic live mode evaluate both elbow branches and let the
    // kinematic/error/current-motion score choose. Explicit high-level branch
    // requests and offline/random tests still evaluate only the requested one.
    const std::array<SwingType, 2> branches = {
        swing,
        swing == SwingType::kForehand ? SwingType::kBackhand
                                      : SwingType::kForehand};
    const int branch_count =
        require_external_goals_ && requested_swing_auto_ && !preparation_plan_
            ? 2
            : 1;

    double best_cost = std::numeric_limits<double>::infinity();
    std::array<double, 7> best_solution{};
    double best_position_error = std::numeric_limits<double>::infinity();
    double best_normal_error = std::numeric_limits<double>::infinity();
    int best_iterations = 0;
    SwingType best_swing = swing;

    for (int branch_index = 0; branch_index < branch_count; ++branch_index) {
      const SwingType candidate_swing = branches[branch_index];
      const auto& stroke_seed = candidate_swing == SwingType::kForehand
                                    ? kForehandSeed
                                    : kBackhandSeed;
      std::array<std::array<double, 7>, 4> seeds{};
      seeds[0] = stroke_seed;
      seeds[1] = stroke_seed;
      seeds[1][0] = std::clamp(
          seeds[1][0] +
              (candidate_swing == SwingType::kForehand ? 0.28 : -0.28),
          kRightArmMin[0], kRightArmMax[0]);
      seeds[2] = stroke_seed;
      seeds[2][3] = std::clamp(
          seeds[2][3] + 0.25, kRightArmMin[3], kRightArmMax[3]);
      for (std::size_t joint = 0; joint < 7; ++joint) {
        seeds[3][joint] = state.q[kRightArmSdk[joint]];
      }
      const bool forehand = candidate_swing == SwingType::kForehand;
      if (!BranchCompatible(seeds[3], forehand)) seeds[3] = stroke_seed;

      for (const auto& seed : seeds) {
        std::array<double, 7> candidate{};
        double position_error = 0.0;
        double normal_error = 0.0;
        int iterations = 0;
        if (!SolveFromSeed(base_q, seed, target_position_b_m_,
                           target_normal_b_, candidate, position_error,
                           normal_error, iterations) ||
            !BranchCompatible(candidate, forehand)) {
          continue;
        }
        double posture_distance_sq = 0.0;
        double current_distance_sq = 0.0;
        for (std::size_t joint = 0; joint < candidate.size(); ++joint) {
          const double delta = candidate[joint] - stroke_seed[joint];
          posture_distance_sq += delta * delta;
          const double current_delta =
              candidate[joint] - state.q[kRightArmSdk[joint]];
          current_distance_sq += current_delta * current_delta;
        }
        const double cost =
            position_error +
            kOrientationWeightMPerRad * normal_error * M_PI / 180.0 +
            0.002 * posture_distance_sq +
            (require_external_goals_ ? 0.01 * current_distance_sq : 0.0) +
            0.0002 / std::max(0.001, MinimumJointMargin(candidate));
        if (cost < best_cost) {
          best_cost = cost;
          best_solution = candidate;
          best_position_error = position_error;
          best_normal_error = normal_error;
          best_iterations = iterations;
          best_swing = candidate_swing;
        }
      }
    }

    if (!Finite(best_cost) ||
        best_position_error > kSolvedPositionToleranceM ||
        best_normal_error > kSolvedNormalToleranceDeg ||
        MinimumJointMargin(best_solution) < kStrikeJointMarginRad) {
      return false;
    }
    solved_q_ = best_solution;
    solved_position_error_m_ = best_position_error;
    solved_normal_error_deg_ = best_normal_error;
    solve_iterations_ = best_iterations;
    active_swing_ = best_swing;
    return true;
  } catch (...) {
    return false;
  }
}

bool IkPointArmSource::ComputeStrikeVelocity(
    const std::array<double, 31>& base_q) noexcept {
  try {
    std::array<double, 31> q = base_q;
    for (int axis = 0; axis < 3; ++axis) q[axis] = 0.0;
    for (std::size_t joint = 0; joint < solved_q_.size(); ++joint) {
      q[kRightArmSdk[joint]] = solved_q_[joint];
    }
    Eigen::Matrix<double, 6, 7> jacobian;
    if (!ComputeJacobian(q, jacobian)) return false;
    Eigen::Matrix<double, 3, 1> desired;
    for (int axis = 0; axis < 3; ++axis) {
      desired[axis] = target_velocity_b_mps_[axis];
    }
    const Eigen::Matrix<double, 3, 7> position_jacobian =
        jacobian.topRows<3>();
    const Eigen::Matrix<double, 3, 3> regularized =
        position_jacobian * position_jacobian.transpose() +
        (kVelocityDamping * kVelocityDamping) *
            Eigen::Matrix<double, 3, 3>::Identity();
    const Eigen::Matrix<double, 7, 3> position_pinv =
        position_jacobian.transpose() * regularized.ldlt().solve(
            Eigen::Matrix<double, 3, 3>::Identity());
    Eigen::Matrix<double, 7, 1> qdot =
        position_pinv * desired;
    // The request defines linear velocity, not angular velocity. Use the
    // minimum-norm position solution; trying to freeze the face normal here
    // over-constrains the seven-joint arm and can create a large elbow/wrist
    // velocity even though the requested hit itself is reachable.
    if (!qdot.allFinite()) return false;
    const double max_abs = qdot.cwiseAbs().maxCoeff();
    if (max_abs > max_joint_velocity_rad_s_) {
      qdot *= max_joint_velocity_rad_s_ / max_abs;
    }
    for (std::size_t joint = 0; joint < solved_dq_.size(); ++joint) {
      solved_dq_[joint] = qdot[static_cast<Eigen::Index>(joint)];
    }
    const Eigen::Matrix<double, 6, 1> achieved = jacobian * qdot;
    for (int axis = 0; axis < 3; ++axis) {
      achieved_strike_velocity_b_mps_[axis] = achieved[axis];
    }
    const double requested_speed = Norm3(target_velocity_b_mps_);
    const double achieved_speed = Norm3(achieved_strike_velocity_b_mps_);
    const double error =
        PositionError(target_velocity_b_mps_,
                      achieved_strike_velocity_b_mps_);
    const double normal_rate =
        achieved.tail<3>().norm() / kOrientationWeightMPerRad;
    last_velocity_solve_error_mps_ = error;
    last_normal_rate_rad_s_ = normal_rate;
    const double direction_cos =
        requested_speed > 1.0e-6 && achieved_speed > 1.0e-6
            ? Dot(target_velocity_b_mps_,
                  achieved_strike_velocity_b_mps_) /
                  (requested_speed * achieved_speed)
            : 1.0;
    return error <= std::max(0.12, 0.12 * requested_speed) &&
           direction_cos >= 0.96;
  } catch (...) {
    return false;
  }
}

bool IkPointArmSource::PrepareImpactTransition() noexcept {
  // Keep the terminal five-point boundary's exact endpoint velocity while
  // placing launch_q close enough to the hit pose to stay in front of the
  // torso. The new 0.32 s impact window fits within both the 6 rad/s joint
  // speed cap and the ~100 rad/s^2 hardware acceleration envelope, so a
  // 0.5 rad motion now peaks at ~3.1 rad/s and ~80 rad/s^2 instead of the
  // previous 5.9 rad/s and 293 rad/s^2.
  double duration = std::min(kMaximumImpactTransitionS, follow_through_s_);
  duration = std::min(duration, 0.5 * strike_time_s_);
  for (std::size_t joint = 0; joint < solved_q_.size(); ++joint) {
    const double speed = std::abs(solved_dq_[joint]);
    if (speed < 1.0e-8) continue;
    const double room = solved_dq_[joint] > 0.0
                            ? solved_q_[joint] -
                                  (kRightArmMin[joint] +
                                   kJointLimitMarginRad)
                            : (kRightArmMax[joint] -
                               kJointLimitMarginRad) - solved_q_[joint];
    if (room <= 0.0) return false;
    // 0.5 rad / speed already exceeds the launch pose back-step below; the
    // 1.90 safety factor keeps the quintic out of the joint limits.
    duration = std::min(duration, 1.90 * room / speed);
  }
  if (!Finite(duration) || duration < 0.04) return false;
  impact_transition_s_ = duration;
  // The back-step is fixed at 0.25 of the impact transition so the launch
  // pose stays close to the hit pose (improves body clearance) while the
  // peak acceleration scales with duration^2.
  for (std::size_t joint = 0; joint < launch_q_.size(); ++joint) {
    launch_q_[joint] =
        solved_q_[joint] - 0.25 * impact_transition_s_ * solved_dq_[joint];
    if (launch_q_[joint] < kRightArmMin[joint] + kJointLimitMarginRad ||
        launch_q_[joint] > kRightArmMax[joint] - kJointLimitMarginRad) {
      return false;
    }
  }
  return true;
}

bool IkPointArmSource::SolveFollowThrough(
    const std::array<double, 31>& base_q, SwingType swing) noexcept {
  const double speed = Norm3(target_velocity_b_mps_);
  if (speed < 1.0e-6) {
    follow_q_ = solved_q_;
    solved_dq_.fill(0.0);
    achieved_strike_velocity_b_mps_.fill(0.0);
    return true;
  }
  // Integrating a linear deceleration from the impact joint velocity gives a
  // C1-continuous follow-through without imposing a second static racket pose.
  // That avoids the old failure mode where the follow IK dragged the elbow to
  // another branch or a joint stop immediately after contact.
  const double nominal_distance = 0.5 * speed * follow_through_s_;
  const double distance_scale =
      nominal_distance > follow_through_distance_m_
          ? follow_through_distance_m_ / nominal_distance
          : 1.0;
  for (std::size_t joint = 0; joint < follow_q_.size(); ++joint) {
    follow_q_[joint] =
        solved_q_[joint] +
        0.5 * follow_through_s_ * distance_scale * solved_dq_[joint];
    if (follow_q_[joint] < kRightArmMin[joint] + kJointLimitMarginRad ||
        follow_q_[joint] > kRightArmMax[joint] - kJointLimitMarginRad) {
      return false;
    }
  }
  if (!BranchCompatible(follow_q_, swing == SwingType::kForehand)) {
    return false;
  }

  std::array<double, 31> follow_sdk = base_q;
  for (int axis = 0; axis < 3; ++axis) follow_sdk[axis] = 0.0;
  for (std::size_t joint = 0; joint < follow_q_.size(); ++joint) {
    follow_sdk[kRightArmSdk[joint]] = follow_q_[joint];
  }
  RacketPose follow_pose{};
  if (!ComputePose(follow_sdk, follow_pose)) return false;
  std::array<double, 3> displacement{};
  for (int axis = 0; axis < 3; ++axis) {
    displacement[axis] =
        follow_pose.position_b_m[axis] - target_position_b_m_[axis];
  }
  return Dot(displacement, target_velocity_b_mps_) >
         0.015 * speed;
}

int IkPointArmSource::TrajectorySegmentCount() const noexcept {
  return 3;
}

double IkPointArmSource::TrajectorySegmentDuration(
    int segment) const noexcept {
  const double approach_duration = strike_time_s_ - impact_transition_s_;
  if (segment == 0) return approach_duration;
  if (segment == 1) return impact_transition_s_;
  return follow_through_s_;
}

void IkPointArmSource::TrajectorySegmentBoundary(
    int segment, std::size_t joint, double& q0, double& dq0,
    double& q1, double& dq1) const noexcept {
  if (segment == 0) {
    q0 = approach_from_q_[joint];
    dq0 = approach_from_dq_[joint];
    q1 = launch_q_[joint];
    dq1 = 0.0;
  } else if (segment == 1) {
    q0 = launch_q_[joint];
    dq0 = 0.0;
    q1 = solved_q_[joint];
    dq1 = solved_dq_[joint];
  } else {
    q0 = solved_q_[joint];
    dq0 = solved_dq_[joint];
    q1 = follow_q_[joint];
    dq1 = 0.0;
  }
}

bool IkPointArmSource::TrajectoryWithinLimits() const noexcept {
  constexpr int kSamples = 100;
  const double approach_duration = strike_time_s_ - impact_transition_s_;
  if (!Finite(approach_duration) || approach_duration < 0.04) return false;
  const int segments = TrajectorySegmentCount();
  for (int segment = 0; segment < segments; ++segment) {
    const double duration = TrajectorySegmentDuration(segment);
    if (!Finite(duration) || duration < 0.04) return false;
    // Peak quintic acceleration for rest-to-rest with non-zero endpoints is
    // bounded by 15 * |q1 - q0| / duration^2. The trajectory never asks the
    // policy driver for more than this; the previous 0.16 s impact window
    // produced > 200 rad/s^2 peaks for 0.5 rad motions, which the underlying
    // driver rejected. Enforce the same envelope here so FitTrajectoryDuration
    // extends strike_time_s_ when needed.
    const double max_joint_motion_rad = [&]() noexcept {
      double max_motion = 0.0;
      for (std::size_t joint = 0; joint < solved_q_.size(); ++joint) {
        double q0 = 0.0, dq0 = 0.0, q1 = 0.0, dq1 = 0.0;
        TrajectorySegmentBoundary(segment, joint, q0, dq0, q1, dq1);
        max_motion = std::max(max_motion, std::abs(q1 - q0));
      }
      return max_motion;
    }();
    const double peak_accel_bound_rad_s_sq =
        15.0 * max_joint_motion_rad / (duration * duration);
    if (peak_accel_bound_rad_s_sq > kMaximumImpactAccelRadSSq) return false;
    for (int sample = 0; sample <= kSamples; ++sample) {
      const double elapsed =
          duration * static_cast<double>(sample) / kSamples;
      for (std::size_t joint = 0; joint < solved_q_.size(); ++joint) {
        double q0 = 0.0;
        double dq0 = 0.0;
        double q1 = 0.0;
        double dq1 = 0.0;
        TrajectorySegmentBoundary(
            segment, joint, q0, dq0, q1, dq1);
        const JointSample value =
            QuinticBoundary(q0, dq0, q1, dq1, duration, elapsed);
        if (!Finite(value.q) || !Finite(value.dq) ||
            value.q < kRightArmMin[joint] ||
            value.q > kRightArmMax[joint] ||
            std::abs(value.dq) > max_joint_velocity_rad_s_ + 1.0e-6) {
          return false;
        }
      }
    }
  }
  return true;
}

bool IkPointArmSource::TrajectoryHasBodyClearance() noexcept {
  // The initial preparation plan may start in the powered-off folded pose,
  // which is already inside the conservative envelope. It is allowed only to
  // escape forward into the ready pose. Every real strike, including rolling
  // replans, is checked at 100 samples per segment.
  if (preparation_plan_) {
    planned_min_racket_body_clearance_m_ =
        std::numeric_limits<double>::infinity();
    return true;
  }

  constexpr int kSamples = 100;
  const double approach_duration = strike_time_s_ - impact_transition_s_;
  if (!Finite(approach_duration) || approach_duration < 0.04) return false;

  planned_min_racket_body_clearance_m_ =
      std::numeric_limits<double>::infinity();
  planned_min_clearance_segment_ = -1;
  planned_min_racket_position_b_m_.fill(0.0);
  bool has_clearance = true;
  const int segments = TrajectorySegmentCount();
  for (int segment = 0; segment < segments; ++segment) {
    const double duration = TrajectorySegmentDuration(segment);
    if (!Finite(duration) || duration < 0.04) return false;
    for (int sample = 0; sample <= kSamples; ++sample) {
      const double elapsed =
          duration * static_cast<double>(sample) / kSamples;
      std::array<double, 31> q_sdk = trajectory_base_q_;
      for (int axis = 0; axis < 3; ++axis) q_sdk[axis] = 0.0;
      for (std::size_t joint = 0; joint < solved_q_.size(); ++joint) {
        double q0 = 0.0;
        double dq0 = 0.0;
        double q1 = 0.0;
        double dq1 = 0.0;
        TrajectorySegmentBoundary(
            segment, joint, q0, dq0, q1, dq1);
        const JointSample value =
            QuinticBoundary(q0, dq0, q1, dq1, duration, elapsed);
        if (!Finite(value.q)) return false;
        q_sdk[kRightArmSdk[joint]] = value.q;
      }
      RacketPose pose{};
      if (!ComputePose(q_sdk, pose) || !Finite(pose.position_b_m[0])) {
        return false;
      }
      const double clearance =
          pose.position_b_m[0] - kRacketBodyProtectionPlaneX;
      if (clearance < planned_min_racket_body_clearance_m_) {
        planned_min_racket_body_clearance_m_ = clearance;
        planned_min_clearance_segment_ = segment;
        planned_min_racket_position_b_m_ = pose.position_b_m;
      }
      if (clearance < -1.0e-6) has_clearance = false;
    }
  }
  return has_clearance;
}

bool IkPointArmSource::FitTrajectoryDuration() noexcept {
  // The impact velocity and follow-through are already independently checked.
  // What remains is the measured-pose -> solved-pose quintic.  Its peak joint
  // speed monotonically falls as the approach is lengthened, so a bounded
  // expansion followed by bisection finds the earliest safe execution time.
  // When the impact segment still violates the acceleration envelope (e.g.
  // very large joint motion at the hit pose), extend impact_transition_s_ up
  // to the new kMaximumImpactTransitionS = 0.32 s cap before growing the
  // overall strike time.
  if (TrajectoryWithinLimits()) return true;

  // First, see if a longer impact transition alone fits the envelope. This
  // keeps the planner from lengthening the approach needlessly when the
  // bottleneck is the 0->strike-velocity ramp.
  if (impact_transition_s_ < kMaximumImpactTransitionS) {
    const double saved_impact = impact_transition_s_;
    impact_transition_s_ = std::min(
        kMaximumImpactTransitionS, follow_through_s_);
    if (TrajectoryWithinLimits()) return true;
    impact_transition_s_ = saved_impact;
  }

  const double requested = strike_time_s_;
  double lower = requested;
  double upper = requested;
  constexpr double kMaximumPlannedStrikeTimeS = 8.0;
  while (upper < kMaximumPlannedStrikeTimeS) {
    upper = std::min(kMaximumPlannedStrikeTimeS,
                     std::max(upper + 0.10, upper * 1.35));
    strike_time_s_ = upper;
    if (TrajectoryWithinLimits()) {
      for (int iteration = 0; iteration < 24; ++iteration) {
        const double mid = 0.5 * (lower + upper);
        strike_time_s_ = mid;
        if (TrajectoryWithinLimits()) {
          upper = mid;
        } else {
          lower = mid;
        }
      }
      strike_time_s_ = upper;
      return true;
    }
    lower = upper;
  }
  strike_time_s_ = requested;
  return false;
}

double IkPointArmSource::EstimateTrackingMinimumStrikeTimeS() const noexcept {
  if (!Finite(impact_transition_s_) || impact_transition_s_ <= 0.0) return 0.0;
  double approach_s = 0.0;
  for (std::size_t joint = 0; joint < approach_from_q_.size(); ++joint) {
    const double distance =
        std::abs(launch_q_[joint] - approach_from_q_[joint]);
    approach_s = std::max(
        approach_s, 1.875 * distance /
                         kConservativeTrackingJointSpeedRadS);
  }
  return approach_s + impact_transition_s_;
}

void IkPointArmSource::FillSafeHold(ArmTarget& target) const noexcept {
  target = ArmTarget{};
  target.has_arm_gains = true;
  for (int joint = 0; joint < 7; ++joint) {
    target.q[joint] = 0.0;
    target.kp[joint] = 0.0;
    target.kd[joint] = 0.0;
    target.q[7 + joint] = safe_hold_q_[joint];
    target.dq[7 + joint] = 0.0;
    target.kp[7 + joint] = kRightArmKp[joint];
    target.kd[7 + joint] = kRightArmKd[joint];
  }
}

bool IkPointArmSource::StartTarget(const robot_io::RobotState& state,
                                   double time_s) noexcept {
  const auto planning_start = std::chrono::steady_clock::now();
  const auto finish_timing = [this, planning_start]() noexcept {
    last_planning_duration_ms_ =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - planning_start).count();
  };
  const auto old_position = target_position_b_m_;
  const auto old_normal = target_normal_b_;
  const auto old_velocity = target_velocity_b_mps_;
  const auto old_solved_q = solved_q_;
  const auto old_solved_dq = solved_dq_;
  const auto old_follow_q = follow_q_;
  const auto old_approach_q = approach_from_q_;
  const auto old_approach_dq = approach_from_dq_;
  const auto old_launch_q = launch_q_;
  const double old_impact_transition_s = impact_transition_s_;
  const auto old_achieved_velocity = achieved_strike_velocity_b_mps_;
  const double old_solved_position_error = solved_position_error_m_;
  const double old_solved_normal_error = solved_normal_error_deg_;
  const int old_iterations = solve_iterations_;
  const SwingType old_swing = active_swing_;

  const SwingType desired_swing =
      random_targets_ ? next_random_swing_ : requested_swing_;
  const int attempts = random_targets_ ? kMaxRandomAttempts : 1;
  std::array<double, 31> base_q{};
  for (int sdk = 0; sdk < 31; ++sdk) base_q[sdk] = state.q[sdk];
  trajectory_base_q_ = base_q;

  for (int attempt = 0; attempt < attempts; ++attempt) {
    if (random_targets_) GenerateRandomTarget(desired_swing);
    if (!Solve(state, desired_swing)) {
      ++rejected_target_count_;
      ++solve_reject_count_;
      continue;
    }
    if (!ComputeStrikeVelocity(base_q)) {
      ++rejected_target_count_;
      ++velocity_reject_count_;
      continue;
    }
    if (!PrepareImpactTransition()) {
      ++rejected_target_count_;
      ++trajectory_reject_count_;
      last_trajectory_reject_reason_ = "impact_transition";
      continue;
    }
    if (!SolveFollowThrough(base_q, active_swing_)) {
      ++rejected_target_count_;
      ++follow_reject_count_;
      continue;
    }
    // StartTarget is entered from measured hold.  Treat the start as
    // stationary instead of feeding quantized state.dq noise into the
    // quintic boundary near a joint limit.
    for (int joint = 0; joint < 7; ++joint) {
      // A rolling prediction may preempt an approach every few control ticks.
      // Continue from the last C1 command boundary instead of repeatedly
      // resetting the commanded velocity to zero.
      if (have_last_command_) {
        approach_from_q_[joint] = last_command_q_[joint];
        approach_from_dq_[joint] = last_command_dq_[joint];
      } else {
        approach_from_q_[joint] = state.q[kRightArmSdk[joint]];
        approach_from_dq_[joint] = 0.0;
      }
    }
    // The inflated-body trajectory checker is the single collision policy.
    if (!FitTrajectoryDuration()) {
      ++rejected_target_count_;
      ++trajectory_reject_count_;
      last_trajectory_reject_reason_ = "joint_limits_or_speed";
      continue;
    }
    if (!TrajectoryHasBodyClearance()) {
      ++rejected_target_count_;
      ++trajectory_reject_count_;
      last_trajectory_reject_reason_ = "racket_body_clearance";
      continue;
    }
    last_trajectory_reject_reason_ = "none";
    tracking_minimum_strike_time_s_ = EstimateTrackingMinimumStrikeTimeS();
    plan_origin_s_ = time_s;
    time_to_strike_s_ = strike_time_s_;
    phase_ = Phase::kApproach;
    strike_reported_ = false;
    // Lock the planner to the committed trajectory until the follow-through
    // is done. CanAcceptExternalGoal() refuses new SetGoal calls during this
    // window, so a 100 Hz prediction stream cannot keep rebuilding the
    // approach and starving the actual strike.
    committed_strike_active_ = !preparation_plan_;
    committed_external_goal_sequence_ =
        committed_strike_active_ ? pending_external_goal_sequence_ : 0u;
    if (preparation_plan_) {
      // The completed preparation endpoint is the only measured/solved pose
      // allowed as the live failure hold reference.
      safe_hold_q_ = follow_q_;
      safe_hold_valid_ = true;
    }
    next_target_time_s_ = time_s + target_interval_s_;
    next_plan_retry_time_s_ = 0.0;
    if (random_targets_ && alternate_swings_) {
      next_random_swing_ =
          desired_swing == SwingType::kForehand
              ? SwingType::kBackhand
              : SwingType::kForehand;
    }
    finish_timing();
    return true;
  }

  target_position_b_m_ = old_position;
  target_normal_b_ = old_normal;
  target_velocity_b_mps_ = old_velocity;
  solved_q_ = old_solved_q;
  solved_dq_ = old_solved_dq;
  follow_q_ = old_follow_q;
  approach_from_q_ = old_approach_q;
  approach_from_dq_ = old_approach_dq;
  launch_q_ = old_launch_q;
  impact_transition_s_ = old_impact_transition_s;
  achieved_strike_velocity_b_mps_ = old_achieved_velocity;
  solved_position_error_m_ = old_solved_position_error;
  solved_normal_error_deg_ = old_solved_normal_error;
  solve_iterations_ = old_iterations;
  active_swing_ = old_swing;
  finish_timing();
  return false;
}

void IkPointArmSource::FillTarget(const robot_io::RobotState& state,
                                  double time_s,
                                  ArmTarget& target) noexcept {
  target = ArmTarget{};
  target.has_arm_gains = true;
  for (int joint = 0; joint < 7; ++joint) {
    target.q[joint] = state.q[5 + joint];
    target.kp[joint] = 0.0;
    target.kd[joint] = 0.0;
  }

  const double elapsed_s = std::max(0.0, time_s - plan_origin_s_);
  time_to_strike_s_ = std::max(0.0, strike_time_s_ - elapsed_s);
  for (int joint = 0; joint < 7; ++joint) {
    JointSample sample{};
    const double approach_duration = strike_time_s_ - impact_transition_s_;
    if (elapsed_s < approach_duration) {
      phase_ = Phase::kApproach;
      sample = QuinticBoundary(
          approach_from_q_[joint], approach_from_dq_[joint],
          launch_q_[joint], 0.0, approach_duration, elapsed_s);
    } else if (elapsed_s < strike_time_s_) {
      phase_ = Phase::kApproach;
      sample = QuinticBoundary(
          launch_q_[joint], 0.0, solved_q_[joint], solved_dq_[joint],
          impact_transition_s_, elapsed_s - approach_duration);
    } else if (elapsed_s <= strike_time_s_ + follow_through_s_) {
      phase_ = strike_reported_ ? Phase::kFollowThrough : Phase::kStrike;
      sample = QuinticBoundary(
          solved_q_[joint], solved_dq_[joint], follow_q_[joint], 0.0,
          follow_through_s_, elapsed_s - strike_time_s_);
    } else {
      phase_ = Phase::kHold;
      // The follow-through is the end of the live strike; release the
      // committed_strike_active_ lock so the next SetGoal can preempt the
      // recovery motion and start a new approach.
      committed_strike_active_ = false;
      sample = {follow_q_[joint], 0.0};
    }
    target.q[7 + joint] = sample.q;
    target.dq[7 + joint] = sample.dq;
    target.kp[7 + joint] = kRightArmKp[joint];
    target.kd[7 + joint] = kRightArmKd[joint];
    last_command_q_[joint] = sample.q;
    last_command_dq_[joint] = sample.dq;
  }
  have_last_command_ = true;
  if (elapsed_s >= strike_time_s_ && !strike_reported_) {
    strike_reported_ = true;
    if (preparation_plan_) {
      prepared_ = true;
    } else {
      ++strike_count_;
    }
  }
}

void IkPointArmSource::UpdateMeasuredError(
    const robot_io::RobotState& state) noexcept {
  if (state.q.size() < 31 || state.dq.size() < 31) return;
  try {
    std::array<double, 31> q{};
    for (int sdk = 0; sdk < 31; ++sdk) q[sdk] = state.q[sdk];
    RacketPose pose{};
    if (!ComputePose(q, pose)) return;
    last_racket_position_b_m_ = pose.position_b_m;
    last_position_error_m_ =
        PositionError(pose.position_b_m, target_position_b_m_);
    last_normal_error_deg_ =
        ComputeNormalErrorDeg(pose.normal_b, target_normal_b_);
  } catch (...) {
  }
}

void IkPointArmSource::SnapshotImpactTelemetry(
    const robot_io::RobotState& state, double time_s) noexcept {
  impact_position_error_m_ = last_position_error_m_;
  impact_normal_error_deg_ = last_normal_error_deg_;
  impact_timing_error_s_ = time_s - (plan_origin_s_ + strike_time_s_);
  if (last_source_deadline_ns_ > 0) {
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    impact_source_deadline_error_s_ =
        static_cast<double>(now_ns - last_source_deadline_ns_) / 1.0e9;
  } else {
    impact_source_deadline_error_s_ = 0.0;
  }
  impact_max_joint_error_rad_ = 0.0;
  impact_measured_velocity_b_mps_.fill(0.0);
  if (state.q.size() < 31 || state.dq.size() < 31) return;
  try {
    std::array<double, 31> q{};
    Eigen::Matrix<double, 6, 7> jacobian;
    Eigen::Matrix<double, 7, 1> dq;
    for (int sdk = 0; sdk < 31; ++sdk) q[sdk] = state.q[sdk];
    for (int joint = 0; joint < 7; ++joint) {
      dq(joint) = state.dq[kRightArmSdk[joint]];
      impact_max_joint_error_rad_ = std::max(
          impact_max_joint_error_rad_,
          std::abs(solved_q_[joint] - state.q[kRightArmSdk[joint]]));
    }
    if (!ComputeJacobian(q, jacobian)) return;
    const Eigen::Matrix<double, 3, 1> velocity = jacobian.topRows<3>() * dq;
    for (int axis = 0; axis < 3; ++axis) {
      if (Finite(velocity(axis))) impact_measured_velocity_b_mps_[axis] = velocity(axis);
    }
  } catch (...) {
  }
}

bool IkPointArmSource::Update(const robot_io::RobotState& state,
                              double time_s,
                              ArmTarget& target) noexcept {
  if (!Finite(time_s) || state.q.size() < 31 || state.dq.size() < 31) {
    return false;
  }
  if (require_external_goals_ && !have_external_goal_) {
    // A live-network controller must be passive until it has a fresh,
    // frame-validated target.  In particular, never execute the static YAML
    // target as a fallback when the planner is absent.
    FillSafeHold(target);
    UpdateMeasuredError(state);
    return true;
  }
  const bool periodic_target_due =
      random_targets_ && initialized_ && time_s >= next_target_time_s_;
  if (!initialized_ || goal_dirty_ || periodic_target_due) {
    if (!initialized_ && time_s < next_plan_retry_time_s_) {
      FillSafeHold(target);
      UpdateMeasuredError(state);
      return true;
    }
    if (!StartTarget(state, time_s)) {
      // A Cartesian plan is optional work inside a hard real-time command
      // callback. Failure must never return false and pin the driver's policy
      // tick forever.  In external-command mode this must also never resume
      // an old completed trajectory: its follow-through endpoint can be a
      // low pose, so doing that after an unreachable ball visibly drops the
      // arm.  Hold the *measured* pose and let the next rolling prediction
      // start a fresh plan from there.
      if (require_external_goals_ || !initialized_) {
        next_plan_retry_time_s_ = time_s + 0.25;
        FillSafeHold(target);
      } else {
        next_target_time_s_ =
            periodic_target_due ? time_s + target_interval_s_
                                : time_s + 0.25;
        FillTarget(state, time_s, target);
      }
      if (require_external_goals_) {
        // Do not let one syntactically valid but unreachable network sample
        // monopolize the controller forever. Drop the failed plan while
        // holding measured position so CanAcceptExternalGoal() can consume a
        // newer rolling prediction on the next policy tick.
        have_external_goal_ = false;
        goal_dirty_ = false;
        // Do not ratchet a physically sagging measured pose into the next
        // hold target. Keep the last validated preparation pose.
        if (!safe_hold_valid_) {
          for (int joint = 0; joint < 7; ++joint)
            safe_hold_q_[joint] = state.q[kRightArmSdk[joint]];
          safe_hold_valid_ = true;
        }
        phase_ = Phase::kHold;
        // A failed plan releases the committed_strike_active_ lock so the
        // next SetGoal does not hit the locked branch and get silently
        // dropped.
        committed_strike_active_ = false;
      }
      UpdateMeasuredError(state);
      return true;
    }
    initialized_ = true;
    goal_dirty_ = false;
  }
  const std::uint64_t strike_count_before = strike_count_;
  FillTarget(state, time_s, target);
  UpdateMeasuredError(state);
  if (strike_count_ != strike_count_before) {
    SnapshotImpactTelemetry(state, time_s);
  }
  if (automatic_ready_recovery_ && require_external_goals_ && initialized_ &&
      phase_ == Phase::kHold && !preparation_plan_ && !goal_dirty_) {
    // The completed follow-through is not a useful waiting pose. Return to
    // the same forward ready stance automatically; a new packet can preempt
    // this recovery through CanAcceptExternalGoal().
    QueueReadyPlan();
  }
  return true;
}

}  // namespace a3_deploy::control
