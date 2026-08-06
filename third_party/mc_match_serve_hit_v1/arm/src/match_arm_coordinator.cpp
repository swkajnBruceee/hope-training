#include "match_arm_coordinator.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace a3_deploy::control {
namespace {

template <std::size_t N>
bool ReadArray(const YAML::Node& node, std::array<double, N>& out,
               const char* name, std::string& error) {
  if (!node || !node.IsSequence() || node.size() != N) {
    error = std::string(name) + " must contain exactly " +
            std::to_string(N) + " values";
    return false;
  }
  for (std::size_t i = 0; i < N; ++i) {
    out[i] = node[i].as<double>();
    if (!std::isfinite(out[i])) {
      error = std::string(name) + " contains NaN or infinity";
      return false;
    }
  }
  return true;
}

constexpr double kMinimumJerkPeakVelocityFactor = 1.875;

}  // namespace

bool MatchArmCoordinator::Load(const std::string& serve_path,
                               const std::string& rally_pose_path,
                               const std::string& hit_config_path,
                               const std::string& robot_xml,
                               std::string& error) {
  if (!serve_trajectory_.Load(serve_path, error)) return false;
  std::array<double, 14> dq{};
  if (!serve_trajectory_.Sample(0.0, serve_first_q_, dq) ||
      !serve_trajectory_.Sample(serve_trajectory_.DurationS(),
                                serve_final_q_, dq)) {
    error = "match coordinator cannot sample serve endpoints";
    return false;
  }
  if (!LoadRallyPose(rally_pose_path, error)) return false;

  hit_source_ = std::make_unique<IkPointArmSource>(100.0);
  if (!hit_source_->Load(hit_config_path, robot_xml, error)) return false;
  // The match coordinator owns the right-arm ready transition/recovery. The IK
  // source executes only validated live strikes and never generates targets.
  hit_source_->RequireExternalGoals(true, false);
  hit_source_->SetAutomaticReadyRecovery(false);

  initialized_ = false;
  state_ = State::kInitialServeEntry;
  serve_requested_.store(false, std::memory_order_release);
  return true;
}

bool MatchArmCoordinator::LoadRallyPose(const std::string& path,
                                        std::string& error) {
  try {
    const YAML::Node root = YAML::LoadFile(path);
    if (!ReadArray(root["right_q"], rally_right_q_, "right_q", error)) {
      return false;
    }
    if (root["serve_to_rally_s"])
      serve_to_rally_s_ = root["serve_to_rally_s"].as<double>();
    if (root["hit_to_rally_s"])
      hit_to_rally_s_ = root["hit_to_rally_s"].as<double>();
    if (root["rally_to_serve_s"])
      rally_to_serve_s_ = root["rally_to_serve_s"].as<double>();
    if (root["serve_entry_hold_s"])
      serve_entry_hold_s_ = root["serve_entry_hold_s"].as<double>();
    if (root["right_ready_max_velocity_rad_s"])
      right_ready_max_velocity_rad_s_ =
          root["right_ready_max_velocity_rad_s"].as<double>();
    if (root["left_return_s"])
      left_return_nominal_s_ = root["left_return_s"].as<double>();
    if (root["left_return_max_s"])
      left_return_max_s_ = root["left_return_max_s"].as<double>();
    if (root["left_return_max_velocity_rad_s"])
      left_return_max_velocity_rad_s_ =
          root["left_return_max_velocity_rad_s"].as<double>();

    if (!std::isfinite(serve_to_rally_s_) || serve_to_rally_s_ < 0.20 ||
        serve_to_rally_s_ > 1.0 ||
        !std::isfinite(hit_to_rally_s_) || hit_to_rally_s_ < 0.20 ||
        hit_to_rally_s_ > 1.0 ||
        !std::isfinite(rally_to_serve_s_) || rally_to_serve_s_ < 0.40 ||
        rally_to_serve_s_ > 3.0 ||
        !std::isfinite(serve_entry_hold_s_) || serve_entry_hold_s_ < 0.0 ||
        serve_entry_hold_s_ > 1.0 ||
        !std::isfinite(right_ready_max_velocity_rad_s_) ||
        right_ready_max_velocity_rad_s_ < 1.0 ||
        right_ready_max_velocity_rad_s_ > 8.0 ||
        !std::isfinite(left_return_nominal_s_) ||
        left_return_nominal_s_ < 0.30 || left_return_nominal_s_ > 0.70 ||
        !std::isfinite(left_return_max_s_) ||
        left_return_max_s_ < left_return_nominal_s_ ||
        left_return_max_s_ > 0.70 ||
        !std::isfinite(left_return_max_velocity_rad_s_) ||
        left_return_max_velocity_rad_s_ < 1.0 ||
        left_return_max_velocity_rad_s_ > 10.0) {
      error = "rally transition settings are invalid";
      return false;
    }
    return true;
  } catch (const std::exception& e) {
    error = "rally ready pose: " + std::string(e.what());
    return false;
  }
}

bool MatchArmCoordinator::RequestServe() noexcept {
  return !serve_requested_.exchange(true, std::memory_order_acq_rel);
}

bool MatchArmCoordinator::CanAcceptStrikeGoal() const noexcept {
  return hit_source_ && !serve_requested_.load(std::memory_order_acquire) &&
         (state_ == State::kServeToRally ||
          state_ == State::kRallyReady ||
          state_ == State::kRecoverToRally) &&
         hit_source_->CanAcceptExternalGoal();
}

bool MatchArmCoordinator::SetStrikeGoal(const ArmGoal& goal) noexcept {
  if (!CanAcceptStrikeGoal()) return false;
  std::array<double, 7> q_right{};
  std::array<double, 7> dq_right{};
  for (std::size_t i = 0; i < 7; ++i) {
    q_right[i] = last_command_q_[7 + i];
    dq_right[i] = last_command_dq_[7 + i];
  }
  hit_source_->SeedCommandState(q_right, dq_right);
  if (!hit_source_->SetGoal(goal)) return false;
  strike_count_at_start_ = hit_source_->StrikeCount();
  reject_count_at_start_ = hit_source_->RejectedTargetCount();
  state_ = State::kStriking;
  phase_origin_s_ = 0.0;  // IK source uses absolute match time.
  std::cout << "model3396 match: strike accepted immediately from current "
               "command state seq="
            << goal.sequence << " t_hit=" << goal.time_to_strike_s
            << std::endl;
  return true;
}

void MatchArmCoordinator::MinimumJerk(
    const std::array<double, 14>& q0,
    const std::array<double, 14>& q1,
    double elapsed_s, double duration_s,
    std::array<double, 14>& q,
    std::array<double, 14>& dq) noexcept {
  const double u = std::clamp(elapsed_s / std::max(1.0e-6, duration_s),
                              0.0, 1.0);
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;
  const double s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
  const double ds = (30.0 * u2 - 60.0 * u3 + 30.0 * u4) /
                    std::max(1.0e-6, duration_s);
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = (1.0 - s) * q0[i] + s * q1[i];
    dq[i] = ds * (q1[i] - q0[i]);
  }
}

void MatchArmCoordinator::MinimumJerkRight(
    const std::array<double, 14>& q0,
    const std::array<double, 7>& q1_right,
    double elapsed_s, double duration_s,
    std::array<double, 14>& q,
    std::array<double, 14>& dq) noexcept {
  q = q0;
  dq.fill(0.0);
  const double u = std::clamp(elapsed_s / std::max(1.0e-6, duration_s),
                              0.0, 1.0);
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;
  const double s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
  const double ds = (30.0 * u2 - 60.0 * u3 + 30.0 * u4) /
                    std::max(1.0e-6, duration_s);
  for (std::size_t joint = 0; joint < 7; ++joint) {
    const std::size_t i = 7 + joint;
    q[i] = (1.0 - s) * q0[i] + s * q1_right[joint];
    dq[i] = ds * (q1_right[joint] - q0[i]);
  }
}

void MatchArmCoordinator::MinimumJerkLeft(
    const std::array<double, 7>& q0,
    const std::array<double, 7>& q1_left,
    double elapsed_s, double duration_s,
    std::array<double, 14>& q,
    std::array<double, 14>& dq) noexcept {
  const double u = std::clamp(elapsed_s / std::max(1.0e-6, duration_s),
                              0.0, 1.0);
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;
  const double s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
  const double ds = (30.0 * u2 - 60.0 * u3 + 30.0 * u4) /
                    std::max(1.0e-6, duration_s);
  for (std::size_t i = 0; i < 7; ++i) {
    q[i] = (1.0 - s) * q0[i] + s * q1_left[i];
    dq[i] = ds * (q1_left[i] - q0[i]);
  }
}

std::array<double, 14> MatchArmCoordinator::MeasuredArms(
    const robot_io::RobotState& state) noexcept {
  std::array<double, 14> measured{};
  for (int i = 0; i < 14; ++i) measured[i] = state.q[5 + i];
  return measured;
}

void MatchArmCoordinator::FillRightReady(ArmTarget& target) const noexcept {
  target = ArmTarget{};
  for (int i = 0; i < 7; ++i) {
    target.q[i] = startup_left_q_[i];
    target.dq[i] = 0.0;
    target.q[7 + i] = rally_right_q_[i];
    target.dq[7 + i] = 0.0;
  }
}

double MatchArmCoordinator::RightTransitionDuration(
    const std::array<double, 14>& from,
    double nominal_duration_s) const noexcept {
  double max_delta = 0.0;
  for (std::size_t i = 0; i < 7; ++i) {
    max_delta = std::max(max_delta,
                         std::abs(rally_right_q_[i] - from[7 + i]));
  }
  const double velocity_limited =
      kMinimumJerkPeakVelocityFactor * max_delta /
      std::max(1.0e-6, right_ready_max_velocity_rad_s_);
  return std::max(nominal_duration_s, velocity_limited);
}

double MatchArmCoordinator::LeftReturnDuration(
    const std::array<double, 14>& from) const noexcept {
  double max_delta = 0.0;
  for (std::size_t i = 0; i < 7; ++i) {
    max_delta = std::max(max_delta,
                         std::abs(startup_left_q_[i] - from[i]));
  }
  const double velocity_limited =
      kMinimumJerkPeakVelocityFactor * max_delta /
      std::max(1.0e-6, left_return_max_velocity_rad_s_);
  return std::min(left_return_max_s_,
                  std::max(left_return_nominal_s_, velocity_limited));
}

void MatchArmCoordinator::BeginLeftReturn(
    const std::array<double, 14>& from, double time_s) noexcept {
  for (std::size_t i = 0; i < 7; ++i) {
    left_return_from_q_[i] = from[i];
  }
  left_return_duration_s_ = LeftReturnDuration(from);
  left_return_origin_s_ = time_s;
  left_return_active_ = true;
  left_return_complete_logged_ = false;

  double max_delta = 0.0;
  for (std::size_t i = 0; i < 7; ++i) {
    max_delta = std::max(max_delta,
                         std::abs(startup_left_q_[i] - from[i]));
  }
  const double required_s =
      kMinimumJerkPeakVelocityFactor * max_delta /
      std::max(1.0e-6, left_return_max_velocity_rad_s_);
  std::cout << "model3396 match: left arm returning to startup pose in "
            << left_return_duration_s_ << " s";
  if (required_s > left_return_max_s_) {
    std::cout << " (capped at 0.7 s by competition requirement)";
  }
  std::cout << std::endl;
}

void MatchArmCoordinator::ApplyLeftReturn(double time_s,
                                          ArmTarget& target) noexcept {
  if (left_return_active_) {
    const double elapsed = std::max(0.0, time_s - left_return_origin_s_);
    MinimumJerkLeft(left_return_from_q_, startup_left_q_, elapsed,
                    left_return_duration_s_, target.q, target.dq);
    if (elapsed >= left_return_duration_s_) {
      left_return_active_ = false;
      for (std::size_t i = 0; i < 7; ++i) {
        target.q[i] = startup_left_q_[i];
        target.dq[i] = 0.0;
      }
    }
  } else {
    for (std::size_t i = 0; i < 7; ++i) {
      target.q[i] = startup_left_q_[i];
      target.dq[i] = 0.0;
    }
  }

  if (!left_return_active_ && !left_return_complete_logged_) {
    left_return_complete_logged_ = true;
    std::cout << "model3396 match: left arm startup pose restored and held"
              << std::endl;
  }
}

void MatchArmCoordinator::BeginServeEntry(
    const std::array<double, 14>& from, double time_s, bool initial) noexcept {
  transition_from_q_ = from;
  transition_duration_s_ = initial ? initial_serve_entry_s_ : rally_to_serve_s_;
  phase_origin_s_ = time_s;
  left_return_active_ = false;
  left_return_complete_logged_ = false;
  state_ = State::kInitialServeEntry;
}

void MatchArmCoordinator::BeginRightTransition(
    State next, const std::array<double, 14>& from, double time_s,
    double nominal_duration_s) noexcept {
  transition_from_q_ = from;
  transition_duration_s_ = RightTransitionDuration(from, nominal_duration_s);
  phase_origin_s_ = time_s;
  state_ = next;
  std::cout << "model3396 match: right-arm ready transition "
            << transition_duration_s_ << " s";
  if (transition_duration_s_ > 0.50) {
    std::cout << " (extended by joint-speed bound)";
  }
  std::cout << std::endl;
}

bool MatchArmCoordinator::Update(const robot_io::RobotState& state,
                                 double time_s,
                                 ArmTarget& target) noexcept {
  if (!hit_source_ || !std::isfinite(time_s) || state.q.size() < 19 ||
      state.dq.size() < 19) return false;

  if (!initialized_) {
    initialized_ = true;
    transition_from_q_ = MeasuredArms(state);
    last_command_q_ = transition_from_q_;
    last_command_dq_.fill(0.0);
    for (std::size_t i = 0; i < 7; ++i) {
      startup_left_q_[i] = transition_from_q_[i];
      left_return_from_q_[i] = transition_from_q_[i];
    }
    transition_duration_s_ = initial_serve_entry_s_;
    phase_origin_s_ = time_s;
    state_ = State::kInitialServeEntry;
    std::cout << "model3396 match: entering serve pose" << std::endl;
  }

  // A queued serve may preempt only idle-attractor/recovery motion. It never
  // cuts through an active serve or a committed strike/follow-through.
  if (serve_requested_.load(std::memory_order_acquire) &&
      (state_ == State::kServeToRally ||
       state_ == State::kRallyReady ||
       state_ == State::kRecoverToRally)) {
    serve_requested_.store(false, std::memory_order_release);
    BeginServeEntry(last_command_q_, time_s, false);
    std::cout << "model3396 match: SPACE accepted; current idle/recovery "
                 "motion preempted by next serve"
              << std::endl;
  }

  const double elapsed = std::max(0.0, time_s - phase_origin_s_);
  switch (state_) {
    case State::kInitialServeEntry:
      MinimumJerk(transition_from_q_, serve_first_q_, elapsed,
                  transition_duration_s_, target.q, target.dq);
      if (elapsed >= transition_duration_s_) {
        target.q = serve_first_q_;
        target.dq.fill(0.0);
        state_ = State::kServeEntryHold;
        phase_origin_s_ = time_s;
      }
      break;

    case State::kServeEntryHold:
      target.q = serve_first_q_;
      target.dq.fill(0.0);
      if (elapsed >= serve_entry_hold_s_) {
        state_ = State::kServing;
        phase_origin_s_ = time_s;
        std::cout << "model3396 match: serve started" << std::endl;
      }
      break;

    case State::kServing:
      if (!serve_trajectory_.Sample(elapsed, target.q, target.dq)) return false;
      if (elapsed >= serve_trajectory_.DurationS()) {
        target.q = serve_final_q_;
        target.dq.fill(0.0);
        BeginLeftReturn(serve_final_q_, time_s);
        BeginRightTransition(State::kServeToRally, serve_final_q_, time_s,
                             serve_to_rally_s_);
        std::cout << "model3396 match: serve complete; both arms entering "
                     "independent recovery"
                  << std::endl;
      }
      break;

    case State::kServeToRally:
      MinimumJerkRight(transition_from_q_, rally_right_q_, elapsed,
                       transition_duration_s_, target.q, target.dq);
      ApplyLeftReturn(time_s, target);
      if (elapsed >= transition_duration_s_) {
        FillRightReady(target);
        ApplyLeftReturn(time_s, target);
        state_ = State::kRallyReady;
        phase_origin_s_ = time_s;
        std::cout << "model3396 match: right arm rally ready; waiting for 10D target"
                  << std::endl;
      }
      break;

    case State::kRallyReady:
      FillRightReady(target);
      ApplyLeftReturn(time_s, target);
      break;

    case State::kStriking: {
      ArmTarget hit{};
      if (!hit_source_->Update(state, time_s, hit)) return false;
      // The strike planner owns only the right arm. The left arm continues its
      // independent startup-pose return/hold and never gates strike timing.
      target = hit;
      ApplyLeftReturn(time_s, target);
      const bool strike_finished =
          hit_source_->StrikeCount() > strike_count_at_start_ &&
          !hit_source_->CommittedStrikeActive();
      const bool planning_rejected =
          hit_source_->RejectedTargetCount() > reject_count_at_start_ &&
          !hit_source_->CommittedStrikeActive();
      if (strike_finished || planning_rejected) {
        BeginRightTransition(State::kRecoverToRally, target.q, time_s,
                             hit_to_rally_s_);
        std::cout << "model3396 match: "
                  << (strike_finished ? "strike complete" : "strike rejected")
                  << "; recovering right arm" << std::endl;
      }
      break;
    }

    case State::kRecoverToRally:
      MinimumJerkRight(transition_from_q_, rally_right_q_, elapsed,
                       transition_duration_s_, target.q, target.dq);
      ApplyLeftReturn(time_s, target);
      if (elapsed >= transition_duration_s_) {
        FillRightReady(target);
        ApplyLeftReturn(time_s, target);
        state_ = State::kRallyReady;
        phase_origin_s_ = time_s;
        std::cout << "model3396 match: right arm rally ready" << std::endl;
      }
      break;
  }

  last_command_q_ = target.q;
  last_command_dq_ = target.dq;
  return true;
}

const char* MatchArmCoordinator::StateName() const noexcept {
  switch (state_) {
    case State::kInitialServeEntry: return "serve_entry";
    case State::kServeEntryHold: return "serve_hold";
    case State::kServing: return "serving";
    case State::kServeToRally: return "serve_to_rally";
    case State::kRallyReady: return "rally_ready";
    case State::kStriking: return "striking";
    case State::kRecoverToRally: return "recover_to_rally";
  }
  return "unknown";
}

}  // namespace a3_deploy::control
