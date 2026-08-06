#include "a3_deploy/a3_policy_driver.hpp"
#include "a3_deploy/a3_policy_runtime.hpp"
#include "a3_policy_parameters.hpp"
#include "robot_io/a3_aimrt_backend.hpp"
#include "robot_io/a3_layout_extra.hpp"
#include "robot_io/robot_io_backend.hpp"
#include "control_source.hpp"
#include "model3396_config.hpp"
#include "model3396_leg_source.hpp"
#include "model3396_safety_filter.hpp"
#include "match_arm_coordinator.hpp"
#if HAS_HOPE_RACKET_COMMAND
#include "racket_strike_target_receiver.hpp"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <poll.h>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>

namespace {
volatile std::sig_atomic_t g_exit_signal = 0;

void OnSignal(int signal_number) noexcept {
  g_exit_signal = signal_number;
}

void InstallSignalHandlers() noexcept {
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);
  std::signal(SIGHUP, OnSignal);
}

std::string Resolve(const std::string& path) {
  const std::filesystem::path value(path);
  return value.is_absolute() ? value.string()
                             : std::filesystem::absolute(value).string();
}

class TerminalReplayInput final {
 public:
  TerminalReplayInput() noexcept {
    tty_fd_ = open("/dev/tty", O_RDWR | O_NOCTTY);
    if (tty_fd_ < 0) tty_fd_ = STDIN_FILENO;
    if (!isatty(tty_fd_) || tcgetattr(tty_fd_, &saved_) != 0) return;
    termios raw = saved_;
    raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    enabled_ = tcsetattr(tty_fd_, TCSANOW, &raw) == 0;
  }

  ~TerminalReplayInput() {
    if (enabled_) tcsetattr(tty_fd_, TCSANOW, &saved_);
    if (tty_fd_ >= 0 && tty_fd_ != STDIN_FILENO) close(tty_fd_);
  }

  bool SpacePressed() const noexcept {
    if (!enabled_) return false;
    pollfd fd{tty_fd_, POLLIN, 0};
    bool pressed = false;
    while (poll(&fd, 1, 0) > 0 && (fd.revents & POLLIN)) {
      char c = 0;
      if (read(tty_fd_, &c, 1) != 1) break;
      pressed = pressed || c == ' ';
    }
    return pressed;
  }

  bool Enabled() const noexcept { return enabled_; }

 private:
  int tty_fd_{-1};
  termios saved_{};
  bool enabled_{false};
};

enum class RunState : std::uint8_t {
  kRunning,
  kReturning,
  kDone,
};

constexpr double kCommandHz = 100.0;
constexpr double kLegPolicyOutputGain = 3.5;
constexpr char kPolicyModelPath[] = "model/model_3396_lower_stage_a_policy.onnx";
constexpr char kAimrtConfigPath[] = "config/a3_aimrt_config.iceoryx.yaml";
constexpr char kUpperTrajectoryPath[] = "arm/serve_upper_trajectory.bin";
constexpr char kRallyReadyPosePath[] = "arm/rally_ready_pose.yaml";
constexpr char kHitConfigPath[] = "arm/hit_ik_point.yaml";
constexpr char kRobotXmlPath[] = "models/hit/kinematics/a3_t2d5.xml";
}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  using namespace a3_deploy;
  using namespace a3_deploy::control;
  using namespace a3_deploy::model3396;

  InstallSignalHandlers();

  const std::string model_path = Resolve(kPolicyModelPath);
  const std::string aimrt_cfg = Resolve(kAimrtConfigPath);
  const std::string trajectory_path = Resolve(kUpperTrajectoryPath);
  const std::string rally_pose_path = Resolve(kRallyReadyPosePath);
  const std::string hit_config_path = Resolve(kHitConfigPath);
  const std::string robot_xml_path = Resolve(kRobotXmlPath);
  const double policy_hz = kCommandHz;

  std::string backend_config =
      "cfg_file_path=" + aimrt_cfg +
      ",sync_mode=min_skew_pair" +
      ",sync_hz=" + std::to_string(policy_hz * 2.0) +
      ",leg_only=true,arm_enabled=true,waist_enabled=true,waist_zero=false";

  auto backend = robot_io::CreateBackend("a3");
  if (!backend) {
    std::cerr << "model3396: A3 backend unavailable\n";
    return 2;
  }
  if (!backend->Init(backend_config)) {
    std::cerr << "model3396: backend init failed\n";
    return 3;
  }

  A3PolicyRuntimeOptions runtime_options;
  runtime_options.backend = "ort_cpu";
  auto leg_policy = CreateA3PolicyRuntime(runtime_options);
  if (!leg_policy ||
      !leg_policy->Initialize(model_path, runtime_options) ||
      leg_policy->GetInputDimension() != kObsDim ||
      leg_policy->GetActionDimension() != kActionDim) {
    std::cerr << "model3396: ONNX contract mismatch\n";
    backend->Stop();
    return 4;
  }

  auto leg_source = std::make_unique<Model3396LegSource>(
      std::move(leg_policy), kLegPolicyOutputGain, policy_hz);
  MatchArmCoordinator arm_source;
  std::string arm_error;
  if (!arm_source.Load(trajectory_path, rally_pose_path, hit_config_path,
                       robot_xml_path, arm_error)) {
    std::cerr << "model3396: " << arm_error << "\n";
    backend->Stop();
    return 4;
  }

#if HAS_HOPE_RACKET_COMMAND
  RacketStrikeTargetReceiver::Config receiver_config{};
  receiver_config.topic = "/racket/strike_target";
  receiver_config.expected_frame = "a3_base_yaw";
  receiver_config.max_sample_age_s = 0.12;
  receiver_config.actuation_lead_s = 0.005;
  auto strike_receiver =
      std::make_unique<RacketStrikeTargetReceiver>(receiver_config);
  if (!strike_receiver->Start(arm_error)) {
    std::cerr << "model3396: " << arm_error << "\n";
    backend->Stop();
    return 4;
  }
#else
  std::unique_ptr<int> strike_receiver;
  std::cerr << "model3396 warning: built without HOPE racket target support; "
               "serve/rally state machine remains available\n";
#endif

  std::cout << "model3396 ready: command=100Hz leg_policy=50Hz "
               "arm=match_serve_then_10d_strike gain="
            << kLegPolicyOutputGain << "\n";

  SafetyFilter safety;
  // MuJoCo is born in the desired upper-body pose. The deployer only holds
  // the first valid observed state; it does not transition the upper body.
  constexpr double kStartupLegBlendS = 2.0;
  // Positive waist pitch is forward lean; negative is backward lean.
  constexpr double kWaistPitchBackwardTargetRad = -5.0 * M_PI / 180.0;
  std::array<double, 31> startup_q{};
  std::array<double, 12> startup_leg_q{};
  bool startup_pose_valid = false;
  constexpr double kShutdownBlendS = 2.0;
  constexpr double kShutdownHoldS = 0.5;
  constexpr double kShutdownPositionToleranceRad = 0.08;
  constexpr double kShutdownVelocityToleranceRadS = 0.25;
  const std::uint64_t shutdown_blend_ticks = static_cast<std::uint64_t>(
      std::max(1.0, std::ceil(kShutdownBlendS * policy_hz)));
  const std::uint64_t shutdown_hold_ticks = static_cast<std::uint64_t>(
      std::max(1.0, std::ceil(kShutdownHoldS * policy_hz)));
  std::atomic<RunState> run_state{RunState::kRunning};
  bool shutdown_initialized = false;
  std::uint64_t shutdown_frame = 0;
  std::uint64_t shutdown_settle_frame = 0;
  std::array<double, 31> shutdown_from_q{};

  A3PolicyDriverOptions dopt;
  dopt.policy_hz = policy_hz;
  dopt.send_safe_halt_before_first_command = false;
  auto command = [&](std::uint64_t tick, const robot_io::RobotState& state,
                     robot_io::RobotCommand& out) noexcept -> bool {
    if (state.q.size() < 31 || state.dq.size() < 31) return false;
    if (!startup_pose_valid) {
      for (int sdk = 0; sdk < 31; ++sdk) startup_q[sdk] = state.q[sdk];
      for (int i = 0; i < 6; ++i) {
        startup_leg_q[i] = state.q[kLeftLegBackend[i]];
        startup_leg_q[i + 6] = state.q[kRightLegBackend[i]];
      }
      startup_pose_valid = true;
    }
    if (run_state.load(std::memory_order_acquire) != RunState::kRunning) {
      if (!shutdown_initialized) {
        for (int sdk = 0; sdk < 31; ++sdk) shutdown_from_q[sdk] = state.q[sdk];
        shutdown_frame = 0;
        shutdown_settle_frame = 0;
        shutdown_initialized = true;
        std::cerr << "model3396 shutdown: smooth return to immutable startup pose over "
                  << kShutdownBlendS << "s, then hold for "
                  << kShutdownHoldS << "s\n";
      }
      // Advance only after a valid command frame. A transient state-sync gap
      // may extend wall-clock duration, but cannot skip interpolation frames.
      const double u = std::clamp(
          static_cast<double>(std::min(shutdown_frame, shutdown_blend_ticks)) /
              static_cast<double>(shutdown_blend_ticks), 0.0, 1.0);
      const double s = u * u * (3.0 - 2.0 * u);
      out.q_des = Eigen::VectorXd::Zero(31);
      out.dq_des = Eigen::VectorXd::Zero(31);
      out.tau_ff = Eigen::VectorXd::Zero(31);
      out.kp = Eigen::VectorXd::Zero(31);
      out.kd = Eigen::VectorXd::Zero(31);
      const double ds_dt = (u > 0.0 && u < 1.0)
          ? 6.0 * u * (1.0 - u) / kShutdownBlendS : 0.0;
      for (int sdk = 0; sdk < 31; ++sdk) {
        out.q_des[sdk] = (1.0 - s) * shutdown_from_q[sdk] + s * startup_q[sdk];
        out.dq_des[sdk] = ds_dt * (startup_q[sdk] - shutdown_from_q[sdk]);
      }
      for (int i = 0; i < robot_io::kA3PolicyDof; ++i) {
        const int sdk = robot_io::kA3PolicyToSdkIdx[i];
        out.kp[sdk] = a3_pd_stand_kps[i];
        out.kd[sdk] = a3_pd_stand_kds[i];
      }
      out.kp[3] = out.kp[4] = 0.0;
      out.kd[3] = out.kd[4] = 0.0;
      // Do not run the generic effort estimator limiter here. On a
      // gravity-loaded waist it can observe effort above its software cap and
      // rewrite q_des back to state.q forever, making return impossible. This
      // path is already a bounded 2 s cubic position trajectory with zero
      // feed-forward effort; actuator-side limits remain active.
      for (int sdk = 0; sdk < 31; ++sdk) {
        if (!std::isfinite(out.q_des[sdk]) ||
            !std::isfinite(out.dq_des[sdk]) ||
            !std::isfinite(out.tau_ff[sdk]) ||
            !std::isfinite(out.kp[sdk]) ||
            !std::isfinite(out.kd[sdk])) {
          return false;
        }
      }
      if (shutdown_frame < shutdown_blend_ticks) {
        ++shutdown_frame;
      } else {
        double max_position_error = 0.0;
        double max_velocity = 0.0;
        for (int i = 0; i < robot_io::kA3PolicyDof; ++i) {
          const int sdk = robot_io::kA3PolicyToSdkIdx[i];
          max_position_error = std::max(
              max_position_error, std::abs(state.q[sdk] - startup_q[sdk]));
          max_velocity = std::max(max_velocity, std::abs(state.dq[sdk]));
        }
        if (max_position_error <= kShutdownPositionToleranceRad &&
            max_velocity <= kShutdownVelocityToleranceRadS) {
          ++shutdown_settle_frame;
        } else {
          shutdown_settle_frame = 0;
        }
      }
      if (shutdown_settle_frame >= shutdown_hold_ticks) {
        RunState expected = RunState::kReturning;
        if (run_state.compare_exchange_strong(
                expected, RunState::kDone, std::memory_order_acq_rel)) {
          std::cerr << "model3396 shutdown: startup pose return sequence complete\n";
        }
      }
      return true;
    }
    control::LegTarget leg_target{};
    if (!leg_source->Update(state, leg_target)) return false;
    const double alpha = std::min(
        1.0, static_cast<double>(tick) /
                 std::max(1.0, kStartupLegBlendS * policy_hz));
    out.q_des = Eigen::VectorXd::Zero(31); out.dq_des = Eigen::VectorXd::Zero(31);
    out.tau_ff = Eigen::VectorXd::Zero(31); out.kp = Eigen::VectorXd::Zero(31); out.kd = Eigen::VectorXd::Zero(31);
    // The policy overwrites only the twelve leg targets below.
    for (int i = 0; i < 31; ++i) { out.q_des[i] = startup_q[i]; }
    // The model's final two action channels are masked and do not control the
    // waist. Keep yaw/roll at zero and smoothly move waist pitch to 5 degrees
    // backward from the measured entry pose.
    for (int sdk = robot_io::kA3WaistStart;
         sdk < robot_io::kA3WaistStart + robot_io::kA3WaistCount; ++sdk) {
      const double target = sdk == robot_io::kA3PolicyToSdkIdx[2]
          ? kWaistPitchBackwardTargetRad : 0.0;
      out.q_des[sdk] = (1.0 - alpha) * startup_q[sdk] + alpha * target;
    }
    for (int i = 0; i < 6; ++i) {
      out.q_des[kLeftLegBackend[i]] =
          (1.0 - alpha) * startup_leg_q[i] + alpha * leg_target.q[i];
      out.q_des[kRightLegBackend[i]] =
          (1.0 - alpha) * startup_leg_q[i + 6] +
          alpha * leg_target.q[i + 6];
    }
    // The match coordinator owns arm timing: one automatic serve, independent
    // left-arm return, a preemptible right-arm idle attractor, and committed
    // 10-D strike/follow-through execution.
    const double arm_motion_time_s = static_cast<double>(tick) / policy_hz;
#if HAS_HOPE_RACKET_COMMAND
    if (strike_receiver && arm_source.CanAcceptStrikeGoal()) {
      control::ArmGoal strike_goal{};
      if (strike_receiver->TakeLatest(strike_goal)) {
        if (!arm_source.SetStrikeGoal(strike_goal)) {
          std::cerr << "model3396 match: rejected 10D target seq="
                    << strike_goal.sequence << "\n";
        }
      }
    }
#endif
    control::ArmTarget arm_target{};
    if (!arm_source.Update(state, arm_motion_time_s, arm_target)) return false;
    for (int i = 0; i < 14; ++i) {
      const int sdk = 5 + i;
      out.q_des[sdk] = arm_target.q[i];
      out.dq_des[sdk] = alpha * arm_target.dq[i];
      const int policy_i = 3 + i;
      if (arm_target.has_arm_gains && i >= 7 &&
          arm_target.kp[i] > 0.0 && arm_target.kd[i] >= 0.0) {
        out.kp[sdk] = alpha * arm_target.kp[i];
        out.kd[sdk] = alpha * arm_target.kd[i];
      } else {
        out.kp[sdk] = alpha * a3_pd_stand_kps[policy_i];
        out.kd[sdk] = alpha * a3_pd_stand_kds[policy_i];
      }
    }
    // Ramp waist and leg gains with the same measured-pose -> target alpha.
    // Arms follow the independently loaded reference; neck remains unpublished.
    for (int i = 0; i < 3; ++i) {
      const int sdk = robot_io::kA3PolicyToSdkIdx[i];
      out.kp[sdk] = alpha * a3_pd_stand_kps[i];
      out.kd[sdk] = alpha * a3_pd_stand_kds[i];
    }
    for (int i = 17; i < 29; ++i) {
      const int sdk = robot_io::kA3PolicyToSdkIdx[i];
      out.kp[sdk] = alpha * a3_pd_stand_kps[i];
      out.kd[sdk] = alpha * a3_pd_stand_kds[i];
    }
    out.kp[3] = out.kp[4] = 0.0; out.kd[3] = out.kd[4] = 0.0;
    if (!safety.LimitTorque(state, out, 0.8)) return false;
    return true;
  };
  if (!backend->Start()) {
    std::cerr << "model3396: backend start failed\n";
    return 5;
  }

  auto driver = std::make_unique<A3PolicyDriver>(*backend, command, dopt);
  if (!driver->StartDriver()) {
    backend->Stop();
    return 6;
  }

  // Reinstall after backend startup so every termination request enters the
  // same ordered path: return first, then stop the driver/backend.
  InstallSignalHandlers();

  TerminalReplayInput replay_input;
  std::cout << "model3396 match: serving once automatically, then rally mode";
  if (replay_input.Enabled()) {
    std::cout << "; SPACE=serve again, Ctrl+C=return and exit";
  }
  std::cout << "\n";

  while (g_exit_signal == 0) {
    if (replay_input.SpacePressed()) {
      if (arm_source.RequestServe()) {
        std::cout << "model3396 match: serve requested" << std::endl;
      } else {
        std::cout << "model3396 match: serve ignored; one serve is already pending"
                  << std::endl;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  std::cerr << "model3396 shutdown: signal " << g_exit_signal
            << " received; returning before shutdown\n";
  run_state.store(RunState::kReturning, std::memory_order_release);
  while (run_state.load(std::memory_order_acquire) != RunState::kDone) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

#if HAS_HOPE_RACKET_COMMAND
  if (strike_receiver) strike_receiver->Stop();
#endif
  driver->StopDriver();
  backend->Stop();
  std::cout << "model3396 stopped after reset: ticks="
            << driver->PolicyTickCount()
            << " safe_halts=" << driver->SafeHaltCount()
            << "\n";
  return 0;
}
