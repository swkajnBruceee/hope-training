#include "ik_point_arm_source.hpp"

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using a3_deploy::control::ArmGoal;
using a3_deploy::control::ArmTarget;
using a3_deploy::control::IkPointArmSource;

namespace {

constexpr std::array<int, 7> kRightArmSdk = {12, 13, 14, 15, 16, 17, 18};
constexpr std::array<const char*, 7> kRightArmNames = {
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
};
constexpr std::array<const char*, 3> kWaistNames = {
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"};
constexpr std::array<const char*, 31> kSdkJointNames = {
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint"};

struct Args {
  fs::path goal;
  fs::path ready;
  fs::path planner_config;
  fs::path robot_xml;
  fs::path output_dir;
  double control_hz{100.0};
  double hold_seconds{0.10};
  double max_seconds{12.0};
  bool allow_ready_mismatch{false};
};

struct GoalInput {
  std::string goal_id{"unnamed_goal"};
  std::string frame{"a3_base_yaw"};
  std::array<double, 3> position{};
  std::array<double, 3> velocity{};
  std::array<double, 3> normal{1.0, 0.0, 0.0};
  double time_to_strike_s{1.2};
  int swing_type{-1};
  std::uint64_t sequence{1};
};

struct ReadyInput {
  std::string ready_id{"unnamed_ready"};
  std::string swing_type{"shared"};
  std::array<double, 31> q{};
  std::array<double, 31> dq{};
};

struct Sample {
  double time_s{0.0};
  std::string phase;
  bool is_hit{false};
  std::array<double, 3> waist_q{};
  std::array<double, 3> waist_dq{};
  std::array<double, 7> arm_q{};
  std::array<double, 7> arm_dq{};
  std::array<double, 3> racket_position{};
  std::array<double, 3> racket_normal{};
};

[[noreturn]] void Usage(const char* argv0, const std::string& error = {}) {
  if (!error.empty()) std::cerr << "error: " << error << "\n\n";
  std::cerr
      << "Usage:\n  " << argv0
      << " --goal GOAL.yaml --ready READY.yaml --planner-config hit_ik_point.yaml"
         " --robot-xml a3_t2d5.xml --output-dir OUT [options]\n\n"
         "Options:\n"
         "  --control-hz HZ       Sampling frequency (default 100)\n"
         "  --hold-seconds SEC    Samples after follow-through hold (default 0.10)\n"
         "  --max-seconds SEC     Hard generation timeout (default 12)\n"
         "  --allow-ready-mismatch  Allow explicit stroke goal to use a READY marked for another stroke\n";
  std::exit(error.empty() ? 0 : 2);
}

bool Finite(double x) { return std::isfinite(x); }

std::string JsonEscape(std::string_view value) {
  std::ostringstream out;
  for (const char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20U) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(ch)) << std::dec;
        } else {
          out << ch;
        }
    }
  }
  return out.str();
}

void WriteJsonNumber(std::ostream& out, double value) {
  if (Finite(value)) {
    out << std::setprecision(17) << value;
  } else {
    out << "null";
  }
}

template <std::size_t N>
std::array<double, N> ReadArray(const YAML::Node& node,
                                const std::string& name) {
  if (!node || !node.IsSequence() || node.size() != N) {
    throw std::runtime_error(name + " must contain exactly " +
                             std::to_string(N) + " values");
  }
  std::array<double, N> result{};
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = node[i].as<double>();
    if (!Finite(result[i])) {
      throw std::runtime_error(name + " contains NaN or infinity");
    }
  }
  return result;
}

double ParseDouble(std::string_view text, const std::string& name) {
  std::string owned(text);
  std::size_t consumed = 0;
  const double value = std::stod(owned, &consumed);
  if (consumed != owned.size() || !Finite(value)) {
    throw std::runtime_error("invalid " + name + ": " + owned);
  }
  return value;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) Usage(argv[0], "missing value after " + key);
      return argv[++i];
    };
    if (key == "--goal") args.goal = next();
    else if (key == "--ready") args.ready = next();
    else if (key == "--planner-config") args.planner_config = next();
    else if (key == "--robot-xml") args.robot_xml = next();
    else if (key == "--output-dir") args.output_dir = next();
    else if (key == "--control-hz") args.control_hz = ParseDouble(next(), key);
    else if (key == "--hold-seconds") args.hold_seconds = ParseDouble(next(), key);
    else if (key == "--max-seconds") args.max_seconds = ParseDouble(next(), key);
    else if (key == "--allow-ready-mismatch") args.allow_ready_mismatch = true;
    else if (key == "--help" || key == "-h") Usage(argv[0]);
    else Usage(argv[0], "unknown argument: " + key);
  }
  if (args.goal.empty() || args.ready.empty() || args.planner_config.empty() ||
      args.robot_xml.empty() || args.output_dir.empty()) {
    Usage(argv[0], "all required paths must be provided");
  }
  if (args.control_hz <= 0.0 || args.hold_seconds < 0.0 ||
      args.max_seconds <= 0.0) {
    Usage(argv[0], "frequency and durations are invalid");
  }
  return args;
}

GoalInput LoadGoal(const fs::path& path) {
  const YAML::Node root = YAML::LoadFile(path.string());
  GoalInput goal;
  if (root["goal_id"]) goal.goal_id = root["goal_id"].as<std::string>();
  if (root["frame"]) goal.frame = root["frame"].as<std::string>();
  if (goal.frame != "a3_base_yaw" && goal.frame != "initial_base_heading") {
    throw std::runtime_error(
        "goal.frame must be a3_base_yaw or initial_base_heading; this wrapper "
        "does not silently transform world coordinates");
  }
  goal.position = ReadArray<3>(root["position_m"], "position_m");
  goal.velocity =
      ReadArray<3>(root["linear_velocity_mps"], "linear_velocity_mps");
  goal.normal = ReadArray<3>(root["racket_normal"], "racket_normal");
  const double norm = std::sqrt(goal.normal[0] * goal.normal[0] +
                                goal.normal[1] * goal.normal[1] +
                                goal.normal[2] * goal.normal[2]);
  if (!Finite(norm) || norm < 1e-8) {
    throw std::runtime_error("racket_normal must be non-zero");
  }
  for (double& value : goal.normal) value /= norm;
  goal.time_to_strike_s = root["time_to_strike_s"].as<double>();
  if (!Finite(goal.time_to_strike_s) || goal.time_to_strike_s <= 0.0) {
    throw std::runtime_error("time_to_strike_s must be positive");
  }
  const std::string swing =
      root["swing_type"] ? root["swing_type"].as<std::string>() : "backhand";
  if (swing == "backhand") goal.swing_type = -1;
  else if (swing == "forehand") goal.swing_type = 1;
  else if (swing == "auto") goal.swing_type = 0;
  else throw std::runtime_error("swing_type must be backhand, forehand or auto");
  if (root["sequence"]) goal.sequence = root["sequence"].as<std::uint64_t>();
  return goal;
}

std::map<std::string, int> JointIndex() {
  std::map<std::string, int> result;
  for (std::size_t i = 0; i < kSdkJointNames.size(); ++i) {
    result.emplace(kSdkJointNames[i], static_cast<int>(i));
  }
  return result;
}

void ApplyJointMap(const YAML::Node& node, std::array<double, 31>& values,
                   const std::string& name) {
  if (!node) return;
  if (!node.IsMap()) throw std::runtime_error(name + " must be a map");
  const auto index = JointIndex();
  for (const auto& entry : node) {
    const std::string joint = entry.first.as<std::string>();
    const auto found = index.find(joint);
    if (found == index.end()) {
      throw std::runtime_error(name + " contains unknown joint: " + joint);
    }
    const double value = entry.second.as<double>();
    if (!Finite(value)) throw std::runtime_error(name + " contains non-finite value");
    values[static_cast<std::size_t>(found->second)] = value;
  }
}

ReadyInput LoadReady(const fs::path& path) {
  const YAML::Node root = YAML::LoadFile(path.string());
  ReadyInput ready;
  if (root["ready_id"]) ready.ready_id = root["ready_id"].as<std::string>();
  if (root["swing_type"]) ready.swing_type = root["swing_type"].as<std::string>();
  if (ready.swing_type != "forehand" && ready.swing_type != "backhand" &&
      ready.swing_type != "shared" && ready.swing_type != "auto") {
    throw std::runtime_error(
        "ready.swing_type must be forehand, backhand, shared or auto");
  }
  if (root["q_sdk_31"]) ready.q = ReadArray<31>(root["q_sdk_31"], "q_sdk_31");
  if (root["dq_sdk_31"]) ready.dq = ReadArray<31>(root["dq_sdk_31"], "dq_sdk_31");
  ApplyJointMap(root["joint_positions"], ready.q, "joint_positions");
  ApplyJointMap(root["joint_velocities"], ready.dq, "joint_velocities");
  for (double value : ready.q) {
    if (!Finite(value)) throw std::runtime_error("ready q contains non-finite value");
  }
  for (double value : ready.dq) {
    if (!Finite(value)) throw std::runtime_error("ready dq contains non-finite value");
  }
  return ready;
}

std::string RequestedSwingName(const GoalInput& goal) {
  return goal.swing_type < 0 ? "backhand"
       : goal.swing_type > 0 ? "forehand"
                             : "auto";
}

bool ReadyMatchesGoal(const GoalInput& goal, const ReadyInput& ready) {
  const std::string requested = RequestedSwingName(goal);
  if (ready.swing_type == "shared" || ready.swing_type == "auto") return true;
  if (requested == "auto") return false;
  return requested == ready.swing_type;
}

void ValidateReadyGoalContract(const GoalInput& goal, const ReadyInput& ready,
                               bool allow_mismatch) {
  if (ReadyMatchesGoal(goal, ready)) return;
  if (allow_mismatch) return;
  const std::string requested = RequestedSwingName(goal);
  if (requested == "auto") {
    throw std::runtime_error(
        "auto goal requires a READY marked shared/auto, or explicit per-stroke "
        "generation; use --allow-ready-mismatch only for diagnostics");
  }
  throw std::runtime_error(
      "goal requests " + requested + " but READY '" + ready.ready_id +
      "' is marked " + ready.swing_type +
      "; use the matching READY or --allow-ready-mismatch for diagnostics");
}

ArmGoal ToArmGoal(const GoalInput& input) {
  ArmGoal goal{};
  goal.valid = true;
  goal.has_cartesian_position = true;
  goal.position_m = input.position;
  goal.has_cartesian_linear_velocity = true;
  goal.linear_velocity_mps = input.velocity;
  goal.has_racket_normal = true;
  goal.racket_normal = input.normal;
  goal.has_time_to_strike = true;
  goal.time_to_strike_s = input.time_to_strike_s;
  goal.swing_type = input.swing_type;
  goal.sequence = input.sequence;
  return goal;
}

void WriteCsv(const fs::path& path, const std::vector<Sample>& samples) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open " + path.string());
  out << std::setprecision(17);
  out << "time_s,phase,is_hit_frame";
  for (const char* name : kWaistNames) out << ',' << name;
  for (const char* name : kRightArmNames) out << ',' << name;
  for (const char* name : kWaistNames) out << ',' << name << "_velocity";
  for (const char* name : kRightArmNames) out << ',' << name << "_velocity";
  out << ",racket_x_m,racket_y_m,racket_z_m"
         ",racket_normal_x,racket_normal_y,racket_normal_z\n";
  for (const Sample& sample : samples) {
    out << sample.time_s << ',' << sample.phase << ',' << (sample.is_hit ? 1 : 0);
    for (double value : sample.waist_q) out << ',' << value;
    for (double value : sample.arm_q) out << ',' << value;
    for (double value : sample.waist_dq) out << ',' << value;
    for (double value : sample.arm_dq) out << ',' << value;
    for (double value : sample.racket_position) out << ',' << value;
    for (double value : sample.racket_normal) out << ',' << value;
    out << '\n';
  }
}

template <std::size_t N>
void WriteJsonArray(std::ostream& out, const std::array<double, N>& values) {
  out << '[';
  for (std::size_t i = 0; i < N; ++i) {
    if (i) out << ',';
    WriteJsonNumber(out, values[i]);
  }
  out << ']';
}

void WriteNormalizedGoal(const fs::path& path, const GoalInput& goal) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open " + path.string());
  out << "{\n"
      << "  \"schema_version\": \"a3_canonical_strike_goal/v1\",\n"
      << "  \"goal_id\": \"" << JsonEscape(goal.goal_id) << "\",\n"
      << "  \"frame\": \"" << JsonEscape(goal.frame) << "\",\n"
      << "  \"swing_type\": \""
      << (goal.swing_type < 0 ? "backhand" : goal.swing_type > 0 ? "forehand" : "auto")
      << "\",\n  \"position_m\": ";
  WriteJsonArray(out, goal.position);
  out << ",\n  \"linear_velocity_mps\": ";
  WriteJsonArray(out, goal.velocity);
  out << ",\n  \"racket_normal\": ";
  WriteJsonArray(out, goal.normal);
  out << ",\n  \"time_to_strike_s\": " << std::setprecision(17)
      << goal.time_to_strike_s << ",\n"
      << "  \"sequence\": " << goal.sequence << "\n}\n";
}

void WriteDiagnostics(const fs::path& path, const IkPointArmSource& planner,
                      const GoalInput& goal, const ReadyInput& ready,
                      double control_hz, std::size_t hit_frame,
                      std::size_t sample_count, bool success,
                      const std::string& status) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open " + path.string());
  out << std::boolalpha;
  out << "{\n"
      << "  \"schema_version\": \"a3_ik_point_offline_diagnostics/v2\",\n"
      << "  \"success\": " << success << ",\n"
      << "  \"status\": \"" << JsonEscape(status) << "\",\n"
      << "  \"goal_id\": \"" << JsonEscape(goal.goal_id) << "\",\n"
      << "  \"requested_swing_type\": \""
      << RequestedSwingName(goal) << "\",\n"
      << "  \"selected_swing_type\": \""
      << (success ? planner.SwingName() : "unresolved") << "\",\n"
      << "  \"ready_id\": \"" << JsonEscape(ready.ready_id) << "\",\n"
      << "  \"ready_swing_type\": \""
      << JsonEscape(ready.swing_type) << "\",\n"
      << "  \"ready_goal_contract_match\": "
      << ReadyMatchesGoal(goal, ready) << ",\n"
      << "  \"control_hz\": ";
  WriteJsonNumber(out, control_hz);
  out << ",\n  \"control_dt_s\": ";
  WriteJsonNumber(out, 1.0 / control_hz);
  out << ",\n"
      << "  \"sample_count\": " << sample_count << ",\n"
      << "  \"hit_frame\": " << hit_frame << ",\n"
      << "  \"requested_strike_time_s\": ";
  WriteJsonNumber(out, planner.RequestedStrikeTimeS());
  out << ",\n  \"planned_strike_time_s\": ";
  WriteJsonNumber(out, planner.PlannedStrikeTimeS());
  out << ",\n  \"timing_extension_s\": ";
  WriteJsonNumber(out, planner.TimingExtensionS());
  out << ",\n  \"tracking_minimum_strike_time_s\": ";
  WriteJsonNumber(out, planner.TrackingMinimumStrikeTimeS());
  out << ",\n"
      << "  \"swing\": \"" << planner.SwingName() << "\",\n"
      << "  \"solve_iterations\": " << planner.SolveIterations() << ",\n"
      << "  \"solved_position_error_m\": ";
  WriteJsonNumber(out, planner.SolvedPositionErrorM());
  out << ",\n  \"solved_normal_error_deg\": ";
  WriteJsonNumber(out, planner.SolvedNormalErrorDeg());
  out << ",\n  \"velocity_solve_error_mps\": ";
  WriteJsonNumber(out, planner.LastVelocitySolveErrorMps());
  out << ",\n  \"normal_rate_rad_s\": ";
  WriteJsonNumber(out, planner.LastNormalRateRadS());
  out << ",\n  \"planning_duration_ms\": ";
  WriteJsonNumber(out, planner.LastPlanningDurationMs());
  out << ",\n  \"minimum_body_clearance_m\": ";
  WriteJsonNumber(out, planner.PlannedMinimumRacketBodyClearanceM());
  out << ",\n"
      << "  \"minimum_clearance_segment\": "
      << planner.PlannedMinimumClearanceSegment() << ",\n"
      << "  \"trajectory_reject_reason\": \""
      << JsonEscape(planner.LastTrajectoryRejectReason()) << "\",\n"
      << "  \"rejected_target_count\": " << planner.RejectedTargetCount() << ",\n"
      << "  \"solve_reject_count\": " << planner.SolveRejectCount() << ",\n"
      << "  \"velocity_reject_count\": " << planner.VelocityRejectCount() << ",\n"
      << "  \"follow_reject_count\": " << planner.FollowRejectCount() << ",\n"
      << "  \"trajectory_reject_count\": " << planner.TrajectoryRejectCount() << ",\n"
      << "  \"target_position_m\": ";
  WriteJsonArray(out, planner.TargetPosition());
  out << ",\n  \"target_velocity_mps\": ";
  WriteJsonArray(out, planner.TargetVelocity());
  out << ",\n  \"achieved_strike_velocity_mps\": ";
  WriteJsonArray(out, planner.AchievedStrikeVelocity());
  out << ",\n  \"solved_joint_position_rad\": ";
  WriteJsonArray(out, planner.SolvedJointPosition());
  out << ",\n  \"solved_joint_velocity_rad_s\": ";
  WriteJsonArray(out, planner.SolvedJointVelocity());
  out << ",\n  \"follow_joint_position_rad\": ";
  WriteJsonArray(out, planner.FollowJointPosition());
  out << "\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const GoalInput goal_input = LoadGoal(args.goal);
    const ReadyInput ready = LoadReady(args.ready);
    ValidateReadyGoalContract(goal_input, ready, args.allow_ready_mismatch);
    fs::create_directories(args.output_dir);

    IkPointArmSource planner(args.control_hz);
    std::string load_error;
    if (!planner.Load(args.planner_config.string(), args.robot_xml.string(),
                      load_error)) {
      throw std::runtime_error("planner load failed: " + load_error);
    }
    planner.RequireExternalGoals(true, false);
    planner.SetAutomaticReadyRecovery(false);

    std::array<double, 7> seed_q{};
    std::array<double, 7> seed_dq{};
    for (std::size_t i = 0; i < 7; ++i) {
      seed_q[i] = ready.q[static_cast<std::size_t>(kRightArmSdk[i])];
      seed_dq[i] = ready.dq[static_cast<std::size_t>(kRightArmSdk[i])];
    }
    planner.SeedCommandState(seed_q, seed_dq);

    if (!planner.SetGoal(ToArmGoal(goal_input))) {
      WriteDiagnostics(args.output_dir / "diagnostics.json", planner,
                       goal_input, ready, args.control_hz, 0, 0, false,
                       "SET_GOAL_REJECTED");
      throw std::runtime_error("SetGoal rejected the normalized input");
    }

    robot_io::RobotState state;
    state.q = Eigen::VectorXd(31);
    state.dq = Eigen::VectorXd(31);
    state.tau_est = Eigen::VectorXd::Zero(31);
    for (int i = 0; i < 31; ++i) {
      state.q[i] = ready.q[static_cast<std::size_t>(i)];
      state.dq[i] = ready.dq[static_cast<std::size_t>(i)];
    }

    const double dt = 1.0 / args.control_hz;
    const std::size_t max_steps =
        static_cast<std::size_t>(std::ceil(args.max_seconds * args.control_hz));
    std::vector<Sample> samples;
    samples.reserve(max_steps);
    bool plan_started = false;
    bool strike_seen = false;
    std::optional<double> first_hold_time;

    for (std::size_t step = 0; step < max_steps; ++step) {
      const double time_s = static_cast<double>(step) * dt;
      ArmTarget target{};
      if (!planner.Update(state, time_s, target)) {
        throw std::runtime_error("planner.Update returned false at step " +
                                 std::to_string(step));
      }

      const std::string phase = planner.PhaseName();
      if (planner.CommittedStrikeActive() || phase == "approach" ||
          phase == "strike" || phase == "follow_through") {
        plan_started = true;
      }
      if (planner.StrikeCount() > 0 || phase == "strike" ||
          phase == "follow_through") {
        strike_seen = true;
      }

      Sample sample;
      sample.time_s = time_s;
      sample.phase = phase;
      for (std::size_t i = 0; i < 3; ++i) {
        sample.waist_q[i] = ready.q[i];
        sample.waist_dq[i] = ready.dq[i];
      }
      std::array<double, 31> q_for_fk = ready.q;
      for (std::size_t i = 0; i < 7; ++i) {
        sample.arm_q[i] = target.q[7 + i];
        sample.arm_dq[i] = target.dq[7 + i];
        q_for_fk[static_cast<std::size_t>(kRightArmSdk[i])] = sample.arm_q[i];
      }
      if (!planner.EvaluateRacketPose(q_for_fk, sample.racket_position,
                                      sample.racket_normal)) {
        throw std::runtime_error("EvaluateRacketPose failed at step " +
                                 std::to_string(step));
      }
      samples.push_back(sample);

      // Perfect-tracking rollout. This does not claim PhysX validity; it only
      // lets the source produce a deterministic open-loop reference.
      for (std::size_t i = 0; i < 7; ++i) {
        const int sdk = kRightArmSdk[i];
        state.q[sdk] = sample.arm_q[i];
        state.dq[sdk] = sample.arm_dq[i];
      }

      if (!plan_started && phase == "hold" && planner.RejectedTargetCount() > 0) {
        break;
      }
      if (strike_seen && phase == "hold") {
        if (!first_hold_time) first_hold_time = time_s;
        if (time_s - *first_hold_time >= args.hold_seconds) break;
      }
    }

    const bool success = plan_started && strike_seen && first_hold_time.has_value();
    const double planned_hit_time = planner.PlannedStrikeTimeS();
    std::size_t hit_frame = 0;
    double best_time_error = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const double error = std::abs(samples[i].time_s - planned_hit_time);
      if (error < best_time_error) {
        best_time_error = error;
        hit_frame = i;
      }
    }
    if (!samples.empty()) samples[hit_frame].is_hit = true;

    WriteCsv(args.output_dir / "trajectory_100hz.csv", samples);
    WriteNormalizedGoal(args.output_dir / "normalized_goal.json", goal_input);
    const std::string status = success
                                   ? "KINEMATIC_CANDIDATE"
                                   : planner.RejectedTargetCount() > 0
                                         ? "PLANNING_REJECTED"
                                         : "GENERATION_TIMEOUT";
    WriteDiagnostics(args.output_dir / "diagnostics.json", planner,
                     goal_input, ready, args.control_hz, hit_frame,
                     samples.size(), success, status);

    std::cout << "status=" << status << '\n'
              << "requested_swing=" << RequestedSwingName(goal_input) << '\n'
              << "selected_swing=" << planner.SwingName() << '\n'
              << "ready_id=" << ready.ready_id << '\n'
              << "samples=" << samples.size() << '\n'
              << "hit_frame=" << hit_frame << '\n'
              << "planned_strike_time_s=" << planner.PlannedStrikeTimeS() << '\n'
              << "output_dir=" << fs::absolute(args.output_dir).string() << '\n';
    return success ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}
