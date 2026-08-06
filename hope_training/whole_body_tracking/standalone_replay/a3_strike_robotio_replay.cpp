// Project-side A3 trajectory replay.  Built as a CMake overlay against the
// vendor a3_deploy_example backend; no vendor source file is modified.
#include "a3_deploy/expand_to_backend.hpp"
#include "a3_policy_parameters.hpp"
#include "cnpy.h"
#include "robot_io/a3_layout_extra.hpp"
#include "robot_io/robot_io_backend.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kDof = robot_io::kA3Dof;

struct Payload {
  std::vector<double> timestamps_s;
  std::vector<double> q_des, dq_des, tau_ff, kp, kd;
  std::size_t count = 0;
};

struct StateRow {
  robot_io::RobotState state;
  std::int64_t local_receive_ns = 0;
};

std::int64_t MonotonicNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<double> ReadVector(const cnpy::npz_t& data, const std::string& key,
                               std::size_t expected) {
  const auto it = data.find(key);
  if (it == data.end()) throw std::runtime_error("command NPZ missing " + key);
  const auto& array = it->second;
  if (array.word_size != sizeof(double) || array.num_vals != expected) {
    throw std::runtime_error("unexpected command NPZ shape/dtype for " + key);
  }
  return array.as_vec<double>();
}

Payload LoadPayload(const std::string& path) {
  const auto data = cnpy::npz_load(path);
  const auto time_it = data.find("timestamps_s");
  if (time_it == data.end() || time_it->second.word_size != sizeof(double) ||
      time_it->second.shape.size() != 1 || time_it->second.num_vals == 0) {
    throw std::runtime_error("timestamps_s must be non-empty float64 [T]");
  }
  Payload result;
  result.count = time_it->second.num_vals;
  result.timestamps_s = time_it->second.as_vec<double>();
  for (std::size_t i = 1; i < result.timestamps_s.size(); ++i) {
    if (!(result.timestamps_s[i] > result.timestamps_s[i - 1])) {
      throw std::runtime_error("timestamps_s must be strictly increasing");
    }
  }
  const auto elements = result.count * kDof;
  result.q_des = ReadVector(data, "q_des", elements);
  result.dq_des = ReadVector(data, "dq_des", elements);
  result.tau_ff = ReadVector(data, "tau_ff", elements);
  result.kp = ReadVector(data, "kp", elements);
  result.kd = ReadVector(data, "kd", elements);
  return result;
}

robot_io::RobotCommand MakeCommand(const Payload& payload, std::size_t index) {
  robot_io::RobotCommand command;
  command.q_des.resize(kDof);
  command.dq_des.resize(kDof);
  command.tau_ff.resize(kDof);
  command.kp.resize(kDof);
  command.kd.resize(kDof);
  const std::size_t begin = index * kDof;
  for (int joint = 0; joint < kDof; ++joint) {
    command.q_des[joint] = payload.q_des[begin + joint];
    command.dq_des[joint] = payload.dq_des[begin + joint];
    command.tau_ff[joint] = payload.tau_ff[begin + joint];
    command.kp[joint] = payload.kp[begin + joint];
    command.kd[joint] = payload.kd[begin + joint];
  }
  return command;
}

void WriteRows(const std::filesystem::path& output,
               const std::vector<StateRow>& rows,
               const std::vector<std::int64_t>& command_ns,
               const Payload& payload) {
  std::filesystem::create_directories(output);
  std::ofstream states(output / "raw_state.csv");
  states << "timestamp_ns,state_data_ready_ns,state_sync_ready_ns,local_receive_ns,sync_complete,sync_aligned,sync_skew_ns";
  for (int i = 0; i < kDof; ++i) states << ",q_" << i;
  for (int i = 0; i < kDof; ++i) states << ",dq_" << i;
  for (int i = 0; i < kDof; ++i) states << ",tau_" << i;
  states << "\n";
  for (const auto& row : rows) {
    const auto& s = row.state;
    states << s.timestamp_ns << ',' << s.state_data_ready_ns << ',' << s.state_sync_ready_ns << ','
           << row.local_receive_ns << ',' << s.sync_complete << ',' << s.sync_aligned << ',' << s.sync_skew_ns;
    for (int i = 0; i < kDof; ++i) states << ',' << s.q[i];
    for (int i = 0; i < kDof; ++i) states << ',' << s.dq[i];
    for (int i = 0; i < kDof; ++i) states << ',' << s.tau_est[i];
    states << '\n';
  }
  std::ofstream commands(output / "command.csv");
  commands << "sample,command_monotonic_ns,payload_timestamp_s\n";
  for (std::size_t i = 0; i < command_ns.size(); ++i) {
    commands << i << ',' << command_ns[i] << ',' << std::setprecision(17) << payload.timestamps_s[i] << '\n';
  }
}

void Usage() {
  std::cerr << "usage: a3_strike_robotio_replay --command <canonical_command.npz> --backend-config <cfg_file_path=...,...> --out <dir> [--tail-s 1.0] [--stand-gate-file <json>] [--stand-reset-ready-file <path> --stand-reset-ack-file <json>]\n";
}

void WaitForStandGate(const std::string& stand_gate_file) {
  if (stand_gate_file.empty()) return;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (std::filesystem::exists(stand_gate_file)) {
      std::ifstream file(stand_gate_file);
      const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
      if (content.find("\"passed\":true") != std::string::npos) return;
      throw std::runtime_error("PD-STAND gate rejected: " + content);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  throw std::runtime_error("PD-STAND gate did not produce a result within 5 seconds");
}

void WaitForStandReset(const std::string& ready_file, const std::string& acknowledgement_file) {
  if (ready_file.empty() && acknowledgement_file.empty()) return;
  if (ready_file.empty() || acknowledgement_file.empty()) {
    throw std::runtime_error("--stand-reset-ready-file and --stand-reset-ack-file must be supplied together");
  }
  const std::filesystem::path ready_path(ready_file);
  std::filesystem::create_directories(ready_path.parent_path());
  std::ofstream ready(ready_path);
  if (!ready) throw std::runtime_error("cannot create reset-ready marker: " + ready_file);
  ready << "backend_ready_for_stand_reset\n";
  ready.close();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < deadline) {
    if (std::filesystem::exists(acknowledgement_file)) {
      std::ifstream acknowledgement(acknowledgement_file);
      const std::string content((std::istreambuf_iterator<char>(acknowledgement)), std::istreambuf_iterator<char>());
      if (content.find("\"passed\":true") != std::string::npos) return;
      throw std::runtime_error("stand reset was not acknowledged: " + content);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  throw std::runtime_error("stand reset acknowledgement did not arrive within 10 seconds");
}

}  // namespace

int main(int argc, char** argv) {
  std::string command_path, backend_config, out_dir, stand_gate_file, stand_reset_ready_file, stand_reset_ack_file;
  double tail_s = 1.0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--command" && i + 1 < argc) command_path = argv[++i];
    else if (arg == "--backend-config" && i + 1 < argc) backend_config = argv[++i];
    else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
    else if (arg == "--tail-s" && i + 1 < argc) tail_s = std::stod(argv[++i]);
    else if (arg == "--stand-gate-file" && i + 1 < argc) stand_gate_file = argv[++i];
    else if (arg == "--stand-reset-ready-file" && i + 1 < argc) stand_reset_ready_file = argv[++i];
    else if (arg == "--stand-reset-ack-file" && i + 1 < argc) stand_reset_ack_file = argv[++i];
    else { Usage(); return 64; }
  }
  if (command_path.empty() || backend_config.empty() || out_dir.empty() || tail_s < 0.0) {
    Usage(); return 64;
  }
  try {
    const Payload payload = LoadPayload(command_path);
    auto backend = robot_io::CreateBackend("a3");
    if (!backend || !backend->Init(backend_config) || backend->GetLayout().dof() != kDof) {
      throw std::runtime_error("RobotIOBackend init/layout failed");
    }
    try {
      std::mutex mutex;
      std::condition_variable ready_cv;
      std::vector<StateRow> rows;
      bool ready = false;
      backend->RegisterStateCallback([&](const robot_io::RobotState& state) {
        std::lock_guard<std::mutex> lock(mutex);
        rows.push_back({state, MonotonicNs()});
        ready = ready || (state.q.size() == kDof && state.dq.size() == kDof && state.tau_est.size() == kDof &&
                          state.sync_complete && state.sync_aligned);
        ready_cv.notify_all();
      });
      if (!backend->Start()) throw std::runtime_error("RobotIOBackend::Start failed");
      {
        std::unique_lock<std::mutex> lock(mutex);
        if (!ready_cv.wait_for(lock, std::chrono::seconds(5), [&] { return ready; })) {
          throw std::runtime_error("six-channel aligned RobotState did not become ready within 5 seconds");
        }
      }
      WaitForStandReset(stand_reset_ready_file, stand_reset_ack_file);
      // Explicit PD_STAND before strike.  This is the direct body-drive
      // equivalent of the deploy state-machine gateway, not a native MOTION
      // ownership claim.
      robot_io::RobotCommand stand;
      a3_deploy::ExpandToBackend(a3_default_angles, a3_pd_stand_kps, a3_pd_stand_kds, stand);
      for (int tick = 0; tick < 150; ++tick) {
        if (!backend->SendCommand(stand)) throw std::runtime_error("PD_STAND SendCommand failed");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      WaitForStandGate(stand_gate_file);
      std::vector<std::int64_t> command_ns;
      const auto begin = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i < payload.count; ++i) {
        std::this_thread::sleep_until(begin + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(payload.timestamps_s[i])));
        const auto command = MakeCommand(payload, i);
        if (!backend->SendCommand(command)) throw std::runtime_error("strike SendCommand failed");
        command_ns.push_back(MonotonicNs());
      }
      std::this_thread::sleep_for(std::chrono::duration<double>(tail_s));
      backend->Stop();
      std::vector<StateRow> copy;
      { std::lock_guard<std::mutex> lock(mutex); copy = rows; }
      WriteRows(out_dir, copy, command_ns, payload);
      std::cout << "wrote RobotIOBackend state evidence to " << out_dir << "\n";
    } catch (...) {
      backend->Stop();
      throw;
    }
  } catch (const std::exception& error) {
    std::cerr << "a3_strike_robotio_replay: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
