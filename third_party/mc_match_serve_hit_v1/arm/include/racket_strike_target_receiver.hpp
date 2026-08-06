#pragma once

#include "control_source.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace a3_deploy::control {

// ROS 2 ingress for the exact 10-D high-level racket-state contract. The HDU
// must transform table/world coordinates into a3_base_yaw before publishing.
class RacketStrikeTargetReceiver final {
 public:
  struct Config {
    std::string topic{"/racket/strike_target"};
    std::string expected_frame{"a3_base_yaw"};
    double max_sample_age_s{0.12};
    double actuation_lead_s{0.005};
    float minimum_confidence{0.0F};
  };

  explicit RacketStrikeTargetReceiver(Config config);
  ~RacketStrikeTargetReceiver();
  RacketStrikeTargetReceiver(const RacketStrikeTargetReceiver&) = delete;
  RacketStrikeTargetReceiver& operator=(const RacketStrikeTargetReceiver&) = delete;

  bool Start(std::string& error);
  void Stop() noexcept;
  // Latest-wins mailbox: each sequence is returned at most once.
  bool TakeLatest(ArmGoal& goal) noexcept;

  std::uint64_t ReceivedCount() const noexcept;
  std::uint64_t AcceptedCount() const noexcept;
  std::uint64_t InvalidCount() const noexcept;
  std::uint64_t FrameMismatchCount() const noexcept;
  std::uint64_t StaleCount() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace a3_deploy::control
