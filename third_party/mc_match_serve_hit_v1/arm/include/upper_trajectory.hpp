#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace a3_deploy::control {

// Compact, dependency-free runtime representation of the converted 100 Hz
// upper-body reference. Samples are already reordered to A3 arm topic order:
// left arm seven joints followed by right arm seven joints.
class UpperTrajectory {
 public:
  static constexpr std::size_t kDof = 14;

  bool Load(const std::string& path, std::string& error);
  bool Sample(double time_s, std::array<double, kDof>& q,
              std::array<double, kDof>& qd) const noexcept;

  double DurationS() const noexcept { return duration_s_; }
  bool Empty() const noexcept { return q_.empty(); }

 private:
  double dt_s_{0.0};
  double duration_s_{0.0};
  std::vector<double> q_;
  std::vector<double> qd_;
};

}  // namespace a3_deploy::control
