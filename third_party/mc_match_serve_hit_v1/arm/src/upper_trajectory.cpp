#include "upper_trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace a3_deploy::control {
namespace {

constexpr char kMagic[] = "A3UPTRJ1";

template <typename T>
bool Read(std::ifstream& in, T& value) {
  return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

}  // namespace

bool UpperTrajectory::Load(const std::string& path, std::string& error) {
  q_.clear();
  qd_.clear();
  dt_s_ = 0.0;
  duration_s_ = 0.0;

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open upper trajectory: " + path;
    return false;
  }

  // The on-disk header stores the eight visible magic bytes, not the C++
  // string's terminating NUL.
  char magic[sizeof(kMagic) - 1]{};
  std::uint32_t frames = 0;
  std::uint32_t dof = 0;
  if (!in.read(magic, sizeof(magic)) ||
      std::memcmp(magic, kMagic, sizeof(magic)) != 0 ||
      !Read(in, frames) || !Read(in, dof) || !Read(in, dt_s_)) {
    error = "invalid upper trajectory header: " + path;
    return false;
  }
  if (frames < 2 || dof != kDof || !std::isfinite(dt_s_) || dt_s_ <= 0.0) {
    error = "invalid upper trajectory dimensions or rate: " + path;
    return false;
  }

  const std::size_t count = static_cast<std::size_t>(frames) * kDof;
  q_.resize(count);
  qd_.resize(count);
  if (!in.read(reinterpret_cast<char*>(q_.data()),
              static_cast<std::streamsize>(count * sizeof(double))) ||
      !in.read(reinterpret_cast<char*>(qd_.data()),
               static_cast<std::streamsize>(count * sizeof(double)))) {
    q_.clear();
    qd_.clear();
    error = "truncated upper trajectory: " + path;
    return false;
  }
  for (double value : q_) {
    if (!std::isfinite(value)) {
      error = "non-finite upper trajectory position: " + path;
      q_.clear(); qd_.clear();
      return false;
    }
  }
  for (double value : qd_) {
    if (!std::isfinite(value)) {
      error = "non-finite upper trajectory velocity: " + path;
      q_.clear(); qd_.clear();
      return false;
    }
  }
  duration_s_ = static_cast<double>(frames - 1) * dt_s_;
  return true;
}

bool UpperTrajectory::Sample(double time_s, std::array<double, kDof>& q,
                             std::array<double, kDof>& qd) const noexcept {
  if (q_.empty() || qd_.empty() || !std::isfinite(time_s)) return false;
  const std::size_t frames = q_.size() / kDof;
  const double t = std::clamp(time_s, 0.0, duration_s_);
  const double position = t / dt_s_;
  const std::size_t i0 = std::min<std::size_t>(
      static_cast<std::size_t>(std::floor(position)), frames - 2);
  const double alpha = std::clamp(position - static_cast<double>(i0), 0.0, 1.0);
  const std::size_t i1 = i0 + 1;
  for (std::size_t j = 0; j < kDof; ++j) {
    q[j] = (1.0 - alpha) * q_[i0 * kDof + j] + alpha * q_[i1 * kDof + j];
    qd[j] = (1.0 - alpha) * qd_[i0 * kDof + j] + alpha * qd_[i1 * kDof + j];
  }
  return true;
}

}  // namespace a3_deploy::control
