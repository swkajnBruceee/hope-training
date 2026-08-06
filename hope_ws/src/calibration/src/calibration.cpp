#include "calibration.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace calibration {

namespace {

double median(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n % 2 == 0) {
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }
  return v[n / 2];
}

}  // namespace

common::BallPhysics calibrateBallPhysics(
  const std::vector<std::vector<double>> & timestamps,
  const std::vector<std::vector<Eigen::Vector3d>> & trajectories,
  const Eigen::Vector3d & g)
{
  std::vector<double> drag_a;
  std::vector<double> drag_v2;
  std::vector<double> h_rest;
  std::vector<double> v_rest;

  for (size_t idx = 0; idx < trajectories.size(); ++idx) {
    const auto & pos = trajectories[idx];
    const auto & t = timestamps[idx];
    if (t.size() < 4) {
      continue;
    }
    std::vector<double> dt_arr(t.size() - 1);
    std::vector<Eigen::Vector3d> vel(pos.size() - 1);
    for (size_t i = 0; i + 1 < t.size(); ++i) {
      dt_arr[i] = t[i + 1] - t[i];
      vel[i] = (pos[i + 1] - pos[i]) / dt_arr[i];
    }
    std::vector<double> t_vel(vel.size());
    for (size_t i = 0; i < vel.size(); ++i) {
      t_vel[i] = 0.5 * (t[i] + t[i + 1]);
    }
    std::vector<double> dt_vel(vel.size() - 1);
    for (size_t i = 0; i + 1 < vel.size(); ++i) {
      dt_vel[i] = t_vel[i + 1] - t_vel[i];
    }
    std::vector<Eigen::Vector3d> acc(vel.size() - 1);
    for (size_t i = 0; i + 1 < vel.size(); ++i) {
      acc[i] = (vel[i + 1] - vel[i]) / dt_vel[i];
    }
    std::vector<Eigen::Vector3d> vel_mid(vel.size() - 1);
    for (size_t i = 0; i + 1 < vel.size(); ++i) {
      vel_mid[i] = 0.5 * (vel[i] + vel[i + 1]);
    }
    std::vector<double> z_mid(acc.size());
    for (size_t i = 0; i < acc.size(); ++i) {
      z_mid[i] = 0.5 * (pos[i].z() + pos[i + 2].z());
    }

    for (size_t i = 0; i < acc.size(); ++i) {
      if (z_mid[i] > 0.03) {
        Eigen::Vector3d a_minus_g = acc[i] - g;
        drag_a.push_back(a_minus_g.norm());
        drag_v2.push_back(std::pow(vel_mid[i].norm(), 2));
      }
    }

    for (size_t i = 1; i + 1 < pos.size(); ++i) {
      double z_prev = pos[i - 1].z();
      double z_curr = pos[i].z();
      double z_next = pos[i + 1].z();
      if (z_prev > 0.01 && z_curr < 0.005 && z_next > 0.01) {
        size_t idx_pre = (i >= 2) ? (i - 2) : 0;
        size_t idx_post = std::min(i + 1, vel.size() - 1);
        if (idx_pre < vel.size() && idx_post < vel.size()) {
          Eigen::Vector3d v_pre = vel[idx_pre];
          Eigen::Vector3d v_post = vel[idx_post];
          double v_h_pre = std::hypot(v_pre.x(), v_pre.y());
          double v_h_post = std::hypot(v_post.x(), v_post.y());
          if (v_h_pre > 0.1) {
            h_rest.push_back(v_h_post / v_h_pre);
          }
          if (std::abs(v_pre.z()) > 0.1) {
            v_rest.push_back(std::abs(v_post.z()) / std::abs(v_pre.z()));
          }
        }
      }
    }
  }

  double k;
  if (!drag_a.empty()) {
    double sum_av = 0.0, sum_v2 = 0.0;
    for (size_t i = 0; i < drag_a.size(); ++i) {
      sum_av += drag_a[i] * drag_v2[i];
      sum_v2 += drag_v2[i] * drag_v2[i];
    }
    k = (sum_v2 > 0.0) ? (sum_av / sum_v2) : common::BallPhysics().k;
  } else {
    k = common::BallPhysics().k;
  }

  common::BallPhysics defaults;
  double C_h = h_rest.empty() ? defaults.C_h : median(h_rest);
  double C_v = v_rest.empty() ? defaults.C_v : median(v_rest);

  common::BallPhysics result;
  result.k = k;
  result.C_h = C_h;
  result.C_v = C_v;
  return result;
}

CsvTrajectory loadTrajectoryCsv(const std::string & path) {
  std::ifstream fh(path);
  if (!fh) {
    throw std::runtime_error("Cannot open trajectory CSV: " + path);
  }
  CsvTrajectory out;
  std::string line;
  while (std::getline(fh, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::string tok;
    std::vector<std::string> cols;
    while (std::getline(ss, tok, ',')) {
      cols.push_back(tok);
    }
    if (cols.size() < 4) {
      continue;
    }
    std::string first = cols[0];
    std::transform(first.begin(), first.end(), first.begin(),
      [](unsigned char c) { return std::tolower(c); });
    if (first == "t" || first == "time") {
      continue;
    }
    out.t.push_back(std::stod(cols[0]));
    out.p.emplace_back(std::stod(cols[1]), std::stod(cols[2]), std::stod(cols[3]));
  }
  return out;
}

}  // namespace calibration
