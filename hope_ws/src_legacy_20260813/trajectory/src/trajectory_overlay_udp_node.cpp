#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "ball_state_estimator.h"
#include "ball_trajectory_predictor.h"
#include "constants.h"
#include "incoming_fitter.h"

namespace trajectory {
namespace {

constexpr std::array<char, 4> kMagic = {'H', 'T', 'R', 'J'};

template <typename T>
void appendBytes(std::vector<std::uint8_t> & out, const T & value) {
  const auto * ptr = reinterpret_cast<const std::uint8_t *>(&value);
  out.insert(out.end(), ptr, ptr + sizeof(T));
}

}  // namespace

/**
 * High-level phase of the incoming ball.
 *
 *  WaitingForApex      -- track the highest z sample seen so far; no red
 *                         trajectory is sent.
 *  CollectingAfterApex -- current z has dropped below that highest sample;
 *                         lock one physical forward prediction ending at the
 *                         predicted first P1 contact.
 *  PostP1Bounce        -- a real P1 bounce was observed; incoming_samples_
 *                         is cleared and we fall back to the existing
 *                         BallStateEstimator-driven post-bounce (green)
 *                         pipeline.
 */
enum class IncomingFitState {
  WaitingForApex,
  CollectingAfterApex,
  PostP1Bounce,
};

inline const char * incomingStateName(IncomingFitState s) {
  switch (s) {
    case IncomingFitState::WaitingForApex:     return "WaitingForApex";
    case IncomingFitState::CollectingAfterApex: return "CollectingAfterApex";
    case IncomingFitState::PostP1Bounce:        return "PostP1Bounce";
  }
  return "?";
}

class TrajectoryOverlayUdpNode : public rclcpp::Node {
 public:
  TrajectoryOverlayUdpNode()
  : rclcpp::Node("trajectory_overlay_udp"),
    estimator_(config_),
    predictor_(physics_, config_, table_),
    incoming_fitter_(physics_, config_, table_)
  {
    declare_parameter("ball_topic", std::string("/ball/point"));
    declare_parameter("udp_host", std::string("127.0.0.1"));
    declare_parameter("udp_port", 19532);
    declare_parameter("horizon_s", 1.2);
    declare_parameter("draw_period_s", 0.03);
    declare_parameter("sample_stride", 8);

    ball_topic_ = get_parameter("ball_topic").as_string();
    udp_host_ = get_parameter("udp_host").as_string();
    udp_port_ = get_parameter("udp_port").as_int();
    horizon_s_ = get_parameter("horizon_s").as_double();
    draw_period_s_ = get_parameter("draw_period_s").as_double();
    sample_stride_ = std::max(1, static_cast<int>(get_parameter("sample_stride").as_int()));

    declare_parameter("physics.drag_coefficient", physics_.k);
    declare_parameter("physics.table_tangential_retention", physics_.C_h);
    declare_parameter("physics.table_normal_restitution", physics_.C_v);
    declare_parameter("physics.gravity_x", physics_.g.x());
    declare_parameter("physics.gravity_y", physics_.g.y());
    declare_parameter("physics.gravity_z", physics_.g.z());
    declare_parameter("physics.ball_radius", physics_.radius);
    declare_parameter("physics.ball_mass", physics_.mass);
    declare_parameter("table.length", table_.length);
    declare_parameter("table.width", table_.width);
    declare_parameter("table.net_x", table_.net_x);
    declare_parameter("table.net_height", table_.net_height);
    declare_parameter("table.net_overhang", table_.net_overhang);
    declare_parameter("config.dt_integrate", config_.dt_integrate);
    declare_parameter("config.max_predict_time", config_.max_predict_time);
    loadPhysicalModelParams();

    // Stability / fit thresholds (overridable via ROS params).
    declare_parameter("apex_vx_max", -0.2);      // require signed vx toward P1.
    declare_parameter("apex_min_z", 0.15);
    declare_parameter("apex_max_x", 2.5);        // apex must be on P2 side / over net
    declare_parameter("fit_min_samples", 3);     // >=3 to start emitting red.
    declare_parameter("fit_rms_max", 0.08);
    declare_parameter("contact_jump_warn", 0.25);
    declare_parameter("contact_jump_skip", 0.08);
    declare_parameter("bad_fit_streak_to_reset", 3);
    declare_parameter("pre_apex_history_max", 60);

    apex_vx_max_           = get_parameter("apex_vx_max").as_double();
    apex_min_z_            = get_parameter("apex_min_z").as_double();
    apex_max_x_            = get_parameter("apex_max_x").as_double();
    fit_min_samples_       = get_parameter("fit_min_samples").as_int();
    fit_rms_max_           = get_parameter("fit_rms_max").as_double();
    contact_jump_warn_     = get_parameter("contact_jump_warn").as_double();
    contact_jump_skip_     = get_parameter("contact_jump_skip").as_double();
    bad_fit_streak_to_reset_ = std::max(1, static_cast<int>(get_parameter("bad_fit_streak_to_reset").as_int()));
    pre_apex_history_max_ = std::max(8, static_cast<int>(get_parameter("pre_apex_history_max").as_int()));

    estimator_ = BallStateEstimator(config_);
    predictor_ = BallTrajectoryPredictor(physics_, config_, table_);
    incoming_fitter_ = IncomingFitter(physics_, config_, table_);

    openSocket();

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    ball_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      ball_topic_, qos,
      std::bind(&TrajectoryOverlayUdpNode::ballCb, this, std::placeholders::_1));

    incoming_state_ = IncomingFitState::WaitingForApex;

    RCLCPP_INFO(
      get_logger(),
      "trajectory overlay UDP started - ball_topic=%s, udp://%s:%d",
      ball_topic_.c_str(), udp_host_.c_str(), udp_port_);
  }

  ~TrajectoryOverlayUdpNode() override {
    if (socket_fd_ >= 0) {
      close(socket_fd_);
    }
  }

 private:
  void loadPhysicalModelParams() {
    physics_.k = get_parameter("physics.drag_coefficient").as_double();
    physics_.C_h = get_parameter("physics.table_tangential_retention").as_double();
    physics_.C_v = get_parameter("physics.table_normal_restitution").as_double();
    physics_.g = Eigen::Vector3d(
      get_parameter("physics.gravity_x").as_double(),
      get_parameter("physics.gravity_y").as_double(),
      get_parameter("physics.gravity_z").as_double());
    physics_.radius = get_parameter("physics.ball_radius").as_double();
    physics_.mass = get_parameter("physics.ball_mass").as_double();
    table_.length = get_parameter("table.length").as_double();
    table_.width = get_parameter("table.width").as_double();
    table_.net_x = get_parameter("table.net_x").as_double();
    table_.net_height = get_parameter("table.net_height").as_double();
    table_.net_overhang = get_parameter("table.net_overhang").as_double();
    config_.dt_integrate = get_parameter("config.dt_integrate").as_double();
    config_.max_predict_time = get_parameter("config.max_predict_time").as_double();

    RCLCPP_INFO(
      get_logger(),
      "trajectory physics params: k=%.5f g=(%.3f,%.3f,%.3f) radius=%.3f mass=%.4f "
      "C_h=%.3f C_v=%.3f table=(length=%.3f width=%.3f net_x=%.3f) dt=%.4f",
      physics_.k, physics_.g.x(), physics_.g.y(), physics_.g.z(), physics_.radius, physics_.mass,
      physics_.C_h, physics_.C_v, table_.length, table_.width, table_.net_x, config_.dt_integrate);
  }

  void openSocket() {
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
      throw std::runtime_error("failed to create UDP socket");
    }

    std::memset(&target_addr_, 0, sizeof(target_addr_));
    target_addr_.sin_family = AF_INET;
    target_addr_.sin_port = htons(static_cast<uint16_t>(udp_port_));
    if (inet_pton(AF_INET, udp_host_.c_str(), &target_addr_.sin_addr) != 1) {
      throw std::runtime_error("invalid udp_host: " + udp_host_);
    }
  }

  void ballCb(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
    const Eigen::Vector3d p(msg->point.x, msg->point.y, msg->point.z);
    const double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    // ---- 1) Reject obviously bad samples (out of HOPE scene, NaN, huge jumps) ----
    if (!isBallInsideScene(p)) {
      resetStateMachine();
      ++rejected_samples_;
      last_skip_reason_ = "ball_out_of_scene";
      logStateOncePerSecond();
      return;
    }
    if (last_p_valid_) {
      const double jump = (p - last_p_).norm();
      // 1 m in 23 ms (= 1 / 43 Hz) means ~43 m/s -- way above any reasonable
      // ball speed.  Treat as a teleport / wrap-around / sensor glitch.
      if (jump > 1.0) {
        resetStateMachine();
        ++rejected_samples_;
        last_skip_reason_ = "huge_per_step_jump";
        logStateOncePerSecond();
        return;
      }
    }

    // ---- 2) Push into bounce history; check for a real P1 bounce. ----
    pushBounceHistory(p);
    const bool p1_bounced_now = detectP1Bounce();
    if (p1_bounced_now) {
      // We saw a real P1 contact: switch to post-bounce (green) and forget
      // everything we accumulated while the ball was incoming.
      incoming_state_ = IncomingFitState::PostP1Bounce;
      active_after_p1_bounce_ = true;
      incoming_samples_.clear();
      incoming_samples_valid_ = false;
      pre_apex_history_.clear();
      highest_sample_valid_ = false;
      incoming_prediction_locked_ = false;
      locked_incoming_points_.clear();
      estimator_.reset();
      last_send_t_ = -1.0;
      last_first_point_valid_ = false;
      contact_history_valid_ = false;
      bad_fit_streak_ = 0;
      last_p_ = p;
      last_p_valid_ = true;
      last_recv_t_ = t;
      ++accepted_samples_;
      last_skip_reason_ = "p1_bounce_detected_waiting_for_samples";
      logStateOncePerSecond();
      // Do NOT send a packet for the same sample that flipped the state.
      // We need >= 6 post-bounce samples before resuming output.
      return;
    }

    // ---- 3) Highest-point tracking (state machine pre-P1-bounce). ----
    // Keep the highest sample seen so far.  This works even if this node starts
    // after the real apex and the first valid sample is already descending.
    if (incoming_state_ == IncomingFitState::WaitingForApex) {
      pre_apex_history_.push_back({t, p});
      if (static_cast<int>(pre_apex_history_.size()) > pre_apex_history_max_) {
        pre_apex_history_.erase(pre_apex_history_.begin());
      }

      if (!highest_sample_valid_ || p.z() >= highest_sample_.p.z()) {
        highest_sample_ = {t, p};
        highest_sample_valid_ = true;
        last_p_ = p;
        last_p_valid_ = true;
        last_recv_t_ = t;
        ++accepted_samples_;
        last_skip_reason_ = "tracking_highest_point";
        logStateOncePerSecond();
        return;
      }

      if (shouldStartIncomingFitFromHighest(p, t)) {
        // Flush the rolling history starting at the highest sample, then keep
        // collecting future samples.  The first fit may still wait for
        // fit_min_samples_, but it no longer needs to observe the rise.
        incoming_samples_.clear();
        incoming_samples_.reserve(pre_apex_history_.size());
        bool copy = false;
        for (const auto & s : pre_apex_history_) {
          if (!copy && highest_sample_valid_ &&
              std::abs(s.t - highest_sample_.t) <= 1e-6 &&
              (s.p - highest_sample_.p).norm() <= 1e-6) {
            copy = true;
          }
          if (copy) incoming_samples_.push_back(s);
        }
        if (incoming_samples_.empty() && highest_sample_valid_) {
          incoming_samples_.push_back(highest_sample_);
        }
        if (incoming_samples_.empty() ||
            std::abs(incoming_samples_.back().t - t) > 1e-6) {
          incoming_samples_.push_back({t, p});
        }
        pre_apex_history_.clear();
        incoming_samples_valid_ = true;
        incoming_state_ = IncomingFitState::CollectingAfterApex;
        contact_history_valid_ = false;
        bad_fit_streak_ = 0;
        last_apex_p_ = highest_sample_.p;
        last_apex_t_ = highest_sample_.t;
        last_send_t_ = -1.0;
        last_first_point_valid_ = false;
        last_p_ = p;
        last_p_valid_ = true;
        last_recv_t_ = t;
        ++accepted_samples_;
        last_skip_reason_ = "highest_point_passed";
        logStateOncePerSecond();
        return;
      }

      // We have a higher past sample, but direction / bounds checks are not
      // usable yet.  Keep waiting and retain the highest sample.
      last_p_ = p;
      last_p_valid_ = true;
      last_recv_t_ = t;
      ++accepted_samples_;
      last_skip_reason_ = "waiting_after_highest_point";
      logStateOncePerSecond();
      return;
    }

    // ---- 4) CollectingAfterApex: accumulate, refit, send red. ----
    if (incoming_state_ == IncomingFitState::CollectingAfterApex) {
      incoming_samples_.push_back({t, p});
      last_p_ = p;
      last_p_valid_ = true;
      last_recv_t_ = t;
      ++accepted_samples_;

      if (incoming_prediction_locked_) {
        sendPacket(locked_incoming_points_, /*after_p1_bounce=*/false);
        last_send_t_ = t;
        last_sent_n_ = locked_incoming_points_.size();
        ++sent_packets_;
        last_skip_reason_ = "sent_locked_prediction";
        logStateOncePerSecond();
        return;
      }

      if (static_cast<int>(incoming_samples_.size()) < fit_min_samples_) {
        last_skip_reason_ = "accumulating_samples";
        logStateOncePerSecond();
        return;
      }

      // Compute one physical prediction.  After it is accepted, the red line is
      // locked for this incoming arc so later samples cannot reshape it.
      const auto fit = incoming_fitter_.fitAndPredict(
        incoming_samples_, horizon_s_, sample_stride_);

      if (!fit.ok) {
        ++bad_fit_streak_;
        last_skip_reason_ = "fit_failed:" + fit.reason;
        if (bad_fit_streak_ >= bad_fit_streak_to_reset_) {
          // Consecutive fit failures: give up on this arc and go back to
          // waiting for a fresh apex.  Prevents a runaway bad trajectory
          // from being displayed forever.
          incoming_state_ = IncomingFitState::WaitingForApex;
          incoming_samples_.clear();
          incoming_samples_valid_ = false;
          pre_apex_history_.clear();
          highest_sample_valid_ = false;
          incoming_prediction_locked_ = false;
          locked_incoming_points_.clear();
          contact_history_valid_ = false;
          bad_fit_streak_ = 0;
          last_skip_reason_ = "fit_failed_reset_to_waiting";
        }
        logStateOncePerSecond();
        return;
      }

      // ---- Stability gate on predicted contact ----
      if (fit.contact_predicted) {
        if (contact_history_valid_) {
          const double dcontact = (fit.contact - last_contact_).norm();
          if (dcontact > contact_jump_warn_) {
            // Suspicious jump: skip this frame but don't reset yet.
            ++bad_fit_streak_;
            last_skip_reason_ = "contact_jump_too_large";
            if (bad_fit_streak_ >= bad_fit_streak_to_reset_) {
              incoming_state_ = IncomingFitState::WaitingForApex;
              incoming_samples_.clear();
              incoming_samples_valid_ = false;
              pre_apex_history_.clear();
              highest_sample_valid_ = false;
              incoming_prediction_locked_ = false;
              locked_incoming_points_.clear();
              contact_history_valid_ = false;
              bad_fit_streak_ = 0;
              last_skip_reason_ = "contact_jump_reset";
            }
            logStateOncePerSecond();
            return;
          }
          // Soft update: keep the contact within +/- contact_jump_skip_ of the
          // previous prediction to avoid jitter, but still accept the fit.
          if (dcontact > contact_jump_skip_) {
            // Heavy low-pass: blend with previous contact instead of jumping.
            const double alpha = 0.5;
            const Eigen::Vector3d blended =
              alpha * fit.contact + (1.0 - alpha) * last_contact_;
            last_contact_ = blended;
          } else {
            last_contact_ = fit.contact;
          }
        } else {
          last_contact_ = fit.contact;
          contact_history_valid_ = true;
        }
        bad_fit_streak_ = 0;
      }

      // ---- Build the actual outgoing red packet ----
      const auto full_future_points = predictor_.sampleFuture(
        fit.p_ref, fit.v_ref, horizon_s_, sample_stride_);

      std::vector<Eigen::Vector3d> points;
      points.reserve(fit.observed_points.size() + full_future_points.size());
      // Keep the observed prefix exactly as measured (highest -> latest),
      // then continue with the full physically-predicted arc, including the
      // post-bounce segment. Once a real P1 bounce is observed, the state
      // machine still switches to the green post-bounce trajectory.
      for (const auto & q : fit.observed_points) points.push_back(q);
      if (!full_future_points.empty() && !points.empty()) {
        std::size_t start = 0;
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < full_future_points.size(); ++i) {
          const double d = (full_future_points[i] - points.back()).norm();
          if (d < best) {
            best = d;
            start = i;
          }
        }
        if ((full_future_points[start] - points.back()).norm() < 1e-6) {
          start += 1;
        }
        for (std::size_t i = start; i < full_future_points.size(); ++i) {
          points.push_back(full_future_points[i]);
        }
      }

      // Hard filter on the trajectory points: NaN/Inf / bounds / step length.
      points.erase(
        std::remove_if(
          points.begin(), points.end(),
          [](const Eigen::Vector3d & q) {
            if (!std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z())) {
              return true;
            }
            if (q.z() < -0.05 || q.z() > 1.5) return true;
            if (q.x() < -0.7 || q.x() > 3.5) return true;
            if (q.y() < -1.7 || q.y() > 0.2) return true;
            return false;
          }),
        points.end());
      std::vector<Eigen::Vector3d> filtered;
      filtered.reserve(points.size());
      for (std::size_t i = 0; i < points.size(); ++i) {
        if (!filtered.empty()) {
          const double seg = (points[i] - filtered.back()).norm();
          if (seg > 0.5) break;
        }
        filtered.push_back(points[i]);
      }
      points = std::move(filtered);

      if (points.size() < 4) {
        ++skipped_packets_;
        last_skip_reason_ = "too_few_points";
        logStateOncePerSecond();
        return;
      }

      // The red polyline starts at the apex, then passes through the latest
      // observation before continuing to the predicted contact.  Validate that
      // the trajectory still contains the latest observation; do not require
      // the first point (the apex) to stay near the ball.
      double nearest_to_latest = std::numeric_limits<double>::infinity();
      for (const auto & q : points) {
        nearest_to_latest = std::min(nearest_to_latest, (q - last_p_).norm());
      }
      if (nearest_to_latest > 0.35) {
        ++skipped_packets_;
        last_skip_reason_ = "trajectory_far_from_latest_pre";
        logStateOncePerSecond();
        return;
      }

      if (last_first_point_valid_) {
        const double first_jump = (points.front() - last_first_point_).norm();
        if (first_jump > 0.5) {
          ++skipped_packets_;
          last_skip_reason_ = "first_point_teleport";
          logStateOncePerSecond();
          return;
        }
      }

      // RMS is diagnostic only for the incoming red overlay.  The visible
      // trajectory is generated once from the early measured state and the
      // configured physical model; it must not be repeatedly reshaped by
      // later samples.

      locked_incoming_points_ = points;
      incoming_prediction_locked_ = true;

      sendPacket(points, /*after_p1_bounce=*/false);
      last_send_t_ = t;
      last_first_point_ = points.front();
      last_first_point_valid_ = true;
      last_sent_n_ = points.size();
      ++sent_packets_;

      // Cache fit diagnostics for logging.
      last_fit_ok_ = true;
      last_fit_rms_ = fit.rms_error;
      last_fit_contact_ = fit.contact;
      last_fit_contact_predicted_ = fit.contact_predicted;
      last_fit_v_ref_ = fit.v_ref;
      last_fit_p_ref_ = fit.p_ref;
      last_fit_t_span_ = incoming_samples_.back().t - incoming_samples_.front().t;
      last_fit_num_used_ = fit.num_used;

      last_skip_reason_ = "sent";
      logStateOncePerSecond();
      return;
    }

    // ---- 5) PostP1Bounce: keep the existing green pipeline. ----
    estimator_.push(t, p);
    last_p_ = p;
    last_p_valid_ = true;
    last_recv_t_ = t;
    ++accepted_samples_;

    if (!estimator_.ready()) {
      last_skip_reason_ = "estimator_not_ready";
      logStateOncePerSecond();
      return;
    }
    if (last_send_t_ >= 0.0 && t - last_send_t_ < draw_period_s_) {
      last_skip_reason_ = "throttled_by_draw_period";
      logStateOncePerSecond();
      return;
    }

    auto est = estimator_.estimate();

    const double est_to_latest = (est.p - last_p_).norm();
    const double est_speed = est.v.norm();
    if (est_to_latest > 0.25) {
      estimator_.reset();
      last_skip_reason_ = "p_est_far_from_latest";
      logStateOncePerSecond();
      return;
    }
    if (est_speed > 15.0) {
      estimator_.reset();
      last_skip_reason_ = "v_est_too_fast";
      logStateOncePerSecond();
      return;
    }
    if (!std::isfinite(est.p.x()) || !std::isfinite(est.p.y()) || !std::isfinite(est.p.z()) ||
        !std::isfinite(est.v.x()) || !std::isfinite(est.v.y()) || !std::isfinite(est.v.z())) {
      estimator_.reset();
      last_skip_reason_ = "est_nan_or_inf";
      logStateOncePerSecond();
      return;
    }

    last_est_p_ = est.p;
    last_est_v_ = est.v;

    auto points = predictor_.sampleFuture(
      est.p, est.v, horizon_s_, sample_stride_);

    points.erase(
      std::remove_if(
        points.begin(), points.end(),
        [](const Eigen::Vector3d & q) {
          if (!std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z())) {
            return true;
          }
          if (q.z() < -0.05 || q.z() > 1.5) return true;
          if (q.x() < -0.7 || q.x() > 3.5) return true;
          if (q.y() < -1.7 || q.y() > 0.2) return true;
          return false;
        }),
      points.end());
    std::vector<Eigen::Vector3d> filtered;
    filtered.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
      if (!filtered.empty()) {
        const double seg = (points[i] - filtered.back()).norm();
        if (seg > 0.5) break;
      }
      filtered.push_back(points[i]);
    }
    points = std::move(filtered);

    if (points.size() < 2) {
      ++skipped_packets_;
      last_skip_reason_ = "too_few_points";
      logStateOncePerSecond();
      return;
    }

    if (last_first_point_valid_) {
      const double first_jump = (points.front() - last_first_point_).norm();
      if (first_jump > 0.5) {
        ++skipped_packets_;
        last_skip_reason_ = "first_point_teleport";
        logStateOncePerSecond();
        return;
      }
    }

    sendPacket(points, /*after_p1_bounce=*/true);
    last_send_t_ = t;
    last_first_point_ = points.front();
    last_first_point_valid_ = true;
    last_sent_n_ = points.size();
    ++sent_packets_;
    last_skip_reason_ = "sent";
    logStateOncePerSecond();
  }

  void resetStateMachine() {
    incoming_state_ = IncomingFitState::WaitingForApex;
    incoming_samples_.clear();
    incoming_samples_valid_ = false;
    pre_apex_history_.clear();
    highest_sample_valid_ = false;
    incoming_prediction_locked_ = false;
    locked_incoming_points_.clear();
    active_after_p1_bounce_ = false;
    estimator_.reset();
    last_send_t_ = -1.0;
    last_first_point_valid_ = false;
    last_p_valid_ = false;
    t_hist_valid_[0] = false;
    t_hist_valid_[1] = false;
    t_hist_valid_[2] = false;
    contact_history_valid_ = false;
    bad_fit_streak_ = 0;
  }

  bool isBallInsideScene(const Eigen::Vector3d & p) const {
    if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z())) return false;
    if (p.z() < -0.05 || p.z() > 1.5) return false;
    if (p.x() < -0.7 || p.x() > 3.5) return false;
    if (p.y() < -1.7 || p.y() > 0.2) return false;
    return true;
  }

  /**
   * Decide whether the already-recorded highest sample should become the
   * red trajectory start.  This intentionally does not require seeing the
   * rise before the highest point: if the first observed sample is already
   * the highest and subsequent samples descend, red prediction can begin.
   *
   *  - vx < apex_vx_max_     (HOPE: incoming ball -> P1 means vx < 0);
   *  - highest x is not already deep in P1;
   *  - highest z is above apex_min_z_.
   */
  bool shouldStartIncomingFitFromHighest(const Eigen::Vector3d & p, double t) const {
    if (!highest_sample_valid_) return false;
    if (highest_sample_.p.z() <= p.z()) return false;

    double vx_estimate = 0.0;
    bool vx_valid = false;
    if (last_p_valid_ && t > last_recv_t_) {
      vx_estimate = (p.x() - last_p_.x()) / std::max(1e-6, t - last_recv_t_);
      vx_valid = true;
    } else if (p_hist_valid_[0] && p_hist_valid_[2] && t_hist_valid_[0] && t_hist_valid_[2]) {
      vx_estimate = (p_hist_[2].x() - p_hist_[0].x()) /
        std::max(1e-6, t_hist_[2] - t_hist_[0]);
      vx_valid = true;
    }
    if (!vx_valid) return false;
    if (vx_estimate > apex_vx_max_) return false;

    // Highest point must still be over P2 side or near net, not yet on P1.
    if (highest_sample_.p.x() < -physics_.radius) return false;
    if (highest_sample_.p.x() > apex_max_x_) return false;

    // Height sanity (avoid tracking table-skimmers).
    if (highest_sample_.p.z() < apex_min_z_) return false;

    // Sanity: highest point is still well above the table.
    if (highest_sample_.p.z() < 2.0 * physics_.radius) return false;

    return true;
  }

  void logStateOncePerSecond() {
    const double now = this->now().seconds();
    if (now - last_state_log_t_ < 1.0) return;
    last_state_log_t_ = now;

    if (incoming_state_ == IncomingFitState::CollectingAfterApex && last_fit_ok_) {
      const double t_span = last_fit_t_span_;
      const double contact_delta =
        contact_history_valid_ && last_fit_contact_predicted_
          ? (last_fit_contact_ - last_contact_).norm()
          : 0.0;
      RCLCPP_INFO(
        get_logger(),
        "[incoming_fit] state=%s samples=%zu t_span=%.3f rms_error=%.4f "
        "p_ref=(%.3f,%.3f,%.3f) v_ref=(%.3f,%.3f,%.3f) "
        "contact_predicted=%d contact=(%.3f,%.3f,%.3f) contact_delta=%.3f "
        "reason=%s",
        incomingStateName(incoming_state_),
        last_fit_num_used_,
        t_span,
        last_fit_rms_,
        last_fit_p_ref_.x(), last_fit_p_ref_.y(), last_fit_p_ref_.z(),
        last_fit_v_ref_.x(), last_fit_v_ref_.y(), last_fit_v_ref_.z(),
        last_fit_contact_predicted_ ? 1 : 0,
        last_fit_contact_.x(), last_fit_contact_.y(), last_fit_contact_.z(),
        contact_delta,
        last_skip_reason_.c_str());
    } else {
      RCLCPP_INFO(
        get_logger(),
        "[incoming_fit] state=%s samples=%zu reason=%s color=%u "
        "accepted=%zu rejected=%zu sent=%zu skipped=%zu",
        incomingStateName(incoming_state_),
        incoming_samples_.size(),
        last_skip_reason_.c_str(),
        active_after_p1_bounce_ ? 1u : 0u,
        accepted_samples_, rejected_samples_, sent_packets_, skipped_packets_);
    }
  }

  void pushBounceHistory(const Eigen::Vector3d & p) {
    z_hist_[0] = z_hist_[1];
    z_hist_[1] = z_hist_[2];
    z_hist_[2] = p.z();
    t_hist_[0] = t_hist_[1];
    t_hist_[1] = t_hist_[2];
    t_hist_[2] = last_recv_t_;
    t_hist_valid_[0] = t_hist_valid_[1];
    t_hist_valid_[1] = t_hist_valid_[2];
    t_hist_valid_[2] = true;
    p_hist_[0] = p_hist_[1];
    p_hist_[1] = p_hist_[2];
    p_hist_[2] = p;
    p_hist_valid_[0] = p_hist_valid_[1];
    p_hist_valid_[1] = p_hist_valid_[2];
    p_hist_valid_[2] = true;
  }

  bool detectP1Bounce() const {
    // Only the first P1 bounce per ball/episode may flip the state machine.
    if (active_after_p1_bounce_) {
      return false;
    }
    if (!p_hist_valid_[1]) {
      return false;
    }

    const double z_pp = z_hist_[0];
    const double z_p = z_hist_[1];
    const double z_c = z_hist_[2];
    const Eigen::Vector3d & p_prev = p_hist_[1];

    const bool descending_then_rising = z_pp > z_p && z_c > z_p;

    const bool near_table =
      std::abs(z_p - physics_.radius) <= 0.02;

    const bool on_p1_table =
      p_prev.x() >= -physics_.radius &&
      p_prev.x() <= table_.net_x &&
      p_prev.y() >= -table_.width - physics_.radius &&
      p_prev.y() <= physics_.radius;

    return descending_then_rising && near_table && on_p1_table;
  }

  void sendPacket(const std::vector<Eigen::Vector3d> & points, bool after_p1_bounce) {
    const std::uint32_t count =
      static_cast<std::uint32_t>(std::min<std::size_t>(points.size(), max_points_));
    std::vector<std::uint8_t> payload;
    payload.reserve(16 + count * 3 * sizeof(double));
    payload.insert(payload.end(), kMagic.begin(), kMagic.end());
    appendBytes(payload, sequence_++);
    const std::uint32_t color_state = after_p1_bounce ? 1U : 0U;
    appendBytes(payload, color_state);
    appendBytes(payload, count);
    for (std::uint32_t i = 0; i < count; ++i) {
      const double x = points[i].x();
      const double y = points[i].y();
      const double z = points[i].z();
      appendBytes(payload, x);
      appendBytes(payload, y);
      appendBytes(payload, z);
    }

    sendto(
      socket_fd_,
      payload.data(),
      payload.size(),
      0,
      reinterpret_cast<const sockaddr *>(&target_addr_),
      sizeof(target_addr_));
  }

  common::PlannerConfig config_;
  common::BallPhysics physics_;
  common::TableParams table_;
  BallStateEstimator estimator_;
  BallTrajectoryPredictor predictor_;
  IncomingFitter incoming_fitter_;

  std::string ball_topic_;
  std::string udp_host_;
  int udp_port_ = 19532;
  double horizon_s_ = 1.2;
  double draw_period_s_ = 0.03;
  int sample_stride_ = 8;
  int socket_fd_ = -1;
  sockaddr_in target_addr_{};
  std::uint32_t sequence_ = 0;
  static constexpr std::size_t max_points_ = 512;

  std::array<double, 3> z_hist_{1000.0, 1000.0, 1000.0};
  std::array<double, 3> t_hist_{0.0, 0.0, 0.0};
  std::array<bool, 3> t_hist_valid_{false, false, false};
  std::array<Eigen::Vector3d, 3> p_hist_{};
  std::array<bool, 3> p_hist_valid_{false, false, false};
  bool active_after_p1_bounce_ = false;
  double last_send_t_ = -1.0;

  // === state machine / stability bookkeeping ===
  Eigen::Vector3d last_p_{0.0, 0.0, 0.0};
  bool last_p_valid_ = false;
  double last_recv_t_ = -1.0;

  bool last_first_point_valid_ = false;
  Eigen::Vector3d last_first_point_{0.0, 0.0, 0.0};

  // 1-Hz diagnostic counters.
  double last_state_log_t_ = 0.0;
  size_t accepted_samples_ = 0;
  size_t rejected_samples_ = 0;
  size_t sent_packets_ = 0;
  size_t skipped_packets_ = 0;
  std::string last_skip_reason_ = "none";
  size_t last_sent_n_ = 0;
  Eigen::Vector3d last_est_p_{0.0, 0.0, 0.0};
  Eigen::Vector3d last_est_v_{0.0, 0.0, 0.0};

  // === incoming-fit state ===
  IncomingFitState incoming_state_ = IncomingFitState::WaitingForApex;
  std::vector<TimedBallSample> incoming_samples_;
  bool incoming_samples_valid_ = false;
  // Pre-apex rolling history (kept while we wait for the apex) so that, on
  // detection, we already have ~10-30 /ball/point samples ready to flush
  // into incoming_samples_.  This avoids the "wait until the ball is
  // almost on the P1 table" problem of the previous version.
  std::vector<TimedBallSample> pre_apex_history_;
  TimedBallSample highest_sample_;
  bool highest_sample_valid_ = false;
  bool incoming_prediction_locked_ = false;
  std::vector<Eigen::Vector3d> locked_incoming_points_;
  Eigen::Vector3d last_apex_p_{0.0, 0.0, 0.0};
  double last_apex_t_ = 0.0;

  // Stability tracking of the predicted P1 contact.
  Eigen::Vector3d last_contact_{0.0, 0.0, 0.0};
  bool contact_history_valid_ = false;
  int bad_fit_streak_ = 0;

  // Tunables (loaded from ROS params in ctor).
  double apex_vx_max_ = -0.2;
  double apex_min_z_ = 0.15;
  double apex_max_x_ = 2.5;
  int fit_min_samples_ = 6;
  double fit_rms_max_ = 0.08;
  double contact_jump_warn_ = 0.25;
  double contact_jump_skip_ = 0.08;
  int bad_fit_streak_to_reset_ = 3;
  int pre_apex_history_max_ = 60;

  // Cached fit diagnostics for the 1-Hz log.
  bool last_fit_ok_ = false;
  double last_fit_rms_ = 0.0;
  Eigen::Vector3d last_fit_contact_{0.0, 0.0, 0.0};
  bool last_fit_contact_predicted_ = false;
  Eigen::Vector3d last_fit_v_ref_{0.0, 0.0, 0.0};
  Eigen::Vector3d last_fit_p_ref_{0.0, 0.0, 0.0};
  double last_fit_t_span_ = 0.0;
  std::size_t last_fit_num_used_ = 0;

  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr ball_sub_;
};

}  // namespace trajectory

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<trajectory::TrajectoryOverlayUdpNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
