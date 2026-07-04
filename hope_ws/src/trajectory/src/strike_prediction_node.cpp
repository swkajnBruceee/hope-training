#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "ball_state_estimator.h"
#include "ball_trajectory_predictor.h"
#include "constants.h"
#include "incoming_fitter.h"
#include "msgs/msg/predicted_strike.hpp"

namespace trajectory {
namespace {

enum class StrikePhase {
  PreAim,
  StrikeAdjust,
};

enum class PreBounceState {
  WaitingForApex,
  CollectingAfterApex,
  LockedStrike,
};

inline const char * preBounceStateName(PreBounceState state) {
  switch (state) {
    case PreBounceState::WaitingForApex: return "WaitingForApex";
    case PreBounceState::CollectingAfterApex: return "CollectingAfterApex";
    case PreBounceState::LockedStrike: return "LockedStrike";
  }
  return "?";
}

inline const char * phaseName(StrikePhase phase) {
  switch (phase) {
    case StrikePhase::PreAim: return "pre_aim";
    case StrikePhase::StrikeAdjust: return "strike_adjust";
  }
  return "?";
}

}  // namespace

class StrikePredictionNode : public rclcpp::Node {
 public:
  StrikePredictionNode()
  : rclcpp::Node("strike_prediction"),
    predictor_(physics_, config_, table_),
    incoming_fitter_(physics_, config_, table_),
    post_bounce_estimator_(config_)
  {
    declare_parameter("ball_topic", std::string("/ball/point"));
    declare_parameter("pre_aim_strike_topic", std::string("/ball/predicted_strike"));
    declare_parameter("strike_adjust_topic", std::string("/ball/post_bounce_predicted_strike"));
    declare_parameter("strike_topic", std::string(""));
    declare_parameter("apex_vx_max", -0.2);
    declare_parameter("apex_min_z", 0.15);
    declare_parameter("apex_max_x", 2.5);
    declare_parameter("fit_min_samples", 3);
    declare_parameter("fit_rms_max", 0.08);
    declare_parameter("pre_apex_history_max", 60);

    ball_topic_ = get_parameter("ball_topic").as_string();
    pre_aim_strike_topic_ = get_parameter("pre_aim_strike_topic").as_string();
    strike_adjust_topic_ = get_parameter("strike_adjust_topic").as_string();
    const auto legacy_strike_topic = get_parameter("strike_topic").as_string();
    if (!legacy_strike_topic.empty()) {
      pre_aim_strike_topic_ = legacy_strike_topic;
    }
    apex_vx_max_ = get_parameter("apex_vx_max").as_double();
    apex_min_z_ = get_parameter("apex_min_z").as_double();
    apex_max_x_ = get_parameter("apex_max_x").as_double();
    fit_min_samples_ = get_parameter("fit_min_samples").as_int();
    fit_rms_max_ = get_parameter("fit_rms_max").as_double();
    pre_apex_history_max_ = std::max(8, static_cast<int>(get_parameter("pre_apex_history_max").as_int()));

    auto mocap_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    auto out_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();

    ball_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      ball_topic_, mocap_qos,
      std::bind(&StrikePredictionNode::ballCb, this, std::placeholders::_1));
    pre_aim_pub_ = create_publisher<msgs::msg::PredictedStrike>(pre_aim_strike_topic_, out_qos);
    strike_adjust_pub_ = create_publisher<msgs::msg::PredictedStrike>(strike_adjust_topic_, out_qos);

    RCLCPP_INFO(
      get_logger(),
      "strike prediction started - ball_topic=%s, pre_aim_topic=%s, strike_adjust_topic=%s",
      ball_topic_.c_str(), pre_aim_strike_topic_.c_str(), strike_adjust_topic_.c_str());
  }

 private:
  static void fillPoint(geometry_msgs::msg::Point & out, const Eigen::Vector3d & v) {
    out.x = v.x();
    out.y = v.y();
    out.z = v.z();
  }

  static void fillVector(geometry_msgs::msg::Vector3 & out, const Eigen::Vector3d & v) {
    out.x = v.x();
    out.y = v.y();
    out.z = v.z();
  }

  void publishInvalid(
    const std_msgs::msg::Header & header,
    StrikePhase phase,
    const std::string & state,
    const std::string & reason)
  {
    msgs::msg::PredictedStrike out;
    out.header = header;
    out.header.frame_id = "world";
    out.state = std::string(phaseName(phase)) + ":" + state;
    out.reason = reason;
    out.valid = false;
    publisherFor(phase)->publish(out);
  }

  void publishStrike(
    const std_msgs::msg::Header & header,
    StrikePhase phase,
    const std::string & state,
    const std::string & reason,
    const StrikeTarget & strike)
  {
    msgs::msg::PredictedStrike out;
    out.header = header;
    out.header.frame_id = "world";
    out.state = std::string(phaseName(phase)) + ":" + state;
    out.reason = reason;
    fillPoint(out.strike_position, strike.p_ball);
    fillVector(out.strike_velocity, strike.v_ball);
    out.strike_time = strike.t_strike;
    out.time_to_strike = strike.t_strike - last_t_;
    out.predicted_bounces = strike.num_bounces;
    out.valid = strike.valid;
    publisherFor(phase)->publish(out);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "%s strike: p=(%.3f,%.3f,%.3f) v=(%.3f,%.3f,%.3f) tts=%.3f",
      phaseName(phase),
      strike.p_ball.x(), strike.p_ball.y(), strike.p_ball.z(),
      strike.v_ball.x(), strike.v_ball.y(), strike.v_ball.z(),
      out.time_to_strike);
  }

  auto publisherFor(StrikePhase phase) const
    -> const rclcpp::Publisher<msgs::msg::PredictedStrike>::SharedPtr &
  {
    return (phase == StrikePhase::PreAim) ? pre_aim_pub_ : strike_adjust_pub_;
  }

  void resetPreBounceState() {
    pre_bounce_state_ = PreBounceState::WaitingForApex;
    pre_apex_history_.clear();
    incoming_samples_.clear();
    highest_sample_valid_ = false;
    locked_strike_ = trajectory::StrikeTarget{};
  }

  void resetPostBounceState() {
    active_after_p1_bounce_ = false;
    post_bounce_estimator_.reset();
    post_bounce_initialized_ = false;
    post_bounce_strike_ = trajectory::StrikeTarget{};
  }

  void resetAllState() {
    resetPreBounceState();
    resetPostBounceState();
    last_p_valid_ = false;
    t_hist_valid_[0] = false;
    t_hist_valid_[1] = false;
    t_hist_valid_[2] = false;
    p_hist_valid_[0] = false;
    p_hist_valid_[1] = false;
    p_hist_valid_[2] = false;
  }

  bool isBallInsideScene(const Eigen::Vector3d & p) const {
    if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z())) return false;
    if (p.z() < -0.05 || p.z() > 1.5) return false;
    if (p.x() < -0.7 || p.x() > 3.5) return false;
    if (p.y() < -1.7 || p.y() > 0.2) return false;
    return true;
  }

  void pushBounceHistory(const Eigen::Vector3d & p) {
    z_hist_[0] = z_hist_[1];
    z_hist_[1] = z_hist_[2];
    z_hist_[2] = p.z();
    t_hist_[0] = t_hist_[1];
    t_hist_[1] = t_hist_[2];
    t_hist_[2] = last_t_;
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
    const bool near_table = std::abs(z_p - physics_.radius) <= 0.02;
    const bool on_p1_table =
      p_prev.x() >= -physics_.radius &&
      p_prev.x() <= table_.net_x &&
      p_prev.y() >= -table_.width - physics_.radius &&
      p_prev.y() <= physics_.radius;
    return descending_then_rising && near_table && on_p1_table;
  }

  void startPostBouncePhase() {
    active_after_p1_bounce_ = true;
    post_bounce_estimator_.reset();
    post_bounce_initialized_ = false;
    post_bounce_strike_ = trajectory::StrikeTarget{};
    resetPreBounceState();
  }

  bool handlePostBounce(const std_msgs::msg::Header & header, double t, const Eigen::Vector3d & p) {
    if (!active_after_p1_bounce_) {
      return false;
    }

    post_bounce_estimator_.push(t, p);
    if (!post_bounce_initialized_) {
      post_bounce_initialized_ = true;
      publishInvalid(header, StrikePhase::StrikeAdjust, "CollectingAfterBounce", "accumulating_post_bounce_samples");
      return true;
    }
    if (!post_bounce_estimator_.ready()) {
      publishInvalid(header, StrikePhase::StrikeAdjust, "CollectingAfterBounce", "accumulating_post_bounce_samples");
      return true;
    }

    const auto est = post_bounce_estimator_.estimate();
    const double est_to_latest = (est.p - p).norm();
    const double est_speed = est.v.norm();
    if (est_to_latest > 0.25) {
      post_bounce_estimator_.reset();
      publishInvalid(header, StrikePhase::StrikeAdjust, "CollectingAfterBounce", "post_bounce_est_far_from_latest");
      return true;
    }
    if (est_speed > 15.0) {
      post_bounce_estimator_.reset();
      publishInvalid(header, StrikePhase::StrikeAdjust, "CollectingAfterBounce", "post_bounce_v_est_too_fast");
      return true;
    }
    if (!std::isfinite(est.p.x()) || !std::isfinite(est.p.y()) || !std::isfinite(est.p.z()) ||
        !std::isfinite(est.v.x()) || !std::isfinite(est.v.y()) || !std::isfinite(est.v.z())) {
      post_bounce_estimator_.reset();
      publishInvalid(header, StrikePhase::StrikeAdjust, "CollectingAfterBounce", "post_bounce_est_nan_or_inf");
      return true;
    }

    post_bounce_strike_ = predictor_.predict(est.p, est.v, est.t);
    if (!post_bounce_strike_.valid) {
      publishInvalid(header, StrikePhase::StrikeAdjust, "CollectingAfterBounce", "invalid_post_bounce_strike");
      return true;
    }

    publishStrike(
      header,
      StrikePhase::StrikeAdjust,
      "PostBounceLockedStrike",
      "post_bounce_locked_strike",
      post_bounce_strike_);
    return true;
  }

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
    if (highest_sample_.p.x() < -physics_.radius) return false;
    if (highest_sample_.p.x() > apex_max_x_) return false;
    if (highest_sample_.p.z() < apex_min_z_) return false;
    if (highest_sample_.p.z() < 2.0 * physics_.radius) return false;
    return true;
  }

  void ballCb(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
    const Eigen::Vector3d p(msg->point.x, msg->point.y, msg->point.z);
    const double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    last_t_ = t;

    if (!isBallInsideScene(p)) {
      resetAllState();
      publishInvalid(msg->header, StrikePhase::PreAim, preBounceStateName(pre_bounce_state_), "ball_out_of_scene");
      publishInvalid(msg->header, StrikePhase::StrikeAdjust, "Idle", "ball_out_of_scene");
      return;
    }

    pushBounceHistory(p);
    if (detectP1Bounce()) {
      startPostBouncePhase();
      publishInvalid(msg->header, StrikePhase::PreAim, "PostBounceDetected", "p1_bounce_detected");
      publishInvalid(msg->header, StrikePhase::StrikeAdjust, "CollectingAfterBounce", "p1_bounce_detected");
      last_p_ = p;
      last_p_valid_ = true;
      last_recv_t_ = t;
      return;
    }

    if (handlePostBounce(msg->header, t, p)) {
      last_p_ = p;
      last_p_valid_ = true;
      last_recv_t_ = t;
      return;
    }

    if (pre_bounce_state_ == PreBounceState::LockedStrike && locked_strike_.valid) {
      publishStrike(
        msg->header,
        StrikePhase::PreAim,
        preBounceStateName(pre_bounce_state_),
        "locked_strike",
        locked_strike_);
      return;
    }

    if (pre_bounce_state_ == PreBounceState::WaitingForApex) {
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
        publishInvalid(msg->header, StrikePhase::PreAim, preBounceStateName(pre_bounce_state_), "tracking_highest_point");
        return;
      }

      if (!shouldStartIncomingFitFromHighest(p, t)) {
        last_p_ = p;
        last_p_valid_ = true;
        last_recv_t_ = t;
        publishInvalid(msg->header, StrikePhase::PreAim, preBounceStateName(pre_bounce_state_), "waiting_after_highest_point");
        return;
      }

      incoming_samples_.clear();
      bool copy = false;
      for (const auto & sample : pre_apex_history_) {
        if (!copy && highest_sample_valid_ &&
            std::abs(sample.t - highest_sample_.t) <= 1e-6 &&
            (sample.p - highest_sample_.p).norm() <= 1e-6) {
          copy = true;
        }
        if (copy) {
          incoming_samples_.push_back(sample);
        }
      }
      if (incoming_samples_.empty()) {
        incoming_samples_.push_back(highest_sample_);
      }
      if (incoming_samples_.empty() || std::abs(incoming_samples_.back().t - t) > 1e-6) {
        incoming_samples_.push_back({t, p});
      }
      pre_apex_history_.clear();
      pre_bounce_state_ = PreBounceState::CollectingAfterApex;
    } else if (pre_bounce_state_ == PreBounceState::CollectingAfterApex) {
      incoming_samples_.push_back({t, p});
    }

    last_p_ = p;
    last_p_valid_ = true;
    last_recv_t_ = t;

    if (static_cast<int>(incoming_samples_.size()) < fit_min_samples_) {
      publishInvalid(msg->header, StrikePhase::PreAim, preBounceStateName(pre_bounce_state_), "accumulating_samples");
      return;
    }

    const auto fit = incoming_fitter_.fitAndPredict(
      incoming_samples_, config_.max_predict_time, /*sample_stride=*/8);
    if (!fit.ok) {
      publishInvalid(msg->header, StrikePhase::PreAim, preBounceStateName(pre_bounce_state_), "fit_failed:" + fit.reason);
      return;
    }
    if (fit.rms_error > fit_rms_max_) {
      publishInvalid(msg->header, StrikePhase::PreAim, preBounceStateName(pre_bounce_state_), "fit_rms_too_large");
      return;
    }

    locked_strike_ = predictor_.predict(fit.p_ref, fit.v_ref, fit.t_ref);
    if (!locked_strike_.valid) {
      publishInvalid(msg->header, StrikePhase::PreAim, preBounceStateName(pre_bounce_state_), "invalid_strike");
      return;
    }

    pre_bounce_state_ = PreBounceState::LockedStrike;
    publishStrike(
      msg->header,
      StrikePhase::PreAim,
      preBounceStateName(pre_bounce_state_),
      "locked_strike",
      locked_strike_);
  }

  std::string ball_topic_;
  std::string pre_aim_strike_topic_;
  std::string strike_adjust_topic_;

  common::BallPhysics physics_;
  common::PlannerConfig config_;
  common::TableParams table_;
  BallTrajectoryPredictor predictor_;
  IncomingFitter incoming_fitter_;
  BallStateEstimator post_bounce_estimator_;

  PreBounceState pre_bounce_state_ = PreBounceState::WaitingForApex;
  std::vector<TimedBallSample> pre_apex_history_;
  std::vector<TimedBallSample> incoming_samples_;
  TimedBallSample highest_sample_;
  bool highest_sample_valid_ = false;
  bool active_after_p1_bounce_ = false;
  bool post_bounce_initialized_ = false;
  Eigen::Vector3d p_hist_[3] = {
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
  };
  double z_hist_[3] = {0.0, 0.0, 0.0};
  double t_hist_[3] = {0.0, 0.0, 0.0};
  bool p_hist_valid_[3] = {false, false, false};
  bool t_hist_valid_[3] = {false, false, false};
  Eigen::Vector3d last_p_ = Eigen::Vector3d::Zero();
  bool last_p_valid_ = false;
  double last_recv_t_ = 0.0;
  double last_t_ = 0.0;
  double apex_vx_max_ = -0.2;
  double apex_min_z_ = 0.15;
  double apex_max_x_ = 2.5;
  int fit_min_samples_ = 3;
  double fit_rms_max_ = 0.08;
  int pre_apex_history_max_ = 60;
  StrikeTarget locked_strike_;
  StrikeTarget post_bounce_strike_;

  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr ball_sub_;
  rclcpp::Publisher<msgs::msg::PredictedStrike>::SharedPtr pre_aim_pub_;
  rclcpp::Publisher<msgs::msg::PredictedStrike>::SharedPtr strike_adjust_pub_;
};

}  // namespace trajectory

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<trajectory::StrikePredictionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
