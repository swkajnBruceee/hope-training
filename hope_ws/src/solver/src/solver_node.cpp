#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>

#include "msgs/msg/hit_state.hpp"
#include "msgs/msg/predicted_strike.hpp"
#include "msgs/msg/racket_command.hpp"
#include "msgs/msg/target_decision.hpp"

#include "constants.h"
#include "ball_trajectory_predictor.h"
#include "hit_plan.h"
#include "hit_plan_solver.h"

namespace solver {

class HOPESolverNode : public rclcpp::Node {
 public:
  HOPESolverNode() : rclcpp::Node("solver") {
    this->declare_parameter("pre_aim_strike_topic", std::string("/ball/predicted_strike"));
    this->declare_parameter("strike_adjust_topic", std::string("/ball/post_bounce_predicted_strike"));
    this->declare_parameter("predicted_strike_topic", std::string(""));
    this->declare_parameter("x_hit", 0.0);
    this->declare_parameter("target_land_x", 2.055);
    this->declare_parameter("target_land_y", -0.7625);
    this->declare_parameter("target_land_z", 0.0);
    this->declare_parameter("delta_t_flight", 0.5);
    this->declare_parameter("net_clearance_margin", 0.03);
    this->declare_parameter("max_racket_speed", 6.0);
    this->declare_parameter("max_ball_out_speed", -1.0);
    this->declare_parameter("desired_ball_speed", -1.0);
    this->declare_parameter("decision_topic", std::string("/target_decision"));
    this->declare_parameter("hit_state_topic", std::string("/hit/state"));
    this->declare_parameter("drag_k", 0.09375);
    this->declare_parameter("restitution_h", 0.649);
    this->declare_parameter("restitution_v", 0.906);
    this->declare_parameter("restitution_racket", 0.842);

    common::PlannerConfig config;
    config.x_hit = this->get_parameter("x_hit").as_double();
    config.target_land = Eigen::Vector3d(
      this->get_parameter("target_land_x").as_double(),
      this->get_parameter("target_land_y").as_double(),
      this->get_parameter("target_land_z").as_double());
    config.delta_t_flight = this->get_parameter("delta_t_flight").as_double();
    config.C_r = this->get_parameter("restitution_racket").as_double();
    latest_target_ = makeDefaultSolveTarget(config);
    latest_target_.desired_ball_speed = this->get_parameter("desired_ball_speed").as_double();
    latest_target_.max_ball_out_speed = this->get_parameter("max_ball_out_speed").as_double();
    latest_target_.max_racket_speed = this->get_parameter("max_racket_speed").as_double();
    latest_target_.net_clearance_margin = this->get_parameter("net_clearance_margin").as_double();
    latest_target_.valid = true;
    latest_target_.mode = "default_fixed_center";

    common::BallPhysics physics;
    physics.k = this->get_parameter("drag_k").as_double();
    physics.C_h = this->get_parameter("restitution_h").as_double();
    physics.C_v = this->get_parameter("restitution_v").as_double();

    solver_ = std::make_unique<HitPlanSolver>(physics, config, common::TableParams{});
    pre_aim_strike_topic_ = this->get_parameter("pre_aim_strike_topic").as_string();
    strike_adjust_topic_ = this->get_parameter("strike_adjust_topic").as_string();
    const auto legacy_predicted_strike_topic = this->get_parameter("predicted_strike_topic").as_string();
    if (!legacy_predicted_strike_topic.empty()) {
      pre_aim_strike_topic_ = legacy_predicted_strike_topic;
    }
    decision_topic_ = this->get_parameter("decision_topic").as_string();
    hit_state_topic_ = this->get_parameter("hit_state_topic").as_string();

    auto command_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();

    pre_aim_sub_ = this->create_subscription<msgs::msg::PredictedStrike>(
      pre_aim_strike_topic_, command_qos,
      [this](const msgs::msg::PredictedStrike::SharedPtr msg) {
        predictedStrikeCb(msg, StrikeSource::PreAim);
      });
    strike_adjust_sub_ = this->create_subscription<msgs::msg::PredictedStrike>(
      strike_adjust_topic_, command_qos,
      [this](const msgs::msg::PredictedStrike::SharedPtr msg) {
        predictedStrikeCb(msg, StrikeSource::StrikeAdjust);
      });
    decision_sub_ = this->create_subscription<msgs::msg::TargetDecision>(
      decision_topic_, command_qos,
      std::bind(&HOPESolverNode::targetDecisionCb, this, std::placeholders::_1));

    cmd_pub_ = this->create_publisher<msgs::msg::RacketCommand>(
      "/racket/command", command_qos);
    hit_state_pub_ = this->create_publisher<msgs::msg::HitState>(
      hit_state_topic_, command_qos);

    diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/planner/diagnostics", 1);
    diag_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&HOPESolverNode::publishDiagnostics, this));

    RCLCPP_INFO(
      this->get_logger(),
      "HOPE solver started - pre_aim_topic=%s, strike_adjust_topic=%s, decision_topic=%s, hit_state_topic=%s, "
      "x_hit=%.2f, target=(%.3f, %.4f, %.3f), dt=%.3f",
      pre_aim_strike_topic_.c_str(), strike_adjust_topic_.c_str(), decision_topic_.c_str(),
      hit_state_topic_.c_str(), config.x_hit,
      latest_target_.target_land.x(), latest_target_.target_land.y(), latest_target_.target_land.z(),
      latest_target_.delta_t_flight);
  }

 private:
  enum class StrikeSource {
    PreAim,
    StrikeAdjust,
  };

  static const char * strikeSourceName(StrikeSource source) {
    switch (source) {
      case StrikeSource::PreAim: return "pre_aim";
      case StrikeSource::StrikeAdjust: return "strike_adjust";
    }
    return "?";
  }

  void predictedStrikeCb(const msgs::msg::PredictedStrike::SharedPtr msg, StrikeSource source) {
    n_received_++;
    latest_strike_msg_ = *msg;
    if (source == StrikeSource::PreAim) {
      pre_aim_received_ = true;
      latest_pre_aim_msg_ = *msg;
    } else {
      strike_adjust_received_ = true;
      latest_strike_adjust_msg_ = *msg;
      if (msg->valid) {
        strike_adjust_active_ = true;
      }
    }

    if (source == StrikeSource::PreAim && strike_adjust_active_) {
      return;
    }
    if (source == StrikeSource::StrikeAdjust && !msg->valid && !strike_adjust_active_) {
      last_valid_ = false;
      last_tts_ = std::isfinite(msg->time_to_strike) ? msg->time_to_strike : std::nan("");
      last_plan_reason_ = msg->reason.empty() ? "invalid_strike_input" : msg->reason;
      last_phase_ = strikeSourceName(source);
      publishHitState(msg->header, Eigen::Vector3d::Zero(), HitPlan{}, std::nullopt, msg->state, last_plan_reason_, 0);
      return;
    }

    if (!msg->valid) {
      if (source == StrikeSource::StrikeAdjust) {
        strike_adjust_active_ = false;
        if (latest_pre_aim_msg_.valid) {
          processStrikeMessage(latest_pre_aim_msg_, StrikeSource::PreAim);
          return;
        }
      }
      last_valid_ = false;
      last_tts_ = std::isfinite(msg->time_to_strike) ? msg->time_to_strike : std::nan("");
      last_plan_reason_ = msg->reason.empty() ? "invalid_strike_input" : msg->reason;
      last_phase_ = strikeSourceName(source);
      publishHitState(msg->header, Eigen::Vector3d::Zero(), HitPlan{}, std::nullopt, msg->state, last_plan_reason_, 0);
      return;
    }

    processStrikeMessage(*msg, source);
  }

  void processStrikeMessage(const msgs::msg::PredictedStrike & msg, StrikeSource source) {
    trajectory::StrikeTarget strike;
    strike.p_ball = Eigen::Vector3d(
      msg.strike_position.x, msg.strike_position.y, msg.strike_position.z);
    strike.v_ball = Eigen::Vector3d(
      msg.strike_velocity.x, msg.strike_velocity.y, msg.strike_velocity.z);
    strike.t_strike = msg.strike_time;
    strike.num_bounces = msg.predicted_bounces;
    strike.valid = msg.valid;

    const auto plan = solver_->solve(strike, latest_target_);
    last_valid_ = plan.valid;
    last_plan_reason_ = plan.reason;
    last_phase_ = strikeSourceName(source);
    last_tts_ = std::isfinite(msg.time_to_strike) ? msg.time_to_strike : (plan.t_hit - strike.t_strike);
    if (plan.valid) {
      n_valid_++;
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "%s hit plan: hit=(%.3f,%.3f,%.3f) vin=(%.3f,%.3f,%.3f) tts=%.3f",
        strikeSourceName(source),
        plan.p_hit.x(), plan.p_hit.y(), plan.p_hit.z(),
        plan.v_in.x(), plan.v_in.y(), plan.v_in.z(),
        msg.time_to_strike);
    }

    publishRacketCommand(msg.header, plan, msg.time_to_strike, strike.num_bounces);
    publishHitState(
      msg.header,
      strike.p_ball,
      plan,
      msg.time_to_strike,
      plan.valid ? (std::string("ready:") + strikeSourceName(source)) : (std::string("invalid:") + strikeSourceName(source)),
      msg.reason.empty() ? plan.reason : (msg.reason + "|" + plan.reason),
      strike.num_bounces);
  }

  void targetDecisionCb(const msgs::msg::TargetDecision::SharedPtr msg) {
    if (!msg->valid) {
      last_target_reason_ = "invalid_decision_ignored";
      return;
    }

    latest_target_.target_land = Eigen::Vector3d(
      msg->target_land.x,
      msg->target_land.y,
      msg->target_land.z);
    latest_target_.delta_t_flight = msg->delta_t_flight;
    latest_target_.desired_ball_speed = msg->desired_ball_speed;
    latest_target_.max_ball_out_speed = msg->max_ball_out_speed;
    latest_target_.max_racket_speed = msg->max_racket_speed;
    latest_target_.net_clearance_margin = msg->net_clearance_margin;
    latest_target_.valid = msg->valid;
    latest_target_.mode = msg->mode;
    last_target_reason_ = "decision_update";
  }

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

  void publishRacketCommand(
    const std_msgs::msg::Header & header,
    const HitPlan & plan,
    double time_to_strike,
    int predicted_bounces)
  {
    msgs::msg::RacketCommand out;
    out.header = header;
    out.header.frame_id = "world";
    fillPoint(out.position, plan.p_hit);
    fillVector(out.velocity, plan.racket_velocity);
    fillVector(out.ball_velocity_incoming, plan.v_in);
    fillVector(out.normal, plan.racket_normal);
    out.strike_time = plan.t_hit;
    out.time_to_strike = time_to_strike;
    fillVector(out.ball_velocity_outgoing, plan.v_out);
    out.valid = plan.valid;
    out.clears_net = plan.clears_net;
    out.bypasses_net_posts = plan.bypasses_net_posts;
    out.predicted_bounces = predicted_bounces;
    cmd_pub_->publish(out);
  }

  void publishHitState(
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d & current_ball_position,
    const HitPlan & plan,
    const std::optional<double> & time_to_strike,
    const std::string & state,
    const std::string & reason,
    int predicted_bounces)
  {
    msgs::msg::HitState out;
    out.header = header;
    out.header.frame_id = "world";
    out.state = state;
    out.reason = reason;
    fillPoint(out.current_ball_position, current_ball_position);
    fillPoint(out.hit_position, plan.p_hit);
    fillVector(out.ball_velocity_incoming, plan.v_in);
    fillVector(out.racket_velocity, plan.racket_velocity);
    fillVector(out.racket_normal, plan.racket_normal);
    out.racket_orientation = plan.racket_orientation;
    out.strike_time = plan.t_hit;
    out.time_to_strike = time_to_strike.has_value() ? *time_to_strike : std::nan("");
    out.latest_arrival_time = plan.t_hit;
    fillPoint(out.target_land, plan.target_land);
    out.return_flight_time = plan.flight_time;
    fillVector(out.ball_velocity_outgoing, plan.v_out);
    out.valid = plan.valid;
    out.clears_net = plan.clears_net;
    out.bypasses_net_posts = plan.bypasses_net_posts;
    out.predicted_bounces = predicted_bounces;
    hit_state_pub_->publish(out);
  }

  void publishDiagnostics() {
    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = this->get_clock()->now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "solver";
    status.hardware_id = "solver";
    if (n_received_ == 0) {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      status.message = "no strike prediction received yet";
    } else if (last_valid_) {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      status.message = "valid racket command";
    } else {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      status.message = "running; no valid strike";
    }
    diagnostic_msgs::msg::KeyValue kv1;
    kv1.key = "strike_samples_received";
    kv1.value = std::to_string(n_received_);
    diagnostic_msgs::msg::KeyValue kv2;
    kv2.key = "valid_commands";
    kv2.value = std::to_string(n_valid_);
    diagnostic_msgs::msg::KeyValue kv3;
    kv3.key = "last_valid";
    kv3.value = last_valid_ ? "true" : "false";
    diagnostic_msgs::msg::KeyValue kv4;
    kv4.key = "time_to_strike_s";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4f", last_tts_);
    kv4.value = buf;
    diagnostic_msgs::msg::KeyValue kv5;
    kv5.key = "target_mode";
    kv5.value = latest_target_.mode;
    diagnostic_msgs::msg::KeyValue kv6;
    kv6.key = "plan_reason";
    kv6.value = last_plan_reason_;
    diagnostic_msgs::msg::KeyValue kv7;
    kv7.key = "target_reason";
    kv7.value = last_target_reason_;
    diagnostic_msgs::msg::KeyValue kv8;
    kv8.key = "active_phase";
    kv8.value = last_phase_;
    status.values = {kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8};
    arr.status = {status};
    diag_pub_->publish(arr);
  }

  std::unique_ptr<HitPlanSolver> solver_;
  SolveTarget latest_target_;
  msgs::msg::PredictedStrike latest_strike_msg_;
  msgs::msg::PredictedStrike latest_pre_aim_msg_;
  msgs::msg::PredictedStrike latest_strike_adjust_msg_;
  std::string pre_aim_strike_topic_;
  std::string strike_adjust_topic_;
  std::string decision_topic_;
  std::string hit_state_topic_;
  rclcpp::Subscription<msgs::msg::PredictedStrike>::SharedPtr pre_aim_sub_;
  rclcpp::Subscription<msgs::msg::PredictedStrike>::SharedPtr strike_adjust_sub_;
  rclcpp::Subscription<msgs::msg::TargetDecision>::SharedPtr decision_sub_;
  rclcpp::Publisher<msgs::msg::RacketCommand>::SharedPtr cmd_pub_;
  rclcpp::Publisher<msgs::msg::HitState>::SharedPtr hit_state_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;
  int n_received_ = 0;
  int n_valid_ = 0;
  bool last_valid_ = false;
  bool pre_aim_received_ = false;
  bool strike_adjust_received_ = false;
  bool strike_adjust_active_ = false;
  double last_tts_ = std::nan("");
  std::string last_plan_reason_ = "none";
  std::string last_target_reason_ = "default_fixed_center";
  std::string last_phase_ = "none";
};

}  // namespace solver

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<solver::HOPESolverNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
