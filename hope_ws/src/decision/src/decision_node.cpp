#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <string>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <rclcpp/rclcpp.hpp>

#include "msgs/msg/landing_candidate_array.hpp"
#include "msgs/msg/predicted_strike.hpp"
#include "msgs/msg/target_decision.hpp"

#include "landing_decision.h"
#include "target_selector.h"

namespace decision {

class DecisionNode : public rclcpp::Node {
 public:
  DecisionNode() : rclcpp::Node("decision") {
    declare_parameter("target_decision_topic", std::string("/target_decision"));
    declare_parameter("landing_candidates_topic", std::string("/planner/landing_candidates"));
    declare_parameter("pre_aim_strike_topic", std::string("/ball/predicted_strike"));
    declare_parameter("strike_adjust_topic", std::string("/ball/post_bounce_predicted_strike"));
    declare_parameter("publish_rate_hz", 10.0);

    declare_parameter("x_hit", 0.0);
    declare_parameter("target_land_x", 2.055);
    declare_parameter("target_land_y", -0.7625);
    declare_parameter("target_land_z", 0.0);
    declare_parameter("delta_t_flight", 0.5);
    declare_parameter("drag_k", 0.09375);
    declare_parameter("restitution_h", 0.649);
    declare_parameter("restitution_v", 0.906);
    declare_parameter("restitution_racket", 0.842);

    declare_parameter("min_time_to_strike", 0.12);
    declare_parameter("min_flight_time", 0.40);
    declare_parameter("max_flight_time", 0.70);
    declare_parameter("net_clearance_margin", 0.03);
    declare_parameter("net_clearance_comfort", 0.05);
    declare_parameter("racket_speed_rule_cap", 6.0);
    declare_parameter("racket_speed_planning_cap", 5.4);
    declare_parameter("ball_speed_comfort_min", 4.5);
    declare_parameter("ball_speed_comfort_max", 8.5);
    declare_parameter("flight_time_comfort_min", 0.48);
    declare_parameter("flight_time_comfort_max", 0.60);
    declare_parameter("competitiveness_level", 0.25);
    declare_parameter("edge_margin_comfort", 0.20);
    declare_parameter("desired_ball_speed", -1.0);
    declare_parameter("max_ball_out_speed", -1.0);

    target_decision_topic_ = get_parameter("target_decision_topic").as_string();
    landing_candidates_topic_ = get_parameter("landing_candidates_topic").as_string();
    pre_aim_strike_topic_ = get_parameter("pre_aim_strike_topic").as_string();
    strike_adjust_topic_ = get_parameter("strike_adjust_topic").as_string();

    common::PlannerConfig planner_config;
    planner_config.x_hit = get_parameter("x_hit").as_double();
    planner_config.target_land = Eigen::Vector3d(
      get_parameter("target_land_x").as_double(),
      get_parameter("target_land_y").as_double(),
      get_parameter("target_land_z").as_double());
    planner_config.delta_t_flight = get_parameter("delta_t_flight").as_double();
    planner_config.C_r = get_parameter("restitution_racket").as_double();

    common::BallPhysics physics;
    physics.k = get_parameter("drag_k").as_double();
    physics.C_h = get_parameter("restitution_h").as_double();
    physics.C_v = get_parameter("restitution_v").as_double();

    LandingDecisionConfig decision_config;
    decision_config.min_time_to_strike = get_parameter("min_time_to_strike").as_double();
    decision_config.min_flight_time = get_parameter("min_flight_time").as_double();
    decision_config.max_flight_time = get_parameter("max_flight_time").as_double();
    decision_config.net_clearance_margin = get_parameter("net_clearance_margin").as_double();
    decision_config.net_clearance_comfort = get_parameter("net_clearance_comfort").as_double();
    decision_config.racket_speed_rule_cap = get_parameter("racket_speed_rule_cap").as_double();
    decision_config.racket_speed_planning_cap = get_parameter("racket_speed_planning_cap").as_double();
    decision_config.ball_speed_comfort_min = get_parameter("ball_speed_comfort_min").as_double();
    decision_config.ball_speed_comfort_max = get_parameter("ball_speed_comfort_max").as_double();
    decision_config.flight_time_comfort_min = get_parameter("flight_time_comfort_min").as_double();
    decision_config.flight_time_comfort_max = get_parameter("flight_time_comfort_max").as_double();
    decision_config.competitiveness_level = get_parameter("competitiveness_level").as_double();
    decision_config.edge_margin_comfort = get_parameter("edge_margin_comfort").as_double();
    decision_config.desired_ball_speed = get_parameter("desired_ball_speed").as_double();
    decision_config.max_ball_out_speed = get_parameter("max_ball_out_speed").as_double();

    dynamic_planner_ = std::make_unique<LandingDecisionPlanner>(
      decision_config, physics, planner_config, common::TableParams{});

    publisher_ = create_publisher<msgs::msg::TargetDecision>(target_decision_topic_, rclcpp::QoS(10));
    candidates_pub_ =
      create_publisher<msgs::msg::LandingCandidateArray>(landing_candidates_topic_, rclcpp::QoS(10));
    diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/planner/decision_diagnostics", 1);

    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    pre_aim_sub_ = create_subscription<msgs::msg::PredictedStrike>(
      pre_aim_strike_topic_, qos,
      [this](const msgs::msg::PredictedStrike::SharedPtr msg) {
        predictedStrikeCb(msg, StrikeSource::PreAim);
      });
    strike_adjust_sub_ = create_subscription<msgs::msg::PredictedStrike>(
      strike_adjust_topic_, qos,
      [this](const msgs::msg::PredictedStrike::SharedPtr msg) {
        predictedStrikeCb(msg, StrikeSource::StrikeAdjust);
      });

    const double publish_rate_hz = get_parameter("publish_rate_hz").as_double();
    const auto period = std::chrono::duration<double>(1.0 / std::max(1e-3, publish_rate_hz));
    fallback_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&DecisionNode::publishFixedFallbackIfIdle, this));
    diag_timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&DecisionNode::publishDiagnostics, this));

    RCLCPP_INFO(
      get_logger(),
      "decision node started - target_topic=%s, pre_aim_topic=%s, strike_adjust_topic=%s",
      target_decision_topic_.c_str(), pre_aim_strike_topic_.c_str(), strike_adjust_topic_.c_str());
  }

 private:
  enum class StrikeSource {
    PreAim,
    StrikeAdjust,
  };

  static const char * strikeSourceName(StrikeSource source) {
    switch (source) {
      case StrikeSource::PreAim:
        return "pre_aim";
      case StrikeSource::StrikeAdjust:
        return "strike_adjust";
    }
    return "?";
  }

  void predictedStrikeCb(const msgs::msg::PredictedStrike::SharedPtr msg, StrikeSource source) {
    if (source == StrikeSource::StrikeAdjust) {
      strike_adjust_active_ = msg->valid;
    }
    if (source == StrikeSource::PreAim && strike_adjust_active_) {
      return;
    }

    if (!msg->valid) {
      last_reason_ = msg->reason.empty() ? "invalid_strike_input" : msg->reason;
      last_mode_ = "waiting_for_valid_strike";
      last_valid_ = false;
      return;
    }
    processed_valid_strike_ = true;

    trajectory::StrikeTarget strike;
    strike.p_ball = Eigen::Vector3d(
      msg->strike_position.x,
      msg->strike_position.y,
      msg->strike_position.z);
    strike.v_ball = Eigen::Vector3d(
      msg->strike_velocity.x,
      msg->strike_velocity.y,
      msg->strike_velocity.z);
    strike.t_strike = msg->strike_time;
    strike.num_bounces = msg->predicted_bounces;
    strike.valid = msg->valid;

    const auto result = dynamic_planner_->select(strike, msg->time_to_strike);
    last_candidate_count_ = result.candidate_count;
    last_hard_valid_count_ = result.hard_valid_count;
    last_score_ = result.selected.total_score;
    last_reject_reasons_ = result.reject_reasons;
    last_mode_ = result.target.mode;
    last_reason_ = result.target.valid ? "decision_update" : "no_feasible_landing";
    last_valid_ = result.target.valid;
    last_phase_ = strikeSourceName(source);

    publishDecision(result.target);
    publishCandidates(result);
  }

  void publishFixedFallbackIfIdle() {
    if (last_valid_ || processed_valid_strike_) {
      return;
    }
    const auto decision = selector_.selectDefault();
    last_mode_ = decision.mode;
    last_reason_ = "fixed_fallback_idle";
    publishDecision(decision);
  }

  void publishDecision(const TargetDecisionData & decision) {
    msgs::msg::TargetDecision msg;
    msg.header.stamp = now();
    msg.header.frame_id = "world";
    msg.target_land.x = decision.target_land.x();
    msg.target_land.y = decision.target_land.y();
    msg.target_land.z = decision.target_land.z();
    msg.delta_t_flight = decision.delta_t_flight;
    msg.desired_ball_speed = decision.desired_ball_speed;
    msg.max_ball_out_speed = decision.max_ball_out_speed;
    msg.max_racket_speed = decision.max_racket_speed;
    msg.net_clearance_margin = decision.net_clearance_margin;
    msg.valid = decision.valid;
    msg.mode = decision.mode;
    publisher_->publish(msg);
  }

  void publishCandidates(const LandingDecisionResult & result) {
    msgs::msg::LandingCandidateArray msg;
    msg.header.stamp = now();
    msg.header.frame_id = "world";
    msg.candidates.reserve(result.candidates.size());
    for (const auto & candidate : result.candidates) {
      msgs::msg::LandingCandidate out;
      out.target_land.x = candidate.target_land.x();
      out.target_land.y = candidate.target_land.y();
      out.target_land.z = candidate.target_land.z();
      out.delta_t_flight = candidate.delta_t_flight;
      out.score = candidate.total_score;
      out.hard_valid = candidate.hard_valid;
      out.selected = candidate.selected;
      out.mode = candidate.mode;
      out.reason = candidate.hard_reason;
      msg.candidates.push_back(out);
    }
    candidates_pub_->publish(msg);
  }

  void publishDiagnostics() {
    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = get_clock()->now();

    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "decision";
    status.hardware_id = "decision";
    status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message = last_valid_ ? "valid target decision" : "fallback or no feasible landing";

    diagnostic_msgs::msg::KeyValue mode;
    mode.key = "target_mode";
    mode.value = last_mode_;
    diagnostic_msgs::msg::KeyValue reason;
    reason.key = "reason";
    reason.value = last_reason_;
    diagnostic_msgs::msg::KeyValue phase;
    phase.key = "active_phase";
    phase.value = last_phase_;
    diagnostic_msgs::msg::KeyValue candidates;
    candidates.key = "candidate_count";
    candidates.value = std::to_string(last_candidate_count_);
    diagnostic_msgs::msg::KeyValue hard_valid;
    hard_valid.key = "hard_valid_count";
    hard_valid.value = std::to_string(last_hard_valid_count_);
    diagnostic_msgs::msg::KeyValue score;
    score.key = "selected_score";
    char score_buf[32];
    std::snprintf(score_buf, sizeof(score_buf), "%.4f", last_score_);
    score.value = score_buf;
    diagnostic_msgs::msg::KeyValue reject_summary;
    reject_summary.key = "reject_reasons";
    reject_summary.value = rejectReasonsSummary();

    status.values = {mode, reason, phase, candidates, hard_valid, score, reject_summary};
    arr.status = {status};
    diag_pub_->publish(arr);
  }

  std::string rejectReasonsSummary() const {
    std::string out;
    for (const auto & item : last_reject_reasons_) {
      if (!out.empty()) {
        out += ",";
      }
      out += item.first + ":" + std::to_string(item.second);
    }
    return out;
  }

  TargetSelector selector_;
  std::unique_ptr<LandingDecisionPlanner> dynamic_planner_;
  rclcpp::Publisher<msgs::msg::TargetDecision>::SharedPtr publisher_;
  rclcpp::Publisher<msgs::msg::LandingCandidateArray>::SharedPtr candidates_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::Subscription<msgs::msg::PredictedStrike>::SharedPtr pre_aim_sub_;
  rclcpp::Subscription<msgs::msg::PredictedStrike>::SharedPtr strike_adjust_sub_;
  rclcpp::TimerBase::SharedPtr fallback_timer_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::string target_decision_topic_;
  std::string landing_candidates_topic_;
  std::string pre_aim_strike_topic_;
  std::string strike_adjust_topic_;
  bool strike_adjust_active_ = false;
  bool processed_valid_strike_ = false;
  bool last_valid_ = false;
  int last_candidate_count_ = 0;
  int last_hard_valid_count_ = 0;
  double last_score_ = 0.0;
  std::string last_mode_ = "fixed_center";
  std::string last_reason_ = "startup";
  std::string last_phase_ = "none";
  std::map<std::string, int> last_reject_reasons_;
};

}  // namespace decision

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<decision::DecisionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
