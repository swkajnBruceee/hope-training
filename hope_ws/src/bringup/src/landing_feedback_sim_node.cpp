// SPDX-License-Identifier: Apache-2.0
//
// Model-based landing feedback source used before a real racket execution
// stack is available. It turns a planned HitState into an "actual landing"
// sample by integrating the outgoing ball trajectory with configurable
// feedback physics.

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include "msgs/msg/hit_state.hpp"
#include "msgs/msg/landing_feedback.hpp"

namespace {

bool finiteVector(const Eigen::Vector3d & v)
{
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

}  // namespace

class LandingFeedbackSimNode : public rclcpp::Node
{
public:
  LandingFeedbackSimNode()
  : rclcpp::Node("landing_feedback_sim")
  {
    declare_parameter<std::string>("hit_state_topic", "/hit/state");
    declare_parameter<std::string>("feedback_topic", "/planner/landing_feedback");
    declare_parameter<double>("drag_k", 0.09375);
    declare_parameter<double>("gravity_x", 0.0);
    declare_parameter<double>("gravity_y", 0.0);
    declare_parameter<double>("gravity_z", -9.81);
    declare_parameter<double>("ball_radius", 0.02);
    declare_parameter<double>("table_length", 2.74);
    declare_parameter<double>("table_width", 1.525);
    declare_parameter<double>("dt", 0.001);
    declare_parameter<double>("max_horizon_s", 1.5);
    declare_parameter<double>("systematic_landing_bias_x", 0.0);
    declare_parameter<double>("systematic_landing_bias_y", 0.0);

    hit_state_topic_ = get_parameter("hit_state_topic").as_string();
    feedback_topic_ = get_parameter("feedback_topic").as_string();
    drag_k_ = get_parameter("drag_k").as_double();
    gravity_ = Eigen::Vector3d(
      get_parameter("gravity_x").as_double(),
      get_parameter("gravity_y").as_double(),
      get_parameter("gravity_z").as_double());
    ball_radius_ = get_parameter("ball_radius").as_double();
    table_length_ = get_parameter("table_length").as_double();
    table_width_ = get_parameter("table_width").as_double();
    dt_ = std::max(1e-4, get_parameter("dt").as_double());
    max_horizon_s_ = std::max(dt_, get_parameter("max_horizon_s").as_double());
    systematic_landing_bias_ = Eigen::Vector3d(
      get_parameter("systematic_landing_bias_x").as_double(),
      get_parameter("systematic_landing_bias_y").as_double(),
      0.0);

    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    sub_ = create_subscription<msgs::msg::HitState>(
      hit_state_topic_, qos,
      std::bind(&LandingFeedbackSimNode::hitStateCb, this, std::placeholders::_1));
    pub_ = create_publisher<msgs::msg::LandingFeedback>(feedback_topic_, qos);

    RCLCPP_INFO(
      get_logger(),
      "landing feedback sim ready: hit_state=%s feedback=%s drag=%.5f bias=(%.3f,%.3f)",
      hit_state_topic_.c_str(), feedback_topic_.c_str(), drag_k_,
      systematic_landing_bias_.x(), systematic_landing_bias_.y());
  }

private:
  std::optional<Eigen::Vector3d> firstTableLanding(
    const Eigen::Vector3d & p0,
    const Eigen::Vector3d & v0) const
  {
    if (!finiteVector(p0) || !finiteVector(v0)) {
      return std::nullopt;
    }

    Eigen::Vector3d p = p0;
    Eigen::Vector3d v = v0;
    for (double elapsed = 0.0; elapsed < max_horizon_s_; elapsed += dt_) {
      const double speed = v.norm();
      const Eigen::Vector3d a = -drag_k_ * speed * v + gravity_;
      const Eigen::Vector3d p_next = p + v * dt_ + 0.5 * a * dt_ * dt_;
      const Eigen::Vector3d v_next = v + a * dt_;

      if (p_next.z() <= ball_radius_ && v_next.z() < 0.0) {
        const double dz = p.z() - p_next.z();
        double frac = 0.5;
        if (std::abs(dz) > 1e-9) {
          frac = (p.z() - ball_radius_) / dz;
        }
        frac = std::max(0.0, std::min(1.0, frac));
        Eigen::Vector3d landing = p + frac * (p_next - p);
        landing.z() = 0.0;
        landing += systematic_landing_bias_;
        if (landing.x() >= 0.0 && landing.x() <= table_length_ &&
            landing.y() >= -table_width_ && landing.y() <= 0.0) {
          return landing;
        }
        return std::nullopt;
      }

      p = p_next;
      v = v_next;
    }
    return std::nullopt;
  }

  void hitStateCb(const msgs::msg::HitState::SharedPtr msg)
  {
    if (!msg->valid) {
      return;
    }

    const Eigen::Vector3d hit_position(
      msg->hit_position.x, msg->hit_position.y, msg->hit_position.z);
    const Eigen::Vector3d v_out(
      msg->ball_velocity_outgoing.x,
      msg->ball_velocity_outgoing.y,
      msg->ball_velocity_outgoing.z);
    const auto landing = firstTableLanding(hit_position, v_out);
    if (!landing.has_value()) {
      return;
    }

    msgs::msg::LandingFeedback out;
    out.header = msg->header;
    out.header.frame_id = "world";
    out.target_land = msg->target_land;
    out.actual_landing.x = landing->x();
    out.actual_landing.y = landing->y();
    out.actual_landing.z = landing->z();
    out.hit_position = msg->hit_position;
    out.ball_velocity_incoming = msg->ball_velocity_incoming;
    out.ball_velocity_outgoing = msg->ball_velocity_outgoing;
    out.return_flight_time = msg->return_flight_time;
    out.valid = true;
    out.source = "model_sim";
    pub_->publish(out);
  }

  std::string hit_state_topic_;
  std::string feedback_topic_;
  double drag_k_ = 0.09375;
  Eigen::Vector3d gravity_ = Eigen::Vector3d(0.0, 0.0, -9.81);
  double ball_radius_ = 0.02;
  double table_length_ = 2.74;
  double table_width_ = 1.525;
  double dt_ = 0.001;
  double max_horizon_s_ = 1.5;
  Eigen::Vector3d systematic_landing_bias_ = Eigen::Vector3d::Zero();
  rclcpp::Subscription<msgs::msg::HitState>::SharedPtr sub_;
  rclcpp::Publisher<msgs::msg::LandingFeedback>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LandingFeedbackSimNode>());
  rclcpp::shutdown();
  return 0;
}
