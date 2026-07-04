#include <algorithm>
#include <chrono>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "msgs/msg/target_decision.hpp"
#include "target_selector.h"

namespace decision {

class DecisionNode : public rclcpp::Node {
 public:
  DecisionNode() : rclcpp::Node("decision") {
    declare_parameter("target_decision_topic", std::string("/target_decision"));
    declare_parameter("publish_rate_hz", 10.0);

    const auto topic = get_parameter("target_decision_topic").as_string();
    const double publish_rate_hz = get_parameter("publish_rate_hz").as_double();
    const auto period = std::chrono::duration<double>(1.0 / std::max(1e-3, publish_rate_hz));

    publisher_ = create_publisher<msgs::msg::TargetDecision>(topic, rclcpp::QoS(10));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&DecisionNode::publishDecision, this));

    RCLCPP_INFO(get_logger(), "decision node publishing fixed target on %s", topic.c_str());
  }

 private:
  void publishDecision() {
    const auto decision = selector_.selectDefault();
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

  TargetSelector selector_;
  rclcpp::Publisher<msgs::msg::TargetDecision>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace decision

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<decision::DecisionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
