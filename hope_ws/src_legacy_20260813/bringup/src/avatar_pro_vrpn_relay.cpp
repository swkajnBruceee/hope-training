#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace {

constexpr const char * kPoseType = "geometry_msgs/msg/PoseStamped";

struct CandidateMotion
{
  bool has_last_pos{false};
  std::array<double, 3> last_pos{};
  double last_rx{0.0};
  double speed_ema{0.0};
};

bool contains_pose_type(const std::vector<std::string> & types)
{
  return std::find(types.begin(), types.end(), kPoseType) != types.end();
}

}  // namespace

class AvatarProVrpnRelay : public rclcpp::Node
{
public:
  AvatarProVrpnRelay()
  : Node("avatar_pro_vrpn_relay")
  {
    declare_parameter<std::string>("vrpn_namespace", "/vrpn_mocap");
    declare_parameter<std::string>("ppt_object", "PPT");
    declare_parameter<std::string>("p1_object", "P1");
    declare_parameter<std::string>("p2_object", "P2");
    declare_parameter<std::string>("ball_tracking_mode", "auto");
    declare_parameter<std::string>("ball_object", "");
    declare_parameter<std::string>("world_frame", "world");
    declare_parameter<std::string>("table_pose_topic", "/table/pose");
    declare_parameter<std::string>("p1_pose_topic", "/P1/pose");
    declare_parameter<std::string>("p2_pose_topic", "/P2/pose");
    declare_parameter<std::string>("ball_point_topic", "/ball/point");
    declare_parameter<std::string>("poses_topic", "/poses");
    declare_parameter<std::vector<std::string>>("pose_array_order", {"ball", "PPT", "P1", "P2"});
    declare_parameter<bool>("publish_tf", true);
    declare_parameter<bool>("publish_pose_array", true);
    declare_parameter<double>("discovery_period_s", 0.5);
    declare_parameter<double>("ball_motion_ema_alpha", 0.8);
    declare_parameter<double>("ball_lock_speed_mps", 0.2);
    declare_parameter<double>("ball_switch_ratio", 2.0);
    declare_parameter<double>("ball_stale_s", 2.0);

    vrpn_ns_ = get_parameter("vrpn_namespace").as_string();
    while (!vrpn_ns_.empty() && vrpn_ns_.back() == '/') {
      vrpn_ns_.pop_back();
    }
    ppt_object_ = get_parameter("ppt_object").as_string();
    p1_object_ = get_parameter("p1_object").as_string();
    p2_object_ = get_parameter("p2_object").as_string();

    std::string ball_object = get_parameter("ball_object").as_string();
    trim(ball_object);
    mode_ = get_parameter("ball_tracking_mode").as_string();
    trim(mode_);
    std::transform(mode_.begin(), mode_.end(), mode_.begin(), ::tolower);
    if (mode_ != "auto" && mode_ != "rigid_body") {
      RCLCPP_ERROR(
        get_logger(),
        "unknown ball_tracking_mode '%s'; expected 'auto' or 'rigid_body'. Falling back to 'auto'.",
        mode_.c_str());
      mode_ = "auto";
    }
    if (mode_ == "rigid_body") {
      std::string lower_ball_object = ball_object;
      std::transform(lower_ball_object.begin(), lower_ball_object.end(), lower_ball_object.begin(), ::tolower);
      if (ball_object.empty() || lower_ball_object == "auto") {
        RCLCPP_ERROR(
          get_logger(),
          "ball_tracking_mode='rigid_body' needs ball_object. Falling back to 'auto'.");
        mode_ = "auto";
      } else {
        ball_name_ = ball_object;
      }
    }
    if (mode_ == "auto") {
      std::string lower_ball_object = ball_object;
      std::transform(lower_ball_object.begin(), lower_ball_object.end(), lower_ball_object.begin(), ::tolower);
      if (!ball_object.empty() && lower_ball_object != "auto") {
        RCLCPP_WARN(
          get_logger(),
          "ball_tracking_mode='auto' ignores ball_object='%s'. Use 'rigid_body' to pin the ball.",
          ball_object.c_str());
      }
      ball_name_.clear();
    }

    world_frame_ = get_parameter("world_frame").as_string();
    publish_tf_ = get_parameter("publish_tf").as_bool();
    publish_pose_array_ = get_parameter("publish_pose_array").as_bool();
    pose_array_order_ = get_parameter("pose_array_order").as_string_array();
    ema_alpha_ = get_parameter("ball_motion_ema_alpha").as_double();
    lock_speed_ = get_parameter("ball_lock_speed_mps").as_double();
    switch_ratio_ = get_parameter("ball_switch_ratio").as_double();
    stale_s_ = get_parameter("ball_stale_s").as_double();
    discovery_period_s_ = get_parameter("discovery_period_s").as_double();

    rclcpp::QoS sensor_qos(rclcpp::KeepLast(1));
    sensor_qos.best_effort();
    sensor_qos.durability_volatile();
    sensor_qos_ = sensor_qos;

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    pose_array_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
      get_parameter("poses_topic").as_string(), sensor_qos_);
    ball_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
      get_parameter("ball_point_topic").as_string(), sensor_qos_);
    pose_publishers_["PPT"] = create_publisher<geometry_msgs::msg::PoseStamped>(
      get_parameter("table_pose_topic").as_string(), sensor_qos_);
    pose_publishers_["P1"] = create_publisher<geometry_msgs::msg::PoseStamped>(
      get_parameter("p1_pose_topic").as_string(), sensor_qos_);
    pose_publishers_["P2"] = create_publisher<geometry_msgs::msg::PoseStamped>(
      get_parameter("p2_pose_topic").as_string(), sensor_qos_);

    auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(discovery_period_s_));
    discovery_timer_ = create_wall_timer(period, std::bind(&AvatarProVrpnRelay::discover_and_select, this));

    const std::string ball_point_topic = get_parameter("ball_point_topic").as_string();
    if (!ball_name_.empty()) {
      RCLCPP_INFO(
        get_logger(),
        "ball_tracking_mode=rigid_body: tracking named rigid body '%s' -> %s",
        ball_name_.c_str(), ball_point_topic.c_str());
    } else {
      RCLCPP_INFO(
        get_logger(),
        "ball_tracking_mode=auto: locking onto the moving non-rigid marker -> %s",
        ball_point_topic.c_str());
    }
  }

private:
  static void trim(std::string & value)
  {
    auto not_space = [](unsigned char ch) {return !std::isspace(ch);};
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  }

  double now_seconds()
  {
    return static_cast<double>(get_clock()->now().nanoseconds()) * 1e-9;
  }

  std::string topic_to_sender(const std::string & topic) const
  {
    const std::string prefix = vrpn_ns_ + "/";
    if (topic.rfind(prefix, 0) != 0) {
      return "";
    }
    const std::string remainder = topic.substr(prefix.size());
    std::vector<std::string> segments;
    std::size_t start = 0;
    while (start <= remainder.size()) {
      std::size_t slash = remainder.find('/', start);
      if (slash == std::string::npos) {
        segments.push_back(remainder.substr(start));
        break;
      }
      segments.push_back(remainder.substr(start, slash - start));
      start = slash + 1;
    }
    if (segments.size() < 2) {
      return "";
    }
    const std::string & last = segments.back();
    if (last != "pose" && !(last.rfind("pose", 0) == 0 && all_digits(last.substr(4)))) {
      return "";
    }
    return segments.front();
  }

  static bool all_digits(const std::string & value)
  {
    return !value.empty() && std::all_of(value.begin(), value.end(), ::isdigit);
  }

  std::string classify(const std::string & sender) const
  {
    if (sender == ppt_object_) {
      return "PPT";
    }
    if (sender == p1_object_) {
      return "P1";
    }
    if (sender == p2_object_) {
      return "P2";
    }
    if (!ball_name_.empty()) {
      return sender == ball_name_ ? "ball" : "";
    }
    return "ball_candidate";
  }

  void discover_and_select()
  {
    for (const auto & [topic, types] : get_topic_names_and_types()) {
      if (subscriptions_.count(topic) != 0 || !contains_pose_type(types)) {
        continue;
      }
      std::string sender = topic_to_sender(topic);
      if (sender.empty()) {
        continue;
      }
      std::string key = classify(sender);
      if (key.empty()) {
        continue;
      }
      auto sub = create_subscription<geometry_msgs::msg::PoseStamped>(
        topic, sensor_qos_,
        [this, topic, key](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          on_pose(topic, key, *msg);
        });
      subscriptions_[topic] = sub;
      topic_key_[topic] = key;
      if (key == "ball_candidate") {
        motion_[topic] = CandidateMotion{};
      }
      RCLCPP_INFO(get_logger(), "discovered %s -> %s", topic.c_str(), key.c_str());
    }

    if (ball_name_.empty()) {
      select_ball();
    }
  }

  void select_ball()
  {
    const double now = now_seconds();
    std::vector<std::pair<std::string, CandidateMotion>> live;
    for (const auto & entry : motion_) {
      const CandidateMotion & motion = entry.second;
      if (motion.has_last_pos && (now - motion.last_rx) <= stale_s_) {
        live.push_back(entry);
      }
    }
    if (live.empty()) {
      if (!ball_topic_.empty()) {
        RCLCPP_WARN(get_logger(), "ball lost: no live marker candidates");
        ball_topic_.clear();
      }
      return;
    }

    if (live.size() == 1) {
      set_ball_topic(live.front().first, "sole marker candidate");
      return;
    }

    std::sort(
      live.begin(), live.end(),
      [](const auto & lhs, const auto & rhs) {
        return lhs.second.speed_ema > rhs.second.speed_ema;
      });
    const std::string & best_topic = live.front().first;
    const CandidateMotion & best = live.front().second;
    const double runner_speed = live.size() > 1 ? live[1].second.speed_ema : 0.0;

    if (!ball_topic_.empty()) {
      auto locked_it = std::find_if(
        live.begin(), live.end(),
        [this](const auto & item) {return item.first == ball_topic_;});
      if (locked_it != live.end()) {
        if (best_topic == ball_topic_) {
          return;
        }
        if (best.speed_ema < lock_speed_) {
          return;
        }
        if (best.speed_ema < locked_it->second.speed_ema * switch_ratio_) {
          return;
        }
        set_ball_topic(best_topic, "faster marker");
        return;
      }
    }

    if (best.speed_ema >= lock_speed_ && best.speed_ema >= runner_speed * switch_ratio_) {
      set_ball_topic(best_topic, "moving marker");
    }
  }

  void set_ball_topic(const std::string & topic, const std::string & reason)
  {
    if (topic == ball_topic_) {
      return;
    }
    ball_topic_ = topic;
    auto it = motion_.find(topic);
    if (it != motion_.end()) {
      RCLCPP_INFO(
        get_logger(), "ball -> %s (%s %.2f m/s)",
        topic.c_str(), reason.c_str(), it->second.speed_ema);
    } else {
      RCLCPP_INFO(get_logger(), "ball -> %s (%s)", topic.c_str(), reason.c_str());
    }
  }

  void on_pose(
    const std::string & topic,
    const std::string & key,
    const geometry_msgs::msg::PoseStamped & msg)
  {
    auto publisher_it = pose_publishers_.find(key);
    if (publisher_it != pose_publishers_.end()) {
      auto normalized = normalize(msg);
      latest_[key] = normalized;
      publisher_it->second->publish(normalized);
      if (publish_tf_) {
        broadcast_tf(key, normalized);
      }
      return;
    }

    if (key == "ball_candidate") {
      update_motion(topic, msg);
      if (topic != ball_topic_) {
        return;
      }
    }
    emit_ball(normalize(msg));
  }

  void update_motion(const std::string & topic, const geometry_msgs::msg::PoseStamped & msg)
  {
    CandidateMotion & motion = motion_[topic];
    const double now = now_seconds();
    const std::array<double, 3> pos{
      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z};
    if (motion.has_last_pos) {
      const double dt = now - motion.last_rx;
      if (dt > 1e-4) {
        const double dx = pos[0] - motion.last_pos[0];
        const double dy = pos[1] - motion.last_pos[1];
        const double dz = pos[2] - motion.last_pos[2];
        const double speed = std::sqrt(dx * dx + dy * dy + dz * dz) / dt;
        motion.speed_ema = ema_alpha_ * motion.speed_ema + (1.0 - ema_alpha_) * speed;
      }
    }
    motion.last_pos = pos;
    motion.last_rx = now;
    motion.has_last_pos = true;
  }

  geometry_msgs::msg::PoseStamped normalize(const geometry_msgs::msg::PoseStamped & msg) const
  {
    geometry_msgs::msg::PoseStamped normalized = msg;
    if (normalized.header.frame_id.empty()) {
      normalized.header.frame_id = world_frame_;
    }
    return normalized;
  }

  void emit_ball(const geometry_msgs::msg::PoseStamped & normalized)
  {
    latest_["ball"] = normalized;
    geometry_msgs::msg::PointStamped out;
    out.header = normalized.header;
    out.point = normalized.pose.position;
    ball_pub_->publish(out);
    if (publish_tf_) {
      broadcast_tf("ball", normalized);
    }
    if (publish_pose_array_) {
      publish_poses(normalized.header.stamp);
    }
  }

  void broadcast_tf(const std::string & object_key, const geometry_msgs::msg::PoseStamped & msg)
  {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header = msg.header;
    if (tf_msg.header.frame_id.empty()) {
      tf_msg.header.frame_id = world_frame_;
    }
    tf_msg.child_frame_id = object_key;
    tf_msg.transform.translation.x = msg.pose.position.x;
    tf_msg.transform.translation.y = msg.pose.position.y;
    tf_msg.transform.translation.z = msg.pose.position.z;
    tf_msg.transform.rotation = msg.pose.orientation;
    tf_broadcaster_->sendTransform(tf_msg);
  }

  void publish_poses(const builtin_interfaces::msg::Time & stamp)
  {
    if (latest_.count("ball") == 0) {
      return;
    }
    geometry_msgs::msg::PoseArray msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = world_frame_;
    for (const auto & key : pose_array_order_) {
      auto it = latest_.find(key);
      if (it != latest_.end()) {
        msg.poses.push_back(it->second.pose);
      }
    }
    pose_array_pub_->publish(msg);
  }

  std::string vrpn_ns_;
  std::string ppt_object_;
  std::string p1_object_;
  std::string p2_object_;
  std::string mode_;
  std::string ball_name_;
  std::string world_frame_;
  std::string ball_topic_;
  bool publish_tf_{true};
  bool publish_pose_array_{true};
  double discovery_period_s_{0.5};
  double ema_alpha_{0.8};
  double lock_speed_{0.2};
  double switch_ratio_{2.0};
  double stale_s_{2.0};
  std::vector<std::string> pose_array_order_;
  rclcpp::QoS sensor_qos_{rclcpp::KeepLast(1)};
  std::unordered_map<std::string, geometry_msgs::msg::PoseStamped> latest_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr ball_pub_;
  std::unordered_map<
    std::string,
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> pose_publishers_;
  std::unordered_map<
    std::string,
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> subscriptions_;
  std::unordered_map<std::string, std::string> topic_key_;
  std::unordered_map<std::string, CandidateMotion> motion_;
  rclcpp::TimerBase::SharedPtr discovery_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AvatarProVrpnRelay>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
