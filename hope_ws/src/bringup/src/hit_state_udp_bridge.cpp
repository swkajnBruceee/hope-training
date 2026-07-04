// SPDX-License-Identifier: Apache-2.0
//
// hit_state_udp_bridge
//
// ROS 2 -> UDP bridge for the HOPE hit-state topic. It is the C++ counterpart
// of the Python ``ball_truth_udp_bridge``: it subscribes to ``/hit/state``
// (``msgs/msg/HitState``) and republishes the same fields to a local UDP
// socket so the Isaac-side debug_draw overlay can render the planned hit
// point, target landing, racket velocity / normal, and the in/out ball
// velocity arrows directly in the viewport (no Python in the ROS loop).
//
// Wire format
// -----------
// Magic:    "HITS"  (4 bytes, ASCII)
// Sequence: uint32 little-endian (monotonically increasing)
// Validity: uint32 little-endian (1 if hit_state.valid, else 0)
// HasPlan:  uint32 little-endian (1 if state indicates a real plan, else 0)
// Then, IF HasPlan == 1:
//   hit_position (3 doubles, LE)
//   target_land  (3 doubles, LE)
//   ball_v_in    (3 doubles, LE)
//   ball_v_out   (3 doubles, LE)
//   racket_vel   (3 doubles, LE)
//   racket_normal(3 doubles, LE)
//
// All coordinates are in the same frame as the ``HitState.header.frame_id``
// (HOPE ``world``). The Isaac overlay drops points that fall outside the
// scene bounds, just like the trajectory overlay does.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "msgs/msg/hit_state.hpp"
#include "rclcpp/rclcpp.hpp"

namespace {

constexpr std::array<char, 4> kMagic = {'H', 'I', 'T', 'S'};

constexpr std::size_t kHeaderBytes =
  sizeof(char) * 4 /* magic */ +
  sizeof(std::uint32_t) * 3 /* seq, valid, has_plan */;

constexpr std::size_t kPayloadVecs = 6;
constexpr std::size_t kPayloadBytes = kPayloadVecs * 3 * sizeof(double);

constexpr std::size_t kMaxPacketBytes = kHeaderBytes + kPayloadBytes;

template <typename T>
void appendLE(std::vector<std::uint8_t> & out, const T & value)
{
  static_assert(std::is_trivially_copyable<T>::value, "appendLE requires POD");
  const auto * bytes = reinterpret_cast<const std::uint8_t *>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(T));
}

void appendVec3(std::vector<std::uint8_t> & out, double x, double y, double z)
{
  appendLE<double>(out, std::isfinite(x) ? x : 0.0);
  appendLE<double>(out, std::isfinite(y) ? y : 0.0);
  appendLE<double>(out, std::isfinite(z) ? z : 0.0);
}

}  // namespace

class HitStateUdpBridge : public rclcpp::Node
{
public:
  HitStateUdpBridge()
  : rclcpp::Node("hit_state_udp_bridge")
  {
    declare_parameter<std::string>("hit_state_topic", "/hit/state");
    declare_parameter<std::string>("udp_host", "127.0.0.1");
    declare_parameter<int>("udp_port", 19533);
    declare_parameter<int>("drain_limit", 64);
    declare_parameter<double>("min_send_period_s", 0.03);

    hit_state_topic_ = get_parameter("hit_state_topic").as_string();
    udp_host_ = get_parameter("udp_host").as_string();
    udp_port_ = get_parameter("udp_port").as_int();
    drain_limit_ = get_parameter("drain_limit").as_int();
    min_send_period_s_ = get_parameter("min_send_period_s").as_double();

    subscription_ = create_subscription<msgs::msg::HitState>(
      hit_state_topic_,
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
      std::bind(&HitStateUdpBridge::hit_state_cb, this, std::placeholders::_1));

    setup_socket();

    RCLCPP_INFO(
      get_logger(),
      "publishing %s as HITS-packets to udp://%s:%d (min_period=%.3fs)",
      hit_state_topic_.c_str(), udp_host_.c_str(), udp_port_, min_send_period_s_);
  }

  ~HitStateUdpBridge() override
  {
    if (socket_fd_ >= 0) {
      close(socket_fd_);
      socket_fd_ = -1;
    }
  }

private:
  void setup_socket()
  {
    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
      throw std::runtime_error("failed to create UDP socket");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(udp_port_));
    if (inet_pton(AF_INET, udp_host_.c_str(), &addr.sin_addr) != 1) {
      throw std::runtime_error("invalid udp_host; expected IPv4 literal");
    }
    target_addr_ = addr;

    RCLCPP_INFO(
      get_logger(), "hit_state_udp_bridge ready: topic=%s -> udp %s:%d",
      hit_state_topic_.c_str(), udp_host_.c_str(), udp_port_);
  }

  // Decide whether the HitState carries a real plan worth drawing.
  // We treat anything with ``valid == true`` as drawable; otherwise the
  // overlay can keep the previous frame for ``stale_keep_s`` and then clear.
  static bool hasPlan(const msgs::msg::HitState & msg)
  {
    if (!msg.valid) return false;
    if (!std::isfinite(msg.hit_position.x) ||
        !std::isfinite(msg.hit_position.y) ||
        !std::isfinite(msg.hit_position.z)) {
      return false;
    }
    // Make sure the planned hit point is reasonably close to the court.
    const double x = msg.hit_position.x;
    const double y = msg.hit_position.y;
    const double z = msg.hit_position.z;
    if (z < -0.05 || z > 1.5) return false;
    if (x < -0.7 || x > 3.5) return false;
    if (y < -1.7 || y > 0.2) return false;
    return true;
  }

  void hit_state_cb(const msgs::msg::HitState::SharedPtr msg)
  {
    const double now_s = this->now().seconds();
    if (last_send_t_ > 0.0 && (now_s - last_send_t_) < min_send_period_s_) {
      // Throttle: avoid blasting 100Hz updates into the viewport.
      return;
    }

    std::vector<std::uint8_t> payload;
    payload.reserve(kMaxPacketBytes);

    // Magic + header.
    payload.insert(payload.end(), kMagic.begin(), kMagic.end());
    const std::uint32_t seq = sequence_++;
    appendLE<std::uint32_t>(payload, seq);
    appendLE<std::uint32_t>(payload, msg->valid ? 1U : 0U);
    const bool plan = hasPlan(*msg);
    appendLE<std::uint32_t>(payload, plan ? 1U : 0U);

    if (plan) {
      appendVec3(payload, msg->hit_position.x, msg->hit_position.y, msg->hit_position.z);
      appendVec3(payload, msg->target_land.x, msg->target_land.y, msg->target_land.z);
      appendVec3(payload,
                 msg->ball_velocity_incoming.x,
                 msg->ball_velocity_incoming.y,
                 msg->ball_velocity_incoming.z);
      appendVec3(payload,
                 msg->ball_velocity_outgoing.x,
                 msg->ball_velocity_outgoing.y,
                 msg->ball_velocity_outgoing.z);
      appendVec3(payload,
                 msg->racket_velocity.x,
                 msg->racket_velocity.y,
                 msg->racket_velocity.z);
      appendVec3(payload,
                 msg->racket_normal.x,
                 msg->racket_normal.y,
                 msg->racket_normal.z);
    }

    const ssize_t sent = sendto(
      socket_fd_, payload.data(), payload.size(), 0,
      reinterpret_cast<const sockaddr *>(&target_addr_), sizeof(target_addr_));
    if (sent < 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "sendto failed: %s", std::strerror(errno));
      return;
    }
    last_send_t_ = now_s;
    (void)drain_limit_;  // reserved for future "drain to latest" support
  }

  std::string hit_state_topic_;
  std::string udp_host_;
  int udp_port_{19533};
  int drain_limit_{64};
  double min_send_period_s_{0.03};

  int socket_fd_{-1};
  sockaddr_in target_addr_{};

  std::uint32_t sequence_{0};
  double last_send_t_{-1.0};

  rclcpp::Subscription<msgs::msg::HitState>::SharedPtr subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HitStateUdpBridge>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
