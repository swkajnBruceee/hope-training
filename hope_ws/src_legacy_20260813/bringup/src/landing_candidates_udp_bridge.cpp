// SPDX-License-Identifier: Apache-2.0
//
// ROS 2 -> UDP bridge for landing candidate visualization.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "msgs/msg/landing_candidate_array.hpp"
#include "rclcpp/rclcpp.hpp"

namespace {

constexpr std::array<char, 4> kMagic = {'L', 'C', 'A', 'N'};
constexpr std::size_t kMaxCandidates = 64;

template <typename T>
void appendLE(std::vector<std::uint8_t> & out, const T & value)
{
  static_assert(std::is_trivially_copyable<T>::value, "appendLE requires POD");
  const auto * bytes = reinterpret_cast<const std::uint8_t *>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(T));
}

void appendDouble(std::vector<std::uint8_t> & out, double value)
{
  appendLE<double>(out, std::isfinite(value) ? value : 0.0);
}

}  // namespace

class LandingCandidatesUdpBridge : public rclcpp::Node
{
public:
  LandingCandidatesUdpBridge()
  : rclcpp::Node("landing_candidates_udp_bridge")
  {
    declare_parameter<std::string>("candidates_topic", "/planner/landing_candidates");
    declare_parameter<std::string>("udp_host", "127.0.0.1");
    declare_parameter<int>("udp_port", 19534);
    declare_parameter<double>("min_send_period_s", 0.05);

    candidates_topic_ = get_parameter("candidates_topic").as_string();
    udp_host_ = get_parameter("udp_host").as_string();
    udp_port_ = get_parameter("udp_port").as_int();
    min_send_period_s_ = get_parameter("min_send_period_s").as_double();

    setup_socket();
    subscription_ = create_subscription<msgs::msg::LandingCandidateArray>(
      candidates_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
      std::bind(&LandingCandidatesUdpBridge::candidatesCb, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "landing candidates UDP bridge ready: topic=%s -> udp %s:%d",
      candidates_topic_.c_str(), udp_host_.c_str(), udp_port_);
  }

  ~LandingCandidatesUdpBridge() override
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
  }

  void candidatesCb(const msgs::msg::LandingCandidateArray::SharedPtr msg)
  {
    const double now_s = now().seconds();
    if (last_send_t_ > 0.0 && (now_s - last_send_t_) < min_send_period_s_) {
      return;
    }
    last_send_t_ = now_s;

    const std::uint32_t count =
      static_cast<std::uint32_t>(std::min<std::size_t>(msg->candidates.size(), kMaxCandidates));

    std::vector<std::uint8_t> payload;
    payload.reserve(12 + count * 48);
    payload.insert(payload.end(), kMagic.begin(), kMagic.end());
    appendLE<std::uint32_t>(payload, sequence_++);
    appendLE<std::uint32_t>(payload, count);

    for (std::uint32_t i = 0; i < count; ++i) {
      const auto & c = msg->candidates[i];
      appendDouble(payload, c.target_land.x);
      appendDouble(payload, c.target_land.y);
      appendDouble(payload, c.target_land.z);
      appendDouble(payload, c.delta_t_flight);
      appendDouble(payload, c.score);
      const std::uint32_t state = c.selected ? 2U : (c.hard_valid ? 1U : 0U);
      appendLE<std::uint32_t>(payload, state);
    }

    const auto sent = ::sendto(
      socket_fd_, payload.data(), payload.size(), 0,
      reinterpret_cast<const sockaddr *>(&target_addr_), sizeof(target_addr_));
    if (sent < 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "failed to send candidates UDP packet");
    }
  }

  std::string candidates_topic_;
  std::string udp_host_;
  int udp_port_ = 19534;
  double min_send_period_s_ = 0.05;
  int socket_fd_ = -1;
  sockaddr_in target_addr_{};
  std::uint32_t sequence_ = 0;
  double last_send_t_ = 0.0;
  rclcpp::Subscription<msgs::msg::LandingCandidateArray>::SharedPtr subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LandingCandidatesUdpBridge>());
  rclcpp::shutdown();
  return 0;
}
