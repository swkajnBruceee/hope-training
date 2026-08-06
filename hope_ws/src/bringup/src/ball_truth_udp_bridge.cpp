#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "rclcpp/rclcpp.hpp"

namespace {

struct PackedBallPoint
{
  double x;
  double y;
  double z;
};

struct PackedStampedBallPoint
{
  double t;
  double x;
  double y;
  double z;
};

}  // namespace

class BallTruthUdpBridge : public rclcpp::Node
{
public:
  BallTruthUdpBridge()
  : Node("ball_truth_udp_bridge")
  {
    declare_parameter<std::string>("topic", "/ball/point");
    declare_parameter<std::string>("frame_id", "world");
    declare_parameter<std::string>("udp_host", "127.0.0.1");
    declare_parameter<int>("udp_port", 19531);
    declare_parameter<int>("drain_limit", 64);

    topic_ = get_parameter("topic").as_string();
    frame_id_ = get_parameter("frame_id").as_string();
    udp_host_ = get_parameter("udp_host").as_string();
    udp_port_ = get_parameter("udp_port").as_int();
    drain_limit_ = get_parameter("drain_limit").as_int();

    publisher_ = create_publisher<geometry_msgs::msg::PointStamped>(topic_, rclcpp::QoS(10));
    setup_socket();

    timer_ = create_wall_timer(
      std::chrono::milliseconds(1),
      std::bind(&BallTruthUdpBridge::drain_socket, this));

    RCLCPP_INFO(
      get_logger(), "publishing %s from udp://%s:%d",
      topic_.c_str(), udp_host_.c_str(), udp_port_);
  }

  ~BallTruthUdpBridge() override
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

    const int flags = fcntl(socket_fd_, F_GETFL, 0);
    if (flags < 0 || fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
      throw std::runtime_error("failed to set UDP socket non-blocking");
    }

    int reuse = 1;
    if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
      throw std::runtime_error("failed to set SO_REUSEADDR");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(udp_port_));
    if (inet_pton(AF_INET, udp_host_.c_str(), &addr.sin_addr) != 1) {
      throw std::runtime_error("invalid udp_host; expected IPv4 literal");
    }

    if (bind(socket_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
      throw std::runtime_error(
              std::string("failed to bind UDP socket: ") + std::strerror(errno));
    }
  }

  void drain_socket()
  {
    std::array<std::byte, sizeof(PackedStampedBallPoint)> buffer{};
    int drained = 0;
    while (drained < drain_limit_) {
      const ssize_t n = recv(socket_fd_, buffer.data(), buffer.size(), 0);
      if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          return;
        }
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "recv failed: %s", std::strerror(errno));
        return;
      }
      geometry_msgs::msg::PointStamped msg;
      msg.header.frame_id = frame_id_;

      if (n == static_cast<ssize_t>(sizeof(PackedStampedBallPoint))) {
        PackedStampedBallPoint point{};
        std::memcpy(&point, buffer.data(), sizeof(point));
        const auto sec = static_cast<int32_t>(point.t);
        const auto nanosec = static_cast<uint32_t>((point.t - static_cast<double>(sec)) * 1.0e9);
        msg.header.stamp.sec = sec;
        msg.header.stamp.nanosec = nanosec;
        msg.point.x = point.x;
        msg.point.y = point.y;
        msg.point.z = point.z;
      } else if (n == static_cast<ssize_t>(sizeof(PackedBallPoint))) {
        PackedBallPoint point{};
        std::memcpy(&point, buffer.data(), sizeof(point));
        msg.header.stamp = now();
        msg.point.x = point.x;
        msg.point.y = point.y;
        msg.point.z = point.z;
      } else {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "ignored UDP payload with %zd bytes", n);
        ++drained;
        continue;
      }

      publisher_->publish(msg);
      ++drained;
    }
  }

  std::string topic_;
  std::string frame_id_;
  std::string udp_host_;
  int udp_port_{19531};
  int drain_limit_{64};
  int socket_fd_{-1};
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BallTruthUdpBridge>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
