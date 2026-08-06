#include "racket_strike_target_receiver.hpp"

#include <hope_msgs/msg/racket_strike_target.hpp>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <thread>
#include <utility>

namespace a3_deploy::control {
namespace {
bool Finite(double v) noexcept { return std::isfinite(v); }
bool Normalize(std::array<double, 3>& v) noexcept {
  double n2 = 0.0;
  for (double x : v) {
    if (!Finite(x)) return false;
    n2 += x * x;
  }
  const double n = std::sqrt(n2);
  if (!Finite(n) || n < 1.0e-8) return false;
  for (double& x : v) x /= n;
  return true;
}
std::int64_t StampNs(const builtin_interfaces::msg::Time& stamp) noexcept {
  return static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}
}  // namespace

struct RacketStrikeTargetReceiver::Impl {
  explicit Impl(Config c) : config(std::move(c)) {}
  Config config;
  rclcpp::Context::SharedPtr context;
  rclcpp::Node::SharedPtr node;
  rclcpp::Subscription<hope_msgs::msg::RacketStrikeTarget>::SharedPtr sub;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  std::thread spin_thread;
  mutable std::mutex mutex;
  ArmGoal latest{};
  std::chrono::steady_clock::time_point arrival{};
  bool have_latest{false};
  bool have_taken{false};
  std::uint64_t last_taken_sequence{0};
  std::uint64_t received{0};
  std::uint64_t accepted{0};
  std::uint64_t invalid{0};
  std::uint64_t frame_mismatch{0};
  std::uint64_t stale{0};
};

RacketStrikeTargetReceiver::RacketStrikeTargetReceiver(Config config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
RacketStrikeTargetReceiver::~RacketStrikeTargetReceiver() { Stop(); }

bool RacketStrikeTargetReceiver::Start(std::string& error) {
  if (!impl_ || impl_->node) return true;
  if (!Finite(impl_->config.max_sample_age_s) ||
      !Finite(impl_->config.actuation_lead_s) ||
      impl_->config.max_sample_age_s <= 0.0 ||
      impl_->config.actuation_lead_s < 0.0 ||
      impl_->config.actuation_lead_s > 0.05F) {
    error = "racket target receiver: invalid timing configuration";
    return false;
  }
  try {
    impl_->context = std::make_shared<rclcpp::Context>();
    rclcpp::InitOptions options;
    options.auto_initialize_logging(false);
    options.shutdown_on_signal = false;
    impl_->context->init(0, nullptr, options);
    rclcpp::NodeOptions node_options;
    node_options.context(impl_->context);
    impl_->node = std::make_shared<rclcpp::Node>(
        "mc_racket_strike_target_receiver", node_options);
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(1))
                         .best_effort().durability_volatile();
    impl_->sub = impl_->node->create_subscription<
        hope_msgs::msg::RacketStrikeTarget>(
        impl_->config.topic, qos,
        [this](hope_msgs::msg::RacketStrikeTarget::SharedPtr msg) {
          const auto arrival = std::chrono::steady_clock::now();
          std::lock_guard<std::mutex> lock(impl_->mutex);
          ++impl_->received;
          if (!msg->valid || msg->header.frame_id != impl_->config.expected_frame) {
            if (msg->header.frame_id != impl_->config.expected_frame) {
              ++impl_->frame_mismatch;
            } else {
              ++impl_->invalid;
            }
            return;
          }
          if (!Finite(msg->position.x) || !Finite(msg->position.y) ||
              !Finite(msg->position.z) || !Finite(msg->linear_velocity.x) ||
              !Finite(msg->linear_velocity.y) ||
              !Finite(msg->linear_velocity.z) ||
              !Finite(msg->time_to_strike) || msg->time_to_strike <= 0.0 ||
              !std::isfinite(msg->confidence) ||
              msg->confidence < impl_->config.minimum_confidence) {
            ++impl_->invalid;
            return;
          }
          ArmGoal goal{};
          goal.valid = true;
          goal.has_cartesian_position = true;
          goal.position_m = {msg->position.x, msg->position.y, msg->position.z};
          goal.has_cartesian_linear_velocity = true;
          goal.linear_velocity_mps = {msg->linear_velocity.x,
                                      msg->linear_velocity.y,
                                      msg->linear_velocity.z};
          goal.has_racket_normal = true;
          goal.racket_normal = {msg->normal.x, msg->normal.y, msg->normal.z};
          if (!Normalize(goal.racket_normal)) {
            ++impl_->invalid;
            return;
          }
          goal.has_time_to_strike = true;
          goal.source_time_to_strike_s = msg->time_to_strike;
          goal.actuation_lead_s = impl_->config.actuation_lead_s;
          goal.time_to_strike_s = std::max(
              0.0, msg->time_to_strike - impl_->config.actuation_lead_s);
          goal.sequence = msg->sequence_id;
          goal.source_stamp_ns = StampNs(msg->header.stamp);
          goal.source_deadline_ns = goal.source_stamp_ns > 0
              ? goal.source_stamp_ns + static_cast<std::int64_t>(
                    msg->time_to_strike * 1.0e9)
              : 0;
          goal.confidence = msg->confidence;
          impl_->latest = goal;
          impl_->arrival = arrival;
          impl_->have_latest = true;
          ++impl_->accepted;
        });
    impl_->executor =
        std::make_unique<rclcpp::executors::SingleThreadedExecutor>(
            rclcpp::ExecutorOptions(), impl_->context);
    impl_->executor->add_node(impl_->node);
    impl_->spin_thread = std::thread([this] { impl_->executor->spin(); });
    return true;
  } catch (const std::exception& e) {
    error = std::string("racket target receiver: ") + e.what();
    Stop();
    return false;
  }
}

void RacketStrikeTargetReceiver::Stop() noexcept {
  if (!impl_) return;
  try {
    if (impl_->executor) impl_->executor->cancel();
    if (impl_->spin_thread.joinable()) impl_->spin_thread.join();
    if (impl_->executor && impl_->node) impl_->executor->remove_node(impl_->node);
    impl_->sub.reset();
    impl_->node.reset();
    impl_->executor.reset();
    if (impl_->context && impl_->context->is_valid()) impl_->context->shutdown("stop");
    impl_->context.reset();
  } catch (...) {
  }
}

bool RacketStrikeTargetReceiver::TakeLatest(ArmGoal& goal) noexcept {
  if (!impl_) return false;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->have_latest ||
      (impl_->have_taken &&
       impl_->latest.sequence == impl_->last_taken_sequence)) return false;
  const double age = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - impl_->arrival).count();
  if (!Finite(age) || age > impl_->config.max_sample_age_s) {
    impl_->last_taken_sequence = impl_->latest.sequence;
    impl_->have_taken = true;
    ++impl_->stale;
    return false;
  }
  goal = impl_->latest;
  goal.local_receipt_age_s = age;
  goal.time_to_strike_s = std::max(
      0.0, goal.time_to_strike_s - age);
  impl_->last_taken_sequence = goal.sequence;
  impl_->have_taken = true;
  return true;
}

std::uint64_t RacketStrikeTargetReceiver::ReceivedCount() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->received;
}
std::uint64_t RacketStrikeTargetReceiver::AcceptedCount() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->accepted;
}
std::uint64_t RacketStrikeTargetReceiver::InvalidCount() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->invalid;
}
std::uint64_t RacketStrikeTargetReceiver::FrameMismatchCount() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->frame_mismatch;
}
std::uint64_t RacketStrikeTargetReceiver::StaleCount() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->stale;
}

}  // namespace a3_deploy::control
