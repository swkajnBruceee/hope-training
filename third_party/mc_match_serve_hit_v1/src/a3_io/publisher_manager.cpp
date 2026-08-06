// Copyright (c) 2026, AgiBot Inc. All rights reserved.
// 对应 notes/a3_backend_plan.md §? / PR 6
#include "a3_io/publisher_manager.hpp"

namespace a3_io {

#ifdef HAS_A3_ROS_MSGS

A3PublisherManager::A3PublisherManager(Options opt)
    : leg_only_(opt.leg_only),
      arm_enabled_(opt.arm_enabled),
      waist_enabled_(opt.waist_enabled),
      waist_zero_(opt.waist_zero),
      waist_(std::move(opt.waist_publish_fn)),
      leg_(std::move(opt.leg_publish_fn)),
      arm_(std::move(opt.arm_publish_fn)),
      neck_(std::move(opt.neck_publish_fn)) {}

void A3PublisherManager::PublishAll(std::int64_t stamp_ns, std::uint32_t seq,
                                    const robot_io::RobotCommand& cmd_31) {
  if (waist_enabled_) {
    if (waist_zero_) {
      // Legacy hard-zero mode. Model3396 uses waist_zero=false and prepares
      // a measured-pose -> zero trajectory in the policy command itself.
      robot_io::RobotCommand zero = cmd_31;
      zero.q_des.segment(robot_io::kA3WaistStart, robot_io::kA3WaistCount).setZero();
      zero.dq_des.segment(robot_io::kA3WaistStart, robot_io::kA3WaistCount).setZero();
      zero.tau_ff.segment(robot_io::kA3WaistStart, robot_io::kA3WaistCount).setZero();
      zero.kp.segment(robot_io::kA3WaistStart, robot_io::kA3WaistCount).setZero();
      zero.kd.segment(robot_io::kA3WaistStart, robot_io::kA3WaistCount).setZero();
      waist_.Publish(stamp_ns, seq, zero);
    } else {
      waist_.Publish(stamp_ns, seq, cmd_31);
    }
  }
  leg_.Publish(stamp_ns, seq, cmd_31);
  if (arm_enabled_) arm_.Publish(stamp_ns, seq, cmd_31);
  if (!leg_only_) neck_.Publish(stamp_ns, seq, cmd_31);
}

#endif  // HAS_A3_ROS_MSGS

}  // namespace a3_io
