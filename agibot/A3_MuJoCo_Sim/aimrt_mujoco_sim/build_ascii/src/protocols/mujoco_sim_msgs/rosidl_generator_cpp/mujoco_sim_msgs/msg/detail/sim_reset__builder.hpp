// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_sim_msgs:msg/SimReset.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__BUILDER_HPP_
#define MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_sim_msgs/msg/detail/sim_reset__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_sim_msgs
{

namespace msg
{

namespace builder
{

class Init_SimReset_clear_ctrl
{
public:
  explicit Init_SimReset_clear_ctrl(::mujoco_sim_msgs::msg::SimReset & msg)
  : msg_(msg)
  {}
  ::mujoco_sim_msgs::msg::SimReset clear_ctrl(::mujoco_sim_msgs::msg::SimReset::_clear_ctrl_type arg)
  {
    msg_.clear_ctrl = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_sim_msgs::msg::SimReset msg_;
};

class Init_SimReset_zero_all_velocities
{
public:
  explicit Init_SimReset_zero_all_velocities(::mujoco_sim_msgs::msg::SimReset & msg)
  : msg_(msg)
  {}
  Init_SimReset_clear_ctrl zero_all_velocities(::mujoco_sim_msgs::msg::SimReset::_zero_all_velocities_type arg)
  {
    msg_.zero_all_velocities = std::move(arg);
    return Init_SimReset_clear_ctrl(msg_);
  }

private:
  ::mujoco_sim_msgs::msg::SimReset msg_;
};

class Init_SimReset_joint_state
{
public:
  explicit Init_SimReset_joint_state(::mujoco_sim_msgs::msg::SimReset & msg)
  : msg_(msg)
  {}
  Init_SimReset_zero_all_velocities joint_state(::mujoco_sim_msgs::msg::SimReset::_joint_state_type arg)
  {
    msg_.joint_state = std::move(arg);
    return Init_SimReset_zero_all_velocities(msg_);
  }

private:
  ::mujoco_sim_msgs::msg::SimReset msg_;
};

class Init_SimReset_set_joints
{
public:
  explicit Init_SimReset_set_joints(::mujoco_sim_msgs::msg::SimReset & msg)
  : msg_(msg)
  {}
  Init_SimReset_joint_state set_joints(::mujoco_sim_msgs::msg::SimReset::_set_joints_type arg)
  {
    msg_.set_joints = std::move(arg);
    return Init_SimReset_joint_state(msg_);
  }

private:
  ::mujoco_sim_msgs::msg::SimReset msg_;
};

class Init_SimReset_pelvis_twist
{
public:
  explicit Init_SimReset_pelvis_twist(::mujoco_sim_msgs::msg::SimReset & msg)
  : msg_(msg)
  {}
  Init_SimReset_set_joints pelvis_twist(::mujoco_sim_msgs::msg::SimReset::_pelvis_twist_type arg)
  {
    msg_.pelvis_twist = std::move(arg);
    return Init_SimReset_set_joints(msg_);
  }

private:
  ::mujoco_sim_msgs::msg::SimReset msg_;
};

class Init_SimReset_set_base_twist
{
public:
  explicit Init_SimReset_set_base_twist(::mujoco_sim_msgs::msg::SimReset & msg)
  : msg_(msg)
  {}
  Init_SimReset_pelvis_twist set_base_twist(::mujoco_sim_msgs::msg::SimReset::_set_base_twist_type arg)
  {
    msg_.set_base_twist = std::move(arg);
    return Init_SimReset_pelvis_twist(msg_);
  }

private:
  ::mujoco_sim_msgs::msg::SimReset msg_;
};

class Init_SimReset_pelvis_pose
{
public:
  explicit Init_SimReset_pelvis_pose(::mujoco_sim_msgs::msg::SimReset & msg)
  : msg_(msg)
  {}
  Init_SimReset_set_base_twist pelvis_pose(::mujoco_sim_msgs::msg::SimReset::_pelvis_pose_type arg)
  {
    msg_.pelvis_pose = std::move(arg);
    return Init_SimReset_set_base_twist(msg_);
  }

private:
  ::mujoco_sim_msgs::msg::SimReset msg_;
};

class Init_SimReset_set_base
{
public:
  explicit Init_SimReset_set_base(::mujoco_sim_msgs::msg::SimReset & msg)
  : msg_(msg)
  {}
  Init_SimReset_pelvis_pose set_base(::mujoco_sim_msgs::msg::SimReset::_set_base_type arg)
  {
    msg_.set_base = std::move(arg);
    return Init_SimReset_pelvis_pose(msg_);
  }

private:
  ::mujoco_sim_msgs::msg::SimReset msg_;
};

class Init_SimReset_keyframe_id
{
public:
  explicit Init_SimReset_keyframe_id(::mujoco_sim_msgs::msg::SimReset & msg)
  : msg_(msg)
  {}
  Init_SimReset_set_base keyframe_id(::mujoco_sim_msgs::msg::SimReset::_keyframe_id_type arg)
  {
    msg_.keyframe_id = std::move(arg);
    return Init_SimReset_set_base(msg_);
  }

private:
  ::mujoco_sim_msgs::msg::SimReset msg_;
};

class Init_SimReset_mode
{
public:
  explicit Init_SimReset_mode(::mujoco_sim_msgs::msg::SimReset & msg)
  : msg_(msg)
  {}
  Init_SimReset_keyframe_id mode(::mujoco_sim_msgs::msg::SimReset::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return Init_SimReset_keyframe_id(msg_);
  }

private:
  ::mujoco_sim_msgs::msg::SimReset msg_;
};

class Init_SimReset_header
{
public:
  Init_SimReset_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SimReset_mode header(::mujoco_sim_msgs::msg::SimReset::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_SimReset_mode(msg_);
  }

private:
  ::mujoco_sim_msgs::msg::SimReset msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_sim_msgs::msg::SimReset>()
{
  return mujoco_sim_msgs::msg::builder::Init_SimReset_header();
}

}  // namespace mujoco_sim_msgs

#endif  // MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__BUILDER_HPP_
