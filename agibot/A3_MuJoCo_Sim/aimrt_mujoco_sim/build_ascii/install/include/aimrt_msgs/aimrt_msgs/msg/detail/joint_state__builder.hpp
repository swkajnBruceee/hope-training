// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from aimrt_msgs:msg/JointState.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__JOINT_STATE__BUILDER_HPP_
#define AIMRT_MSGS__MSG__DETAIL__JOINT_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "aimrt_msgs/msg/detail/joint_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace aimrt_msgs
{

namespace msg
{

namespace builder
{

class Init_JointState_effort
{
public:
  explicit Init_JointState_effort(::aimrt_msgs::msg::JointState & msg)
  : msg_(msg)
  {}
  ::aimrt_msgs::msg::JointState effort(::aimrt_msgs::msg::JointState::_effort_type arg)
  {
    msg_.effort = std::move(arg);
    return std::move(msg_);
  }

private:
  ::aimrt_msgs::msg::JointState msg_;
};

class Init_JointState_velocity
{
public:
  explicit Init_JointState_velocity(::aimrt_msgs::msg::JointState & msg)
  : msg_(msg)
  {}
  Init_JointState_effort velocity(::aimrt_msgs::msg::JointState::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_JointState_effort(msg_);
  }

private:
  ::aimrt_msgs::msg::JointState msg_;
};

class Init_JointState_position
{
public:
  explicit Init_JointState_position(::aimrt_msgs::msg::JointState & msg)
  : msg_(msg)
  {}
  Init_JointState_velocity position(::aimrt_msgs::msg::JointState::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_JointState_velocity(msg_);
  }

private:
  ::aimrt_msgs::msg::JointState msg_;
};

class Init_JointState_name
{
public:
  Init_JointState_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointState_position name(::aimrt_msgs::msg::JointState::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_JointState_position(msg_);
  }

private:
  ::aimrt_msgs::msg::JointState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::aimrt_msgs::msg::JointState>()
{
  return aimrt_msgs::msg::builder::Init_JointState_name();
}

}  // namespace aimrt_msgs

#endif  // AIMRT_MSGS__MSG__DETAIL__JOINT_STATE__BUILDER_HPP_
