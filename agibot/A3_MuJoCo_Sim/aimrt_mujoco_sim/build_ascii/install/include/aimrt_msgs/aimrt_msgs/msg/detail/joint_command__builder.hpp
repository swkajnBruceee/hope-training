// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from aimrt_msgs:msg/JointCommand.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__JOINT_COMMAND__BUILDER_HPP_
#define AIMRT_MSGS__MSG__DETAIL__JOINT_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "aimrt_msgs/msg/detail/joint_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace aimrt_msgs
{

namespace msg
{

namespace builder
{

class Init_JointCommand_damping
{
public:
  explicit Init_JointCommand_damping(::aimrt_msgs::msg::JointCommand & msg)
  : msg_(msg)
  {}
  ::aimrt_msgs::msg::JointCommand damping(::aimrt_msgs::msg::JointCommand::_damping_type arg)
  {
    msg_.damping = std::move(arg);
    return std::move(msg_);
  }

private:
  ::aimrt_msgs::msg::JointCommand msg_;
};

class Init_JointCommand_stiffness
{
public:
  explicit Init_JointCommand_stiffness(::aimrt_msgs::msg::JointCommand & msg)
  : msg_(msg)
  {}
  Init_JointCommand_damping stiffness(::aimrt_msgs::msg::JointCommand::_stiffness_type arg)
  {
    msg_.stiffness = std::move(arg);
    return Init_JointCommand_damping(msg_);
  }

private:
  ::aimrt_msgs::msg::JointCommand msg_;
};

class Init_JointCommand_effort
{
public:
  explicit Init_JointCommand_effort(::aimrt_msgs::msg::JointCommand & msg)
  : msg_(msg)
  {}
  Init_JointCommand_stiffness effort(::aimrt_msgs::msg::JointCommand::_effort_type arg)
  {
    msg_.effort = std::move(arg);
    return Init_JointCommand_stiffness(msg_);
  }

private:
  ::aimrt_msgs::msg::JointCommand msg_;
};

class Init_JointCommand_velocity
{
public:
  explicit Init_JointCommand_velocity(::aimrt_msgs::msg::JointCommand & msg)
  : msg_(msg)
  {}
  Init_JointCommand_effort velocity(::aimrt_msgs::msg::JointCommand::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_JointCommand_effort(msg_);
  }

private:
  ::aimrt_msgs::msg::JointCommand msg_;
};

class Init_JointCommand_position
{
public:
  explicit Init_JointCommand_position(::aimrt_msgs::msg::JointCommand & msg)
  : msg_(msg)
  {}
  Init_JointCommand_velocity position(::aimrt_msgs::msg::JointCommand::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_JointCommand_velocity(msg_);
  }

private:
  ::aimrt_msgs::msg::JointCommand msg_;
};

class Init_JointCommand_name
{
public:
  Init_JointCommand_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointCommand_position name(::aimrt_msgs::msg::JointCommand::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_JointCommand_position(msg_);
  }

private:
  ::aimrt_msgs::msg::JointCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::aimrt_msgs::msg::JointCommand>()
{
  return aimrt_msgs::msg::builder::Init_JointCommand_name();
}

}  // namespace aimrt_msgs

#endif  // AIMRT_MSGS__MSG__DETAIL__JOINT_COMMAND__BUILDER_HPP_
