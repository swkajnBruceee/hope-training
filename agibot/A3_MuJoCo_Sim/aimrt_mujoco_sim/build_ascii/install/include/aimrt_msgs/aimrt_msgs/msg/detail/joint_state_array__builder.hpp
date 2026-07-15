// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from aimrt_msgs:msg/JointStateArray.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__JOINT_STATE_ARRAY__BUILDER_HPP_
#define AIMRT_MSGS__MSG__DETAIL__JOINT_STATE_ARRAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "aimrt_msgs/msg/detail/joint_state_array__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace aimrt_msgs
{

namespace msg
{

namespace builder
{

class Init_JointStateArray_joints
{
public:
  explicit Init_JointStateArray_joints(::aimrt_msgs::msg::JointStateArray & msg)
  : msg_(msg)
  {}
  ::aimrt_msgs::msg::JointStateArray joints(::aimrt_msgs::msg::JointStateArray::_joints_type arg)
  {
    msg_.joints = std::move(arg);
    return std::move(msg_);
  }

private:
  ::aimrt_msgs::msg::JointStateArray msg_;
};

class Init_JointStateArray_header
{
public:
  Init_JointStateArray_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointStateArray_joints header(::aimrt_msgs::msg::JointStateArray::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_JointStateArray_joints(msg_);
  }

private:
  ::aimrt_msgs::msg::JointStateArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::aimrt_msgs::msg::JointStateArray>()
{
  return aimrt_msgs::msg::builder::Init_JointStateArray_header();
}

}  // namespace aimrt_msgs

#endif  // AIMRT_MSGS__MSG__DETAIL__JOINT_STATE_ARRAY__BUILDER_HPP_
