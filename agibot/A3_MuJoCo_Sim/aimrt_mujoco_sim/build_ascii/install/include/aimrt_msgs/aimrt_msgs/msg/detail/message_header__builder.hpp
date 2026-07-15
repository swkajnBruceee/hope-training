// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from aimrt_msgs:msg/MessageHeader.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__MESSAGE_HEADER__BUILDER_HPP_
#define AIMRT_MSGS__MSG__DETAIL__MESSAGE_HEADER__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "aimrt_msgs/msg/detail/message_header__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace aimrt_msgs
{

namespace msg
{

namespace builder
{

class Init_MessageHeader_sequence
{
public:
  explicit Init_MessageHeader_sequence(::aimrt_msgs::msg::MessageHeader & msg)
  : msg_(msg)
  {}
  ::aimrt_msgs::msg::MessageHeader sequence(::aimrt_msgs::msg::MessageHeader::_sequence_type arg)
  {
    msg_.sequence = std::move(arg);
    return std::move(msg_);
  }

private:
  ::aimrt_msgs::msg::MessageHeader msg_;
};

class Init_MessageHeader_frame_id
{
public:
  explicit Init_MessageHeader_frame_id(::aimrt_msgs::msg::MessageHeader & msg)
  : msg_(msg)
  {}
  Init_MessageHeader_sequence frame_id(::aimrt_msgs::msg::MessageHeader::_frame_id_type arg)
  {
    msg_.frame_id = std::move(arg);
    return Init_MessageHeader_sequence(msg_);
  }

private:
  ::aimrt_msgs::msg::MessageHeader msg_;
};

class Init_MessageHeader_stamp
{
public:
  Init_MessageHeader_stamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MessageHeader_frame_id stamp(::aimrt_msgs::msg::MessageHeader::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return Init_MessageHeader_frame_id(msg_);
  }

private:
  ::aimrt_msgs::msg::MessageHeader msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::aimrt_msgs::msg::MessageHeader>()
{
  return aimrt_msgs::msg::builder::Init_MessageHeader_stamp();
}

}  // namespace aimrt_msgs

#endif  // AIMRT_MSGS__MSG__DETAIL__MESSAGE_HEADER__BUILDER_HPP_
