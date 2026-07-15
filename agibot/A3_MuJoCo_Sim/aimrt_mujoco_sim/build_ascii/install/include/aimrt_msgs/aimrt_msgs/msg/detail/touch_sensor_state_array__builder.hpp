// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from aimrt_msgs:msg/TouchSensorStateArray.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__BUILDER_HPP_
#define AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "aimrt_msgs/msg/detail/touch_sensor_state_array__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace aimrt_msgs
{

namespace msg
{

namespace builder
{

class Init_TouchSensorStateArray_states
{
public:
  explicit Init_TouchSensorStateArray_states(::aimrt_msgs::msg::TouchSensorStateArray & msg)
  : msg_(msg)
  {}
  ::aimrt_msgs::msg::TouchSensorStateArray states(::aimrt_msgs::msg::TouchSensorStateArray::_states_type arg)
  {
    msg_.states = std::move(arg);
    return std::move(msg_);
  }

private:
  ::aimrt_msgs::msg::TouchSensorStateArray msg_;
};

class Init_TouchSensorStateArray_names
{
public:
  explicit Init_TouchSensorStateArray_names(::aimrt_msgs::msg::TouchSensorStateArray & msg)
  : msg_(msg)
  {}
  Init_TouchSensorStateArray_states names(::aimrt_msgs::msg::TouchSensorStateArray::_names_type arg)
  {
    msg_.names = std::move(arg);
    return Init_TouchSensorStateArray_states(msg_);
  }

private:
  ::aimrt_msgs::msg::TouchSensorStateArray msg_;
};

class Init_TouchSensorStateArray_header
{
public:
  Init_TouchSensorStateArray_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_TouchSensorStateArray_names header(::aimrt_msgs::msg::TouchSensorStateArray::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_TouchSensorStateArray_names(msg_);
  }

private:
  ::aimrt_msgs::msg::TouchSensorStateArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::aimrt_msgs::msg::TouchSensorStateArray>()
{
  return aimrt_msgs::msg::builder::Init_TouchSensorStateArray_header();
}

}  // namespace aimrt_msgs

#endif  // AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__BUILDER_HPP_
