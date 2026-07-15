// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from aimrt_msgs:msg/TouchSensorState.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__BUILDER_HPP_
#define AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "aimrt_msgs/msg/detail/touch_sensor_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace aimrt_msgs
{

namespace msg
{

namespace builder
{

class Init_TouchSensorState_pressure
{
public:
  Init_TouchSensorState_pressure()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::aimrt_msgs::msg::TouchSensorState pressure(::aimrt_msgs::msg::TouchSensorState::_pressure_type arg)
  {
    msg_.pressure = std::move(arg);
    return std::move(msg_);
  }

private:
  ::aimrt_msgs::msg::TouchSensorState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::aimrt_msgs::msg::TouchSensorState>()
{
  return aimrt_msgs::msg::builder::Init_TouchSensorState_pressure();
}

}  // namespace aimrt_msgs

#endif  // AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__BUILDER_HPP_
