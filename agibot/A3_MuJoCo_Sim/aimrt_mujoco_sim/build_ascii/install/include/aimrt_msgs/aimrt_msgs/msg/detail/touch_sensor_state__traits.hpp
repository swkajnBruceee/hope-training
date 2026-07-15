// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from aimrt_msgs:msg/TouchSensorState.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__TRAITS_HPP_
#define AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "aimrt_msgs/msg/detail/touch_sensor_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace aimrt_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const TouchSensorState & msg,
  std::ostream & out)
{
  out << "{";
  // member: pressure
  {
    if (msg.pressure.size() == 0) {
      out << "pressure: []";
    } else {
      out << "pressure: [";
      size_t pending_items = msg.pressure.size();
      for (auto item : msg.pressure) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const TouchSensorState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: pressure
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.pressure.size() == 0) {
      out << "pressure: []\n";
    } else {
      out << "pressure:\n";
      for (auto item : msg.pressure) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const TouchSensorState & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace aimrt_msgs

namespace rosidl_generator_traits
{

[[deprecated("use aimrt_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const aimrt_msgs::msg::TouchSensorState & msg,
  std::ostream & out, size_t indentation = 0)
{
  aimrt_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use aimrt_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const aimrt_msgs::msg::TouchSensorState & msg)
{
  return aimrt_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<aimrt_msgs::msg::TouchSensorState>()
{
  return "aimrt_msgs::msg::TouchSensorState";
}

template<>
inline const char * name<aimrt_msgs::msg::TouchSensorState>()
{
  return "aimrt_msgs/msg/TouchSensorState";
}

template<>
struct has_fixed_size<aimrt_msgs::msg::TouchSensorState>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<aimrt_msgs::msg::TouchSensorState>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<aimrt_msgs::msg::TouchSensorState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__TRAITS_HPP_
