// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from aimrt_msgs:msg/TouchSensorStateArray.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__TRAITS_HPP_
#define AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "aimrt_msgs/msg/detail/touch_sensor_state_array__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "aimrt_msgs/msg/detail/message_header__traits.hpp"
// Member 'states'
#include "aimrt_msgs/msg/detail/touch_sensor_state__traits.hpp"

namespace aimrt_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const TouchSensorStateArray & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: names
  {
    if (msg.names.size() == 0) {
      out << "names: []";
    } else {
      out << "names: [";
      size_t pending_items = msg.names.size();
      for (auto item : msg.names) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: states
  {
    if (msg.states.size() == 0) {
      out << "states: []";
    } else {
      out << "states: [";
      size_t pending_items = msg.states.size();
      for (auto item : msg.states) {
        to_flow_style_yaml(item, out);
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
  const TouchSensorStateArray & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: names
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.names.size() == 0) {
      out << "names: []\n";
    } else {
      out << "names:\n";
      for (auto item : msg.names) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: states
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.states.size() == 0) {
      out << "states: []\n";
    } else {
      out << "states:\n";
      for (auto item : msg.states) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const TouchSensorStateArray & msg, bool use_flow_style = false)
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
  const aimrt_msgs::msg::TouchSensorStateArray & msg,
  std::ostream & out, size_t indentation = 0)
{
  aimrt_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use aimrt_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const aimrt_msgs::msg::TouchSensorStateArray & msg)
{
  return aimrt_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<aimrt_msgs::msg::TouchSensorStateArray>()
{
  return "aimrt_msgs::msg::TouchSensorStateArray";
}

template<>
inline const char * name<aimrt_msgs::msg::TouchSensorStateArray>()
{
  return "aimrt_msgs/msg/TouchSensorStateArray";
}

template<>
struct has_fixed_size<aimrt_msgs::msg::TouchSensorStateArray>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<aimrt_msgs::msg::TouchSensorStateArray>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<aimrt_msgs::msg::TouchSensorStateArray>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__TRAITS_HPP_
