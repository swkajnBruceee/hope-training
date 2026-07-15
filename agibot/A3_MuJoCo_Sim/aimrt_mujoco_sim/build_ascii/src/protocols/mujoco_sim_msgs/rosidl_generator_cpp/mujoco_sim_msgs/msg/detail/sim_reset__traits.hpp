// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_sim_msgs:msg/SimReset.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__TRAITS_HPP_
#define MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_sim_msgs/msg/detail/sim_reset__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'pelvis_pose'
#include "geometry_msgs/msg/detail/pose__traits.hpp"
// Member 'pelvis_twist'
#include "geometry_msgs/msg/detail/twist__traits.hpp"
// Member 'joint_state'
#include "sensor_msgs/msg/detail/joint_state__traits.hpp"

namespace mujoco_sim_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const SimReset & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: mode
  {
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << ", ";
  }

  // member: keyframe_id
  {
    out << "keyframe_id: ";
    rosidl_generator_traits::value_to_yaml(msg.keyframe_id, out);
    out << ", ";
  }

  // member: set_base
  {
    out << "set_base: ";
    rosidl_generator_traits::value_to_yaml(msg.set_base, out);
    out << ", ";
  }

  // member: pelvis_pose
  {
    out << "pelvis_pose: ";
    to_flow_style_yaml(msg.pelvis_pose, out);
    out << ", ";
  }

  // member: set_base_twist
  {
    out << "set_base_twist: ";
    rosidl_generator_traits::value_to_yaml(msg.set_base_twist, out);
    out << ", ";
  }

  // member: pelvis_twist
  {
    out << "pelvis_twist: ";
    to_flow_style_yaml(msg.pelvis_twist, out);
    out << ", ";
  }

  // member: set_joints
  {
    out << "set_joints: ";
    rosidl_generator_traits::value_to_yaml(msg.set_joints, out);
    out << ", ";
  }

  // member: joint_state
  {
    out << "joint_state: ";
    to_flow_style_yaml(msg.joint_state, out);
    out << ", ";
  }

  // member: zero_all_velocities
  {
    out << "zero_all_velocities: ";
    rosidl_generator_traits::value_to_yaml(msg.zero_all_velocities, out);
    out << ", ";
  }

  // member: clear_ctrl
  {
    out << "clear_ctrl: ";
    rosidl_generator_traits::value_to_yaml(msg.clear_ctrl, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SimReset & msg,
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

  // member: mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << "\n";
  }

  // member: keyframe_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "keyframe_id: ";
    rosidl_generator_traits::value_to_yaml(msg.keyframe_id, out);
    out << "\n";
  }

  // member: set_base
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "set_base: ";
    rosidl_generator_traits::value_to_yaml(msg.set_base, out);
    out << "\n";
  }

  // member: pelvis_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pelvis_pose:\n";
    to_block_style_yaml(msg.pelvis_pose, out, indentation + 2);
  }

  // member: set_base_twist
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "set_base_twist: ";
    rosidl_generator_traits::value_to_yaml(msg.set_base_twist, out);
    out << "\n";
  }

  // member: pelvis_twist
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pelvis_twist:\n";
    to_block_style_yaml(msg.pelvis_twist, out, indentation + 2);
  }

  // member: set_joints
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "set_joints: ";
    rosidl_generator_traits::value_to_yaml(msg.set_joints, out);
    out << "\n";
  }

  // member: joint_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "joint_state:\n";
    to_block_style_yaml(msg.joint_state, out, indentation + 2);
  }

  // member: zero_all_velocities
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "zero_all_velocities: ";
    rosidl_generator_traits::value_to_yaml(msg.zero_all_velocities, out);
    out << "\n";
  }

  // member: clear_ctrl
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "clear_ctrl: ";
    rosidl_generator_traits::value_to_yaml(msg.clear_ctrl, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SimReset & msg, bool use_flow_style = false)
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

}  // namespace mujoco_sim_msgs

namespace rosidl_generator_traits
{

[[deprecated("use mujoco_sim_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const mujoco_sim_msgs::msg::SimReset & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_sim_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_sim_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_sim_msgs::msg::SimReset & msg)
{
  return mujoco_sim_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_sim_msgs::msg::SimReset>()
{
  return "mujoco_sim_msgs::msg::SimReset";
}

template<>
inline const char * name<mujoco_sim_msgs::msg::SimReset>()
{
  return "mujoco_sim_msgs/msg/SimReset";
}

template<>
struct has_fixed_size<mujoco_sim_msgs::msg::SimReset>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Pose>::value && has_fixed_size<geometry_msgs::msg::Twist>::value && has_fixed_size<sensor_msgs::msg::JointState>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<mujoco_sim_msgs::msg::SimReset>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Pose>::value && has_bounded_size<geometry_msgs::msg::Twist>::value && has_bounded_size<sensor_msgs::msg::JointState>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<mujoco_sim_msgs::msg::SimReset>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__TRAITS_HPP_
