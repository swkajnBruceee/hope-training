// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_sim_msgs:msg/SimReset.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__STRUCT_HPP_
#define MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'pelvis_pose'
#include "geometry_msgs/msg/detail/pose__struct.hpp"
// Member 'pelvis_twist'
#include "geometry_msgs/msg/detail/twist__struct.hpp"
// Member 'joint_state'
#include "sensor_msgs/msg/detail/joint_state__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__mujoco_sim_msgs__msg__SimReset __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_sim_msgs__msg__SimReset __declspec(deprecated)
#endif

namespace mujoco_sim_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SimReset_
{
  using Type = SimReset_<ContainerAllocator>;

  explicit SimReset_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    pelvis_pose(_init),
    pelvis_twist(_init),
    joint_state(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mode = 0;
      this->keyframe_id = 0l;
      this->set_base = false;
      this->set_base_twist = false;
      this->set_joints = false;
      this->zero_all_velocities = false;
      this->clear_ctrl = false;
    }
  }

  explicit SimReset_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    pelvis_pose(_alloc, _init),
    pelvis_twist(_alloc, _init),
    joint_state(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mode = 0;
      this->keyframe_id = 0l;
      this->set_base = false;
      this->set_base_twist = false;
      this->set_joints = false;
      this->zero_all_velocities = false;
      this->clear_ctrl = false;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _mode_type =
    uint8_t;
  _mode_type mode;
  using _keyframe_id_type =
    int32_t;
  _keyframe_id_type keyframe_id;
  using _set_base_type =
    bool;
  _set_base_type set_base;
  using _pelvis_pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _pelvis_pose_type pelvis_pose;
  using _set_base_twist_type =
    bool;
  _set_base_twist_type set_base_twist;
  using _pelvis_twist_type =
    geometry_msgs::msg::Twist_<ContainerAllocator>;
  _pelvis_twist_type pelvis_twist;
  using _set_joints_type =
    bool;
  _set_joints_type set_joints;
  using _joint_state_type =
    sensor_msgs::msg::JointState_<ContainerAllocator>;
  _joint_state_type joint_state;
  using _zero_all_velocities_type =
    bool;
  _zero_all_velocities_type zero_all_velocities;
  using _clear_ctrl_type =
    bool;
  _clear_ctrl_type clear_ctrl;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__mode(
    const uint8_t & _arg)
  {
    this->mode = _arg;
    return *this;
  }
  Type & set__keyframe_id(
    const int32_t & _arg)
  {
    this->keyframe_id = _arg;
    return *this;
  }
  Type & set__set_base(
    const bool & _arg)
  {
    this->set_base = _arg;
    return *this;
  }
  Type & set__pelvis_pose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->pelvis_pose = _arg;
    return *this;
  }
  Type & set__set_base_twist(
    const bool & _arg)
  {
    this->set_base_twist = _arg;
    return *this;
  }
  Type & set__pelvis_twist(
    const geometry_msgs::msg::Twist_<ContainerAllocator> & _arg)
  {
    this->pelvis_twist = _arg;
    return *this;
  }
  Type & set__set_joints(
    const bool & _arg)
  {
    this->set_joints = _arg;
    return *this;
  }
  Type & set__joint_state(
    const sensor_msgs::msg::JointState_<ContainerAllocator> & _arg)
  {
    this->joint_state = _arg;
    return *this;
  }
  Type & set__zero_all_velocities(
    const bool & _arg)
  {
    this->zero_all_velocities = _arg;
    return *this;
  }
  Type & set__clear_ctrl(
    const bool & _arg)
  {
    this->clear_ctrl = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t MODE_ABSOLUTE =
    0u;
  static constexpr uint8_t MODE_KEYFRAME =
    1u;

  // pointer types
  using RawPtr =
    mujoco_sim_msgs::msg::SimReset_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_sim_msgs::msg::SimReset_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_sim_msgs::msg::SimReset_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_sim_msgs::msg::SimReset_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_sim_msgs::msg::SimReset_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_sim_msgs::msg::SimReset_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_sim_msgs::msg::SimReset_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_sim_msgs::msg::SimReset_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_sim_msgs::msg::SimReset_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_sim_msgs::msg::SimReset_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_sim_msgs__msg__SimReset
    std::shared_ptr<mujoco_sim_msgs::msg::SimReset_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_sim_msgs__msg__SimReset
    std::shared_ptr<mujoco_sim_msgs::msg::SimReset_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SimReset_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->mode != other.mode) {
      return false;
    }
    if (this->keyframe_id != other.keyframe_id) {
      return false;
    }
    if (this->set_base != other.set_base) {
      return false;
    }
    if (this->pelvis_pose != other.pelvis_pose) {
      return false;
    }
    if (this->set_base_twist != other.set_base_twist) {
      return false;
    }
    if (this->pelvis_twist != other.pelvis_twist) {
      return false;
    }
    if (this->set_joints != other.set_joints) {
      return false;
    }
    if (this->joint_state != other.joint_state) {
      return false;
    }
    if (this->zero_all_velocities != other.zero_all_velocities) {
      return false;
    }
    if (this->clear_ctrl != other.clear_ctrl) {
      return false;
    }
    return true;
  }
  bool operator!=(const SimReset_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SimReset_

// alias to use template instance with default allocator
using SimReset =
  mujoco_sim_msgs::msg::SimReset_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t SimReset_<ContainerAllocator>::MODE_ABSOLUTE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t SimReset_<ContainerAllocator>::MODE_KEYFRAME;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace mujoco_sim_msgs

#endif  // MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__STRUCT_HPP_
