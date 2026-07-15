// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from aimrt_msgs:msg/TouchSensorState.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__STRUCT_HPP_
#define AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__aimrt_msgs__msg__TouchSensorState __attribute__((deprecated))
#else
# define DEPRECATED__aimrt_msgs__msg__TouchSensorState __declspec(deprecated)
#endif

namespace aimrt_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct TouchSensorState_
{
  using Type = TouchSensorState_<ContainerAllocator>;

  explicit TouchSensorState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit TouchSensorState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _pressure_type =
    std::vector<int16_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int16_t>>;
  _pressure_type pressure;

  // setters for named parameter idiom
  Type & set__pressure(
    const std::vector<int16_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int16_t>> & _arg)
  {
    this->pressure = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    aimrt_msgs::msg::TouchSensorState_<ContainerAllocator> *;
  using ConstRawPtr =
    const aimrt_msgs::msg::TouchSensorState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      aimrt_msgs::msg::TouchSensorState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      aimrt_msgs::msg::TouchSensorState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__aimrt_msgs__msg__TouchSensorState
    std::shared_ptr<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__aimrt_msgs__msg__TouchSensorState
    std::shared_ptr<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TouchSensorState_ & other) const
  {
    if (this->pressure != other.pressure) {
      return false;
    }
    return true;
  }
  bool operator!=(const TouchSensorState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TouchSensorState_

// alias to use template instance with default allocator
using TouchSensorState =
  aimrt_msgs::msg::TouchSensorState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace aimrt_msgs

#endif  // AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__STRUCT_HPP_
