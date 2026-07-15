// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from aimrt_msgs:msg/TouchSensorStateArray.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__STRUCT_HPP_
#define AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__STRUCT_HPP_

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
#include "aimrt_msgs/msg/detail/message_header__struct.hpp"
// Member 'states'
#include "aimrt_msgs/msg/detail/touch_sensor_state__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__aimrt_msgs__msg__TouchSensorStateArray __attribute__((deprecated))
#else
# define DEPRECATED__aimrt_msgs__msg__TouchSensorStateArray __declspec(deprecated)
#endif

namespace aimrt_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct TouchSensorStateArray_
{
  using Type = TouchSensorStateArray_<ContainerAllocator>;

  explicit TouchSensorStateArray_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit TouchSensorStateArray_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    aimrt_msgs::msg::MessageHeader_<ContainerAllocator>;
  _header_type header;
  using _names_type =
    std::vector<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>>;
  _names_type names;
  using _states_type =
    std::vector<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator>>>;
  _states_type states;

  // setters for named parameter idiom
  Type & set__header(
    const aimrt_msgs::msg::MessageHeader_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__names(
    const std::vector<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>> & _arg)
  {
    this->names = _arg;
    return *this;
  }
  Type & set__states(
    const std::vector<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<aimrt_msgs::msg::TouchSensorState_<ContainerAllocator>>> & _arg)
  {
    this->states = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator> *;
  using ConstRawPtr =
    const aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__aimrt_msgs__msg__TouchSensorStateArray
    std::shared_ptr<aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__aimrt_msgs__msg__TouchSensorStateArray
    std::shared_ptr<aimrt_msgs::msg::TouchSensorStateArray_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TouchSensorStateArray_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->names != other.names) {
      return false;
    }
    if (this->states != other.states) {
      return false;
    }
    return true;
  }
  bool operator!=(const TouchSensorStateArray_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TouchSensorStateArray_

// alias to use template instance with default allocator
using TouchSensorStateArray =
  aimrt_msgs::msg::TouchSensorStateArray_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace aimrt_msgs

#endif  // AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__STRUCT_HPP_
