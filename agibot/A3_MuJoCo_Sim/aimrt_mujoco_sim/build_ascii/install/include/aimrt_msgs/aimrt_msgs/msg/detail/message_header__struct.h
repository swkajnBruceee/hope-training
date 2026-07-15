// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from aimrt_msgs:msg/MessageHeader.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__MESSAGE_HEADER__STRUCT_H_
#define AIMRT_MSGS__MSG__DETAIL__MESSAGE_HEADER__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"
// Member 'frame_id'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/MessageHeader in the package aimrt_msgs.
/**
  * World time stamp, which is expressed as seconds and nanoseconds.
 */
typedef struct aimrt_msgs__msg__MessageHeader
{
  builtin_interfaces__msg__Time stamp;
  /// Transform frame with which this data is associated.
  rosidl_runtime_c__String frame_id;
  /// Message's consecutively increasing ID
  uint32_t sequence;
} aimrt_msgs__msg__MessageHeader;

// Struct for a sequence of aimrt_msgs__msg__MessageHeader.
typedef struct aimrt_msgs__msg__MessageHeader__Sequence
{
  aimrt_msgs__msg__MessageHeader * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} aimrt_msgs__msg__MessageHeader__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AIMRT_MSGS__MSG__DETAIL__MESSAGE_HEADER__STRUCT_H_
