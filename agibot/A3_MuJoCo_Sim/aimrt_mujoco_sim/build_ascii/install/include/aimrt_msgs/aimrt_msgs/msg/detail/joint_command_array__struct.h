// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from aimrt_msgs:msg/JointCommandArray.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__JOINT_COMMAND_ARRAY__STRUCT_H_
#define AIMRT_MSGS__MSG__DETAIL__JOINT_COMMAND_ARRAY__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "aimrt_msgs/msg/detail/message_header__struct.h"
// Member 'joints'
#include "aimrt_msgs/msg/detail/joint_command__struct.h"

/// Struct defined in msg/JointCommandArray in the package aimrt_msgs.
typedef struct aimrt_msgs__msg__JointCommandArray
{
  aimrt_msgs__msg__MessageHeader header;
  aimrt_msgs__msg__JointCommand__Sequence joints;
} aimrt_msgs__msg__JointCommandArray;

// Struct for a sequence of aimrt_msgs__msg__JointCommandArray.
typedef struct aimrt_msgs__msg__JointCommandArray__Sequence
{
  aimrt_msgs__msg__JointCommandArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} aimrt_msgs__msg__JointCommandArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AIMRT_MSGS__MSG__DETAIL__JOINT_COMMAND_ARRAY__STRUCT_H_
