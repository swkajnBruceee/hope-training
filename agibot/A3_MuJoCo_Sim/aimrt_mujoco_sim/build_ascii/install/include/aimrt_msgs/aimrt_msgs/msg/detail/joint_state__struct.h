// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from aimrt_msgs:msg/JointState.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__JOINT_STATE__STRUCT_H_
#define AIMRT_MSGS__MSG__DETAIL__JOINT_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'name'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/JointState in the package aimrt_msgs.
typedef struct aimrt_msgs__msg__JointState
{
  rosidl_runtime_c__String name;
  double position;
  double velocity;
  double effort;
} aimrt_msgs__msg__JointState;

// Struct for a sequence of aimrt_msgs__msg__JointState.
typedef struct aimrt_msgs__msg__JointState__Sequence
{
  aimrt_msgs__msg__JointState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} aimrt_msgs__msg__JointState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AIMRT_MSGS__MSG__DETAIL__JOINT_STATE__STRUCT_H_
