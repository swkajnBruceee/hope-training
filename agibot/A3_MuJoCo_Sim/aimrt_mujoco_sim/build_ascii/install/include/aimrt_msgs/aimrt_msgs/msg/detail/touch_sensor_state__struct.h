// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from aimrt_msgs:msg/TouchSensorState.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__STRUCT_H_
#define AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'pressure'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/TouchSensorState in the package aimrt_msgs.
typedef struct aimrt_msgs__msg__TouchSensorState
{
  rosidl_runtime_c__int16__Sequence pressure;
} aimrt_msgs__msg__TouchSensorState;

// Struct for a sequence of aimrt_msgs__msg__TouchSensorState.
typedef struct aimrt_msgs__msg__TouchSensorState__Sequence
{
  aimrt_msgs__msg__TouchSensorState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} aimrt_msgs__msg__TouchSensorState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE__STRUCT_H_
