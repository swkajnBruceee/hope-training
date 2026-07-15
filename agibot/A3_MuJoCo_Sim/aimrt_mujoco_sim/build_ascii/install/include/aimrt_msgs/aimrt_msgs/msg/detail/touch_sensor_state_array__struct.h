// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from aimrt_msgs:msg/TouchSensorStateArray.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__STRUCT_H_
#define AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__STRUCT_H_

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
// Member 'names'
#include "rosidl_runtime_c/string.h"
// Member 'states'
#include "aimrt_msgs/msg/detail/touch_sensor_state__struct.h"

/// Struct defined in msg/TouchSensorStateArray in the package aimrt_msgs.
typedef struct aimrt_msgs__msg__TouchSensorStateArray
{
  aimrt_msgs__msg__MessageHeader header;
  rosidl_runtime_c__String__Sequence names;
  aimrt_msgs__msg__TouchSensorState__Sequence states;
} aimrt_msgs__msg__TouchSensorStateArray;

// Struct for a sequence of aimrt_msgs__msg__TouchSensorStateArray.
typedef struct aimrt_msgs__msg__TouchSensorStateArray__Sequence
{
  aimrt_msgs__msg__TouchSensorStateArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} aimrt_msgs__msg__TouchSensorStateArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AIMRT_MSGS__MSG__DETAIL__TOUCH_SENSOR_STATE_ARRAY__STRUCT_H_
