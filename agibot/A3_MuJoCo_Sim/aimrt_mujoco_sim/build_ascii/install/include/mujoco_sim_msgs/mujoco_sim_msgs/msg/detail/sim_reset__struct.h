// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_sim_msgs:msg/SimReset.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__STRUCT_H_
#define MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'MODE_ABSOLUTE'.
enum
{
  mujoco_sim_msgs__msg__SimReset__MODE_ABSOLUTE = 0
};

/// Constant 'MODE_KEYFRAME'.
enum
{
  mujoco_sim_msgs__msg__SimReset__MODE_KEYFRAME = 1
};

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'pelvis_pose'
#include "geometry_msgs/msg/detail/pose__struct.h"
// Member 'pelvis_twist'
#include "geometry_msgs/msg/detail/twist__struct.h"
// Member 'joint_state'
#include "sensor_msgs/msg/detail/joint_state__struct.h"

/// Struct defined in msg/SimReset in the package mujoco_sim_msgs.
typedef struct mujoco_sim_msgs__msg__SimReset
{
  std_msgs__msg__Header header;
  uint8_t mode;
  int32_t keyframe_id;
  bool set_base;
  geometry_msgs__msg__Pose pelvis_pose;
  bool set_base_twist;
  geometry_msgs__msg__Twist pelvis_twist;
  bool set_joints;
  sensor_msgs__msg__JointState joint_state;
  bool zero_all_velocities;
  bool clear_ctrl;
} mujoco_sim_msgs__msg__SimReset;

// Struct for a sequence of mujoco_sim_msgs__msg__SimReset.
typedef struct mujoco_sim_msgs__msg__SimReset__Sequence
{
  mujoco_sim_msgs__msg__SimReset * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_sim_msgs__msg__SimReset__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SIM_MSGS__MSG__DETAIL__SIM_RESET__STRUCT_H_
