// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mujoco_sim_msgs:msg/SimReset.idl
// generated code does not contain a copyright notice
#include "mujoco_sim_msgs/msg/detail/sim_reset__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `pelvis_pose`
#include "geometry_msgs/msg/detail/pose__functions.h"
// Member `pelvis_twist`
#include "geometry_msgs/msg/detail/twist__functions.h"
// Member `joint_state`
#include "sensor_msgs/msg/detail/joint_state__functions.h"

bool
mujoco_sim_msgs__msg__SimReset__init(mujoco_sim_msgs__msg__SimReset * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    mujoco_sim_msgs__msg__SimReset__fini(msg);
    return false;
  }
  // mode
  // keyframe_id
  // set_base
  // pelvis_pose
  if (!geometry_msgs__msg__Pose__init(&msg->pelvis_pose)) {
    mujoco_sim_msgs__msg__SimReset__fini(msg);
    return false;
  }
  // set_base_twist
  // pelvis_twist
  if (!geometry_msgs__msg__Twist__init(&msg->pelvis_twist)) {
    mujoco_sim_msgs__msg__SimReset__fini(msg);
    return false;
  }
  // set_joints
  // joint_state
  if (!sensor_msgs__msg__JointState__init(&msg->joint_state)) {
    mujoco_sim_msgs__msg__SimReset__fini(msg);
    return false;
  }
  // zero_all_velocities
  // clear_ctrl
  return true;
}

void
mujoco_sim_msgs__msg__SimReset__fini(mujoco_sim_msgs__msg__SimReset * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // mode
  // keyframe_id
  // set_base
  // pelvis_pose
  geometry_msgs__msg__Pose__fini(&msg->pelvis_pose);
  // set_base_twist
  // pelvis_twist
  geometry_msgs__msg__Twist__fini(&msg->pelvis_twist);
  // set_joints
  // joint_state
  sensor_msgs__msg__JointState__fini(&msg->joint_state);
  // zero_all_velocities
  // clear_ctrl
}

bool
mujoco_sim_msgs__msg__SimReset__are_equal(const mujoco_sim_msgs__msg__SimReset * lhs, const mujoco_sim_msgs__msg__SimReset * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // mode
  if (lhs->mode != rhs->mode) {
    return false;
  }
  // keyframe_id
  if (lhs->keyframe_id != rhs->keyframe_id) {
    return false;
  }
  // set_base
  if (lhs->set_base != rhs->set_base) {
    return false;
  }
  // pelvis_pose
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->pelvis_pose), &(rhs->pelvis_pose)))
  {
    return false;
  }
  // set_base_twist
  if (lhs->set_base_twist != rhs->set_base_twist) {
    return false;
  }
  // pelvis_twist
  if (!geometry_msgs__msg__Twist__are_equal(
      &(lhs->pelvis_twist), &(rhs->pelvis_twist)))
  {
    return false;
  }
  // set_joints
  if (lhs->set_joints != rhs->set_joints) {
    return false;
  }
  // joint_state
  if (!sensor_msgs__msg__JointState__are_equal(
      &(lhs->joint_state), &(rhs->joint_state)))
  {
    return false;
  }
  // zero_all_velocities
  if (lhs->zero_all_velocities != rhs->zero_all_velocities) {
    return false;
  }
  // clear_ctrl
  if (lhs->clear_ctrl != rhs->clear_ctrl) {
    return false;
  }
  return true;
}

bool
mujoco_sim_msgs__msg__SimReset__copy(
  const mujoco_sim_msgs__msg__SimReset * input,
  mujoco_sim_msgs__msg__SimReset * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // mode
  output->mode = input->mode;
  // keyframe_id
  output->keyframe_id = input->keyframe_id;
  // set_base
  output->set_base = input->set_base;
  // pelvis_pose
  if (!geometry_msgs__msg__Pose__copy(
      &(input->pelvis_pose), &(output->pelvis_pose)))
  {
    return false;
  }
  // set_base_twist
  output->set_base_twist = input->set_base_twist;
  // pelvis_twist
  if (!geometry_msgs__msg__Twist__copy(
      &(input->pelvis_twist), &(output->pelvis_twist)))
  {
    return false;
  }
  // set_joints
  output->set_joints = input->set_joints;
  // joint_state
  if (!sensor_msgs__msg__JointState__copy(
      &(input->joint_state), &(output->joint_state)))
  {
    return false;
  }
  // zero_all_velocities
  output->zero_all_velocities = input->zero_all_velocities;
  // clear_ctrl
  output->clear_ctrl = input->clear_ctrl;
  return true;
}

mujoco_sim_msgs__msg__SimReset *
mujoco_sim_msgs__msg__SimReset__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_sim_msgs__msg__SimReset * msg = (mujoco_sim_msgs__msg__SimReset *)allocator.allocate(sizeof(mujoco_sim_msgs__msg__SimReset), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_sim_msgs__msg__SimReset));
  bool success = mujoco_sim_msgs__msg__SimReset__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_sim_msgs__msg__SimReset__destroy(mujoco_sim_msgs__msg__SimReset * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_sim_msgs__msg__SimReset__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_sim_msgs__msg__SimReset__Sequence__init(mujoco_sim_msgs__msg__SimReset__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_sim_msgs__msg__SimReset * data = NULL;

  if (size) {
    data = (mujoco_sim_msgs__msg__SimReset *)allocator.zero_allocate(size, sizeof(mujoco_sim_msgs__msg__SimReset), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_sim_msgs__msg__SimReset__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_sim_msgs__msg__SimReset__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
mujoco_sim_msgs__msg__SimReset__Sequence__fini(mujoco_sim_msgs__msg__SimReset__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      mujoco_sim_msgs__msg__SimReset__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

mujoco_sim_msgs__msg__SimReset__Sequence *
mujoco_sim_msgs__msg__SimReset__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_sim_msgs__msg__SimReset__Sequence * array = (mujoco_sim_msgs__msg__SimReset__Sequence *)allocator.allocate(sizeof(mujoco_sim_msgs__msg__SimReset__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_sim_msgs__msg__SimReset__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_sim_msgs__msg__SimReset__Sequence__destroy(mujoco_sim_msgs__msg__SimReset__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_sim_msgs__msg__SimReset__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_sim_msgs__msg__SimReset__Sequence__are_equal(const mujoco_sim_msgs__msg__SimReset__Sequence * lhs, const mujoco_sim_msgs__msg__SimReset__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_sim_msgs__msg__SimReset__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_sim_msgs__msg__SimReset__Sequence__copy(
  const mujoco_sim_msgs__msg__SimReset__Sequence * input,
  mujoco_sim_msgs__msg__SimReset__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_sim_msgs__msg__SimReset);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_sim_msgs__msg__SimReset * data =
      (mujoco_sim_msgs__msg__SimReset *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_sim_msgs__msg__SimReset__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_sim_msgs__msg__SimReset__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_sim_msgs__msg__SimReset__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
