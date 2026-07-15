// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from aimrt_msgs:msg/JointCommand.idl
// generated code does not contain a copyright notice
#include "aimrt_msgs/msg/detail/joint_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `name`
#include "rosidl_runtime_c/string_functions.h"

bool
aimrt_msgs__msg__JointCommand__init(aimrt_msgs__msg__JointCommand * msg)
{
  if (!msg) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__init(&msg->name)) {
    aimrt_msgs__msg__JointCommand__fini(msg);
    return false;
  }
  // position
  // velocity
  // effort
  // stiffness
  // damping
  return true;
}

void
aimrt_msgs__msg__JointCommand__fini(aimrt_msgs__msg__JointCommand * msg)
{
  if (!msg) {
    return;
  }
  // name
  rosidl_runtime_c__String__fini(&msg->name);
  // position
  // velocity
  // effort
  // stiffness
  // damping
}

bool
aimrt_msgs__msg__JointCommand__are_equal(const aimrt_msgs__msg__JointCommand * lhs, const aimrt_msgs__msg__JointCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->name), &(rhs->name)))
  {
    return false;
  }
  // position
  if (lhs->position != rhs->position) {
    return false;
  }
  // velocity
  if (lhs->velocity != rhs->velocity) {
    return false;
  }
  // effort
  if (lhs->effort != rhs->effort) {
    return false;
  }
  // stiffness
  if (lhs->stiffness != rhs->stiffness) {
    return false;
  }
  // damping
  if (lhs->damping != rhs->damping) {
    return false;
  }
  return true;
}

bool
aimrt_msgs__msg__JointCommand__copy(
  const aimrt_msgs__msg__JointCommand * input,
  aimrt_msgs__msg__JointCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__copy(
      &(input->name), &(output->name)))
  {
    return false;
  }
  // position
  output->position = input->position;
  // velocity
  output->velocity = input->velocity;
  // effort
  output->effort = input->effort;
  // stiffness
  output->stiffness = input->stiffness;
  // damping
  output->damping = input->damping;
  return true;
}

aimrt_msgs__msg__JointCommand *
aimrt_msgs__msg__JointCommand__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__JointCommand * msg = (aimrt_msgs__msg__JointCommand *)allocator.allocate(sizeof(aimrt_msgs__msg__JointCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(aimrt_msgs__msg__JointCommand));
  bool success = aimrt_msgs__msg__JointCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
aimrt_msgs__msg__JointCommand__destroy(aimrt_msgs__msg__JointCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    aimrt_msgs__msg__JointCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
aimrt_msgs__msg__JointCommand__Sequence__init(aimrt_msgs__msg__JointCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__JointCommand * data = NULL;

  if (size) {
    data = (aimrt_msgs__msg__JointCommand *)allocator.zero_allocate(size, sizeof(aimrt_msgs__msg__JointCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = aimrt_msgs__msg__JointCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        aimrt_msgs__msg__JointCommand__fini(&data[i - 1]);
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
aimrt_msgs__msg__JointCommand__Sequence__fini(aimrt_msgs__msg__JointCommand__Sequence * array)
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
      aimrt_msgs__msg__JointCommand__fini(&array->data[i]);
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

aimrt_msgs__msg__JointCommand__Sequence *
aimrt_msgs__msg__JointCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__JointCommand__Sequence * array = (aimrt_msgs__msg__JointCommand__Sequence *)allocator.allocate(sizeof(aimrt_msgs__msg__JointCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = aimrt_msgs__msg__JointCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
aimrt_msgs__msg__JointCommand__Sequence__destroy(aimrt_msgs__msg__JointCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    aimrt_msgs__msg__JointCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
aimrt_msgs__msg__JointCommand__Sequence__are_equal(const aimrt_msgs__msg__JointCommand__Sequence * lhs, const aimrt_msgs__msg__JointCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!aimrt_msgs__msg__JointCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
aimrt_msgs__msg__JointCommand__Sequence__copy(
  const aimrt_msgs__msg__JointCommand__Sequence * input,
  aimrt_msgs__msg__JointCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(aimrt_msgs__msg__JointCommand);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    aimrt_msgs__msg__JointCommand * data =
      (aimrt_msgs__msg__JointCommand *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!aimrt_msgs__msg__JointCommand__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          aimrt_msgs__msg__JointCommand__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!aimrt_msgs__msg__JointCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
