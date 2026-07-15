// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from aimrt_msgs:msg/TouchSensorState.idl
// generated code does not contain a copyright notice
#include "aimrt_msgs/msg/detail/touch_sensor_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `pressure`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
aimrt_msgs__msg__TouchSensorState__init(aimrt_msgs__msg__TouchSensorState * msg)
{
  if (!msg) {
    return false;
  }
  // pressure
  if (!rosidl_runtime_c__int16__Sequence__init(&msg->pressure, 0)) {
    aimrt_msgs__msg__TouchSensorState__fini(msg);
    return false;
  }
  return true;
}

void
aimrt_msgs__msg__TouchSensorState__fini(aimrt_msgs__msg__TouchSensorState * msg)
{
  if (!msg) {
    return;
  }
  // pressure
  rosidl_runtime_c__int16__Sequence__fini(&msg->pressure);
}

bool
aimrt_msgs__msg__TouchSensorState__are_equal(const aimrt_msgs__msg__TouchSensorState * lhs, const aimrt_msgs__msg__TouchSensorState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // pressure
  if (!rosidl_runtime_c__int16__Sequence__are_equal(
      &(lhs->pressure), &(rhs->pressure)))
  {
    return false;
  }
  return true;
}

bool
aimrt_msgs__msg__TouchSensorState__copy(
  const aimrt_msgs__msg__TouchSensorState * input,
  aimrt_msgs__msg__TouchSensorState * output)
{
  if (!input || !output) {
    return false;
  }
  // pressure
  if (!rosidl_runtime_c__int16__Sequence__copy(
      &(input->pressure), &(output->pressure)))
  {
    return false;
  }
  return true;
}

aimrt_msgs__msg__TouchSensorState *
aimrt_msgs__msg__TouchSensorState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__TouchSensorState * msg = (aimrt_msgs__msg__TouchSensorState *)allocator.allocate(sizeof(aimrt_msgs__msg__TouchSensorState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(aimrt_msgs__msg__TouchSensorState));
  bool success = aimrt_msgs__msg__TouchSensorState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
aimrt_msgs__msg__TouchSensorState__destroy(aimrt_msgs__msg__TouchSensorState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    aimrt_msgs__msg__TouchSensorState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
aimrt_msgs__msg__TouchSensorState__Sequence__init(aimrt_msgs__msg__TouchSensorState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__TouchSensorState * data = NULL;

  if (size) {
    data = (aimrt_msgs__msg__TouchSensorState *)allocator.zero_allocate(size, sizeof(aimrt_msgs__msg__TouchSensorState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = aimrt_msgs__msg__TouchSensorState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        aimrt_msgs__msg__TouchSensorState__fini(&data[i - 1]);
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
aimrt_msgs__msg__TouchSensorState__Sequence__fini(aimrt_msgs__msg__TouchSensorState__Sequence * array)
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
      aimrt_msgs__msg__TouchSensorState__fini(&array->data[i]);
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

aimrt_msgs__msg__TouchSensorState__Sequence *
aimrt_msgs__msg__TouchSensorState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__TouchSensorState__Sequence * array = (aimrt_msgs__msg__TouchSensorState__Sequence *)allocator.allocate(sizeof(aimrt_msgs__msg__TouchSensorState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = aimrt_msgs__msg__TouchSensorState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
aimrt_msgs__msg__TouchSensorState__Sequence__destroy(aimrt_msgs__msg__TouchSensorState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    aimrt_msgs__msg__TouchSensorState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
aimrt_msgs__msg__TouchSensorState__Sequence__are_equal(const aimrt_msgs__msg__TouchSensorState__Sequence * lhs, const aimrt_msgs__msg__TouchSensorState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!aimrt_msgs__msg__TouchSensorState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
aimrt_msgs__msg__TouchSensorState__Sequence__copy(
  const aimrt_msgs__msg__TouchSensorState__Sequence * input,
  aimrt_msgs__msg__TouchSensorState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(aimrt_msgs__msg__TouchSensorState);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    aimrt_msgs__msg__TouchSensorState * data =
      (aimrt_msgs__msg__TouchSensorState *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!aimrt_msgs__msg__TouchSensorState__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          aimrt_msgs__msg__TouchSensorState__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!aimrt_msgs__msg__TouchSensorState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
