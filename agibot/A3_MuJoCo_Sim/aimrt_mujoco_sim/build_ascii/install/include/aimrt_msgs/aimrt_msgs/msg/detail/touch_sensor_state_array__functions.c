// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from aimrt_msgs:msg/TouchSensorStateArray.idl
// generated code does not contain a copyright notice
#include "aimrt_msgs/msg/detail/touch_sensor_state_array__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "aimrt_msgs/msg/detail/message_header__functions.h"
// Member `names`
#include "rosidl_runtime_c/string_functions.h"
// Member `states`
#include "aimrt_msgs/msg/detail/touch_sensor_state__functions.h"

bool
aimrt_msgs__msg__TouchSensorStateArray__init(aimrt_msgs__msg__TouchSensorStateArray * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!aimrt_msgs__msg__MessageHeader__init(&msg->header)) {
    aimrt_msgs__msg__TouchSensorStateArray__fini(msg);
    return false;
  }
  // names
  if (!rosidl_runtime_c__String__Sequence__init(&msg->names, 0)) {
    aimrt_msgs__msg__TouchSensorStateArray__fini(msg);
    return false;
  }
  // states
  if (!aimrt_msgs__msg__TouchSensorState__Sequence__init(&msg->states, 0)) {
    aimrt_msgs__msg__TouchSensorStateArray__fini(msg);
    return false;
  }
  return true;
}

void
aimrt_msgs__msg__TouchSensorStateArray__fini(aimrt_msgs__msg__TouchSensorStateArray * msg)
{
  if (!msg) {
    return;
  }
  // header
  aimrt_msgs__msg__MessageHeader__fini(&msg->header);
  // names
  rosidl_runtime_c__String__Sequence__fini(&msg->names);
  // states
  aimrt_msgs__msg__TouchSensorState__Sequence__fini(&msg->states);
}

bool
aimrt_msgs__msg__TouchSensorStateArray__are_equal(const aimrt_msgs__msg__TouchSensorStateArray * lhs, const aimrt_msgs__msg__TouchSensorStateArray * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!aimrt_msgs__msg__MessageHeader__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // names
  if (!rosidl_runtime_c__String__Sequence__are_equal(
      &(lhs->names), &(rhs->names)))
  {
    return false;
  }
  // states
  if (!aimrt_msgs__msg__TouchSensorState__Sequence__are_equal(
      &(lhs->states), &(rhs->states)))
  {
    return false;
  }
  return true;
}

bool
aimrt_msgs__msg__TouchSensorStateArray__copy(
  const aimrt_msgs__msg__TouchSensorStateArray * input,
  aimrt_msgs__msg__TouchSensorStateArray * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!aimrt_msgs__msg__MessageHeader__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // names
  if (!rosidl_runtime_c__String__Sequence__copy(
      &(input->names), &(output->names)))
  {
    return false;
  }
  // states
  if (!aimrt_msgs__msg__TouchSensorState__Sequence__copy(
      &(input->states), &(output->states)))
  {
    return false;
  }
  return true;
}

aimrt_msgs__msg__TouchSensorStateArray *
aimrt_msgs__msg__TouchSensorStateArray__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__TouchSensorStateArray * msg = (aimrt_msgs__msg__TouchSensorStateArray *)allocator.allocate(sizeof(aimrt_msgs__msg__TouchSensorStateArray), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(aimrt_msgs__msg__TouchSensorStateArray));
  bool success = aimrt_msgs__msg__TouchSensorStateArray__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
aimrt_msgs__msg__TouchSensorStateArray__destroy(aimrt_msgs__msg__TouchSensorStateArray * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    aimrt_msgs__msg__TouchSensorStateArray__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
aimrt_msgs__msg__TouchSensorStateArray__Sequence__init(aimrt_msgs__msg__TouchSensorStateArray__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__TouchSensorStateArray * data = NULL;

  if (size) {
    data = (aimrt_msgs__msg__TouchSensorStateArray *)allocator.zero_allocate(size, sizeof(aimrt_msgs__msg__TouchSensorStateArray), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = aimrt_msgs__msg__TouchSensorStateArray__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        aimrt_msgs__msg__TouchSensorStateArray__fini(&data[i - 1]);
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
aimrt_msgs__msg__TouchSensorStateArray__Sequence__fini(aimrt_msgs__msg__TouchSensorStateArray__Sequence * array)
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
      aimrt_msgs__msg__TouchSensorStateArray__fini(&array->data[i]);
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

aimrt_msgs__msg__TouchSensorStateArray__Sequence *
aimrt_msgs__msg__TouchSensorStateArray__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__TouchSensorStateArray__Sequence * array = (aimrt_msgs__msg__TouchSensorStateArray__Sequence *)allocator.allocate(sizeof(aimrt_msgs__msg__TouchSensorStateArray__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = aimrt_msgs__msg__TouchSensorStateArray__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
aimrt_msgs__msg__TouchSensorStateArray__Sequence__destroy(aimrt_msgs__msg__TouchSensorStateArray__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    aimrt_msgs__msg__TouchSensorStateArray__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
aimrt_msgs__msg__TouchSensorStateArray__Sequence__are_equal(const aimrt_msgs__msg__TouchSensorStateArray__Sequence * lhs, const aimrt_msgs__msg__TouchSensorStateArray__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!aimrt_msgs__msg__TouchSensorStateArray__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
aimrt_msgs__msg__TouchSensorStateArray__Sequence__copy(
  const aimrt_msgs__msg__TouchSensorStateArray__Sequence * input,
  aimrt_msgs__msg__TouchSensorStateArray__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(aimrt_msgs__msg__TouchSensorStateArray);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    aimrt_msgs__msg__TouchSensorStateArray * data =
      (aimrt_msgs__msg__TouchSensorStateArray *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!aimrt_msgs__msg__TouchSensorStateArray__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          aimrt_msgs__msg__TouchSensorStateArray__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!aimrt_msgs__msg__TouchSensorStateArray__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
