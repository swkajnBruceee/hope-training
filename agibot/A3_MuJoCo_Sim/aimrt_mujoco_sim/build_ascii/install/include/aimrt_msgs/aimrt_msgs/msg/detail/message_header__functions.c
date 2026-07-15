// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from aimrt_msgs:msg/MessageHeader.idl
// generated code does not contain a copyright notice
#include "aimrt_msgs/msg/detail/message_header__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"
// Member `frame_id`
#include "rosidl_runtime_c/string_functions.h"

bool
aimrt_msgs__msg__MessageHeader__init(aimrt_msgs__msg__MessageHeader * msg)
{
  if (!msg) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    aimrt_msgs__msg__MessageHeader__fini(msg);
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__init(&msg->frame_id)) {
    aimrt_msgs__msg__MessageHeader__fini(msg);
    return false;
  }
  // sequence
  return true;
}

void
aimrt_msgs__msg__MessageHeader__fini(aimrt_msgs__msg__MessageHeader * msg)
{
  if (!msg) {
    return;
  }
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
  // frame_id
  rosidl_runtime_c__String__fini(&msg->frame_id);
  // sequence
}

bool
aimrt_msgs__msg__MessageHeader__are_equal(const aimrt_msgs__msg__MessageHeader * lhs, const aimrt_msgs__msg__MessageHeader * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->frame_id), &(rhs->frame_id)))
  {
    return false;
  }
  // sequence
  if (lhs->sequence != rhs->sequence) {
    return false;
  }
  return true;
}

bool
aimrt_msgs__msg__MessageHeader__copy(
  const aimrt_msgs__msg__MessageHeader * input,
  aimrt_msgs__msg__MessageHeader * output)
{
  if (!input || !output) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__copy(
      &(input->frame_id), &(output->frame_id)))
  {
    return false;
  }
  // sequence
  output->sequence = input->sequence;
  return true;
}

aimrt_msgs__msg__MessageHeader *
aimrt_msgs__msg__MessageHeader__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__MessageHeader * msg = (aimrt_msgs__msg__MessageHeader *)allocator.allocate(sizeof(aimrt_msgs__msg__MessageHeader), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(aimrt_msgs__msg__MessageHeader));
  bool success = aimrt_msgs__msg__MessageHeader__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
aimrt_msgs__msg__MessageHeader__destroy(aimrt_msgs__msg__MessageHeader * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    aimrt_msgs__msg__MessageHeader__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
aimrt_msgs__msg__MessageHeader__Sequence__init(aimrt_msgs__msg__MessageHeader__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__MessageHeader * data = NULL;

  if (size) {
    data = (aimrt_msgs__msg__MessageHeader *)allocator.zero_allocate(size, sizeof(aimrt_msgs__msg__MessageHeader), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = aimrt_msgs__msg__MessageHeader__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        aimrt_msgs__msg__MessageHeader__fini(&data[i - 1]);
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
aimrt_msgs__msg__MessageHeader__Sequence__fini(aimrt_msgs__msg__MessageHeader__Sequence * array)
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
      aimrt_msgs__msg__MessageHeader__fini(&array->data[i]);
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

aimrt_msgs__msg__MessageHeader__Sequence *
aimrt_msgs__msg__MessageHeader__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aimrt_msgs__msg__MessageHeader__Sequence * array = (aimrt_msgs__msg__MessageHeader__Sequence *)allocator.allocate(sizeof(aimrt_msgs__msg__MessageHeader__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = aimrt_msgs__msg__MessageHeader__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
aimrt_msgs__msg__MessageHeader__Sequence__destroy(aimrt_msgs__msg__MessageHeader__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    aimrt_msgs__msg__MessageHeader__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
aimrt_msgs__msg__MessageHeader__Sequence__are_equal(const aimrt_msgs__msg__MessageHeader__Sequence * lhs, const aimrt_msgs__msg__MessageHeader__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!aimrt_msgs__msg__MessageHeader__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
aimrt_msgs__msg__MessageHeader__Sequence__copy(
  const aimrt_msgs__msg__MessageHeader__Sequence * input,
  aimrt_msgs__msg__MessageHeader__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(aimrt_msgs__msg__MessageHeader);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    aimrt_msgs__msg__MessageHeader * data =
      (aimrt_msgs__msg__MessageHeader *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!aimrt_msgs__msg__MessageHeader__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          aimrt_msgs__msg__MessageHeader__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!aimrt_msgs__msg__MessageHeader__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
