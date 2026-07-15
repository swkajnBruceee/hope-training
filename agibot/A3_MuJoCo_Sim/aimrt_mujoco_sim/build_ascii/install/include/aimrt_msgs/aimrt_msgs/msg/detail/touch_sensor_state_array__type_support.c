// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from aimrt_msgs:msg/TouchSensorStateArray.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "aimrt_msgs/msg/detail/touch_sensor_state_array__rosidl_typesupport_introspection_c.h"
#include "aimrt_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "aimrt_msgs/msg/detail/touch_sensor_state_array__functions.h"
#include "aimrt_msgs/msg/detail/touch_sensor_state_array__struct.h"


// Include directives for member types
// Member `header`
#include "aimrt_msgs/msg/message_header.h"
// Member `header`
#include "aimrt_msgs/msg/detail/message_header__rosidl_typesupport_introspection_c.h"
// Member `names`
#include "rosidl_runtime_c/string_functions.h"
// Member `states`
#include "aimrt_msgs/msg/touch_sensor_state.h"
// Member `states`
#include "aimrt_msgs/msg/detail/touch_sensor_state__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  aimrt_msgs__msg__TouchSensorStateArray__init(message_memory);
}

void aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_fini_function(void * message_memory)
{
  aimrt_msgs__msg__TouchSensorStateArray__fini(message_memory);
}

size_t aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__size_function__TouchSensorStateArray__names(
  const void * untyped_member)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return member->size;
}

const void * aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_const_function__TouchSensorStateArray__names(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void * aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_function__TouchSensorStateArray__names(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__fetch_function__TouchSensorStateArray__names(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const rosidl_runtime_c__String * item =
    ((const rosidl_runtime_c__String *)
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_const_function__TouchSensorStateArray__names(untyped_member, index));
  rosidl_runtime_c__String * value =
    (rosidl_runtime_c__String *)(untyped_value);
  *value = *item;
}

void aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__assign_function__TouchSensorStateArray__names(
  void * untyped_member, size_t index, const void * untyped_value)
{
  rosidl_runtime_c__String * item =
    ((rosidl_runtime_c__String *)
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_function__TouchSensorStateArray__names(untyped_member, index));
  const rosidl_runtime_c__String * value =
    (const rosidl_runtime_c__String *)(untyped_value);
  *item = *value;
}

bool aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__resize_function__TouchSensorStateArray__names(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  rosidl_runtime_c__String__Sequence__fini(member);
  return rosidl_runtime_c__String__Sequence__init(member, size);
}

size_t aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__size_function__TouchSensorStateArray__states(
  const void * untyped_member)
{
  const aimrt_msgs__msg__TouchSensorState__Sequence * member =
    (const aimrt_msgs__msg__TouchSensorState__Sequence *)(untyped_member);
  return member->size;
}

const void * aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_const_function__TouchSensorStateArray__states(
  const void * untyped_member, size_t index)
{
  const aimrt_msgs__msg__TouchSensorState__Sequence * member =
    (const aimrt_msgs__msg__TouchSensorState__Sequence *)(untyped_member);
  return &member->data[index];
}

void * aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_function__TouchSensorStateArray__states(
  void * untyped_member, size_t index)
{
  aimrt_msgs__msg__TouchSensorState__Sequence * member =
    (aimrt_msgs__msg__TouchSensorState__Sequence *)(untyped_member);
  return &member->data[index];
}

void aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__fetch_function__TouchSensorStateArray__states(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const aimrt_msgs__msg__TouchSensorState * item =
    ((const aimrt_msgs__msg__TouchSensorState *)
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_const_function__TouchSensorStateArray__states(untyped_member, index));
  aimrt_msgs__msg__TouchSensorState * value =
    (aimrt_msgs__msg__TouchSensorState *)(untyped_value);
  *value = *item;
}

void aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__assign_function__TouchSensorStateArray__states(
  void * untyped_member, size_t index, const void * untyped_value)
{
  aimrt_msgs__msg__TouchSensorState * item =
    ((aimrt_msgs__msg__TouchSensorState *)
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_function__TouchSensorStateArray__states(untyped_member, index));
  const aimrt_msgs__msg__TouchSensorState * value =
    (const aimrt_msgs__msg__TouchSensorState *)(untyped_value);
  *item = *value;
}

bool aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__resize_function__TouchSensorStateArray__states(
  void * untyped_member, size_t size)
{
  aimrt_msgs__msg__TouchSensorState__Sequence * member =
    (aimrt_msgs__msg__TouchSensorState__Sequence *)(untyped_member);
  aimrt_msgs__msg__TouchSensorState__Sequence__fini(member);
  return aimrt_msgs__msg__TouchSensorState__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_message_member_array[3] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(aimrt_msgs__msg__TouchSensorStateArray, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "names",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(aimrt_msgs__msg__TouchSensorStateArray, names),  // bytes offset in struct
    NULL,  // default value
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__size_function__TouchSensorStateArray__names,  // size() function pointer
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_const_function__TouchSensorStateArray__names,  // get_const(index) function pointer
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_function__TouchSensorStateArray__names,  // get(index) function pointer
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__fetch_function__TouchSensorStateArray__names,  // fetch(index, &value) function pointer
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__assign_function__TouchSensorStateArray__names,  // assign(index, value) function pointer
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__resize_function__TouchSensorStateArray__names  // resize(index) function pointer
  },
  {
    "states",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(aimrt_msgs__msg__TouchSensorStateArray, states),  // bytes offset in struct
    NULL,  // default value
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__size_function__TouchSensorStateArray__states,  // size() function pointer
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_const_function__TouchSensorStateArray__states,  // get_const(index) function pointer
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__get_function__TouchSensorStateArray__states,  // get(index) function pointer
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__fetch_function__TouchSensorStateArray__states,  // fetch(index, &value) function pointer
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__assign_function__TouchSensorStateArray__states,  // assign(index, value) function pointer
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__resize_function__TouchSensorStateArray__states  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_message_members = {
  "aimrt_msgs__msg",  // message namespace
  "TouchSensorStateArray",  // message name
  3,  // number of fields
  sizeof(aimrt_msgs__msg__TouchSensorStateArray),
  aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_message_member_array,  // message members
  aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_init_function,  // function to initialize message memory (memory has to be allocated)
  aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_message_type_support_handle = {
  0,
  &aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_aimrt_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, aimrt_msgs, msg, TouchSensorStateArray)() {
  aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, aimrt_msgs, msg, MessageHeader)();
  aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, aimrt_msgs, msg, TouchSensorState)();
  if (!aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_message_type_support_handle.typesupport_identifier) {
    aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &aimrt_msgs__msg__TouchSensorStateArray__rosidl_typesupport_introspection_c__TouchSensorStateArray_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
