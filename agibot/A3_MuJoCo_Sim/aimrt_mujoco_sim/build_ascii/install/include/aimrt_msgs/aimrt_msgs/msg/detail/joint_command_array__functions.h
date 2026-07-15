// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from aimrt_msgs:msg/JointCommandArray.idl
// generated code does not contain a copyright notice

#ifndef AIMRT_MSGS__MSG__DETAIL__JOINT_COMMAND_ARRAY__FUNCTIONS_H_
#define AIMRT_MSGS__MSG__DETAIL__JOINT_COMMAND_ARRAY__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "aimrt_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "aimrt_msgs/msg/detail/joint_command_array__struct.h"

/// Initialize msg/JointCommandArray message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * aimrt_msgs__msg__JointCommandArray
 * )) before or use
 * aimrt_msgs__msg__JointCommandArray__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
bool
aimrt_msgs__msg__JointCommandArray__init(aimrt_msgs__msg__JointCommandArray * msg);

/// Finalize msg/JointCommandArray message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
void
aimrt_msgs__msg__JointCommandArray__fini(aimrt_msgs__msg__JointCommandArray * msg);

/// Create msg/JointCommandArray message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * aimrt_msgs__msg__JointCommandArray__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
aimrt_msgs__msg__JointCommandArray *
aimrt_msgs__msg__JointCommandArray__create();

/// Destroy msg/JointCommandArray message.
/**
 * It calls
 * aimrt_msgs__msg__JointCommandArray__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
void
aimrt_msgs__msg__JointCommandArray__destroy(aimrt_msgs__msg__JointCommandArray * msg);

/// Check for msg/JointCommandArray message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
bool
aimrt_msgs__msg__JointCommandArray__are_equal(const aimrt_msgs__msg__JointCommandArray * lhs, const aimrt_msgs__msg__JointCommandArray * rhs);

/// Copy a msg/JointCommandArray message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
bool
aimrt_msgs__msg__JointCommandArray__copy(
  const aimrt_msgs__msg__JointCommandArray * input,
  aimrt_msgs__msg__JointCommandArray * output);

/// Initialize array of msg/JointCommandArray messages.
/**
 * It allocates the memory for the number of elements and calls
 * aimrt_msgs__msg__JointCommandArray__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
bool
aimrt_msgs__msg__JointCommandArray__Sequence__init(aimrt_msgs__msg__JointCommandArray__Sequence * array, size_t size);

/// Finalize array of msg/JointCommandArray messages.
/**
 * It calls
 * aimrt_msgs__msg__JointCommandArray__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
void
aimrt_msgs__msg__JointCommandArray__Sequence__fini(aimrt_msgs__msg__JointCommandArray__Sequence * array);

/// Create array of msg/JointCommandArray messages.
/**
 * It allocates the memory for the array and calls
 * aimrt_msgs__msg__JointCommandArray__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
aimrt_msgs__msg__JointCommandArray__Sequence *
aimrt_msgs__msg__JointCommandArray__Sequence__create(size_t size);

/// Destroy array of msg/JointCommandArray messages.
/**
 * It calls
 * aimrt_msgs__msg__JointCommandArray__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
void
aimrt_msgs__msg__JointCommandArray__Sequence__destroy(aimrt_msgs__msg__JointCommandArray__Sequence * array);

/// Check for msg/JointCommandArray message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
bool
aimrt_msgs__msg__JointCommandArray__Sequence__are_equal(const aimrt_msgs__msg__JointCommandArray__Sequence * lhs, const aimrt_msgs__msg__JointCommandArray__Sequence * rhs);

/// Copy an array of msg/JointCommandArray messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_aimrt_msgs
bool
aimrt_msgs__msg__JointCommandArray__Sequence__copy(
  const aimrt_msgs__msg__JointCommandArray__Sequence * input,
  aimrt_msgs__msg__JointCommandArray__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // AIMRT_MSGS__MSG__DETAIL__JOINT_COMMAND_ARRAY__FUNCTIONS_H_
