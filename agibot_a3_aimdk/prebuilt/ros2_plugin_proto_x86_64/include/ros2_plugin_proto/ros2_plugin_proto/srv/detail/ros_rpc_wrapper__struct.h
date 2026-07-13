// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ros2_plugin_proto:srv/RosRpcWrapper.idl
// generated code does not contain a copyright notice

#ifndef ROS2_PLUGIN_PROTO__SRV__DETAIL__ROS_RPC_WRAPPER__STRUCT_H_
#define ROS2_PLUGIN_PROTO__SRV__DETAIL__ROS_RPC_WRAPPER__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'serialization_type'
// Member 'context'
#include "rosidl_runtime_c/string.h"
// Member 'data'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in srv/RosRpcWrapper in the package ros2_plugin_proto.
typedef struct ros2_plugin_proto__srv__RosRpcWrapper_Request
{
  rosidl_runtime_c__String serialization_type;
  rosidl_runtime_c__String__Sequence context;
  rosidl_runtime_c__octet__Sequence data;
} ros2_plugin_proto__srv__RosRpcWrapper_Request;

// Struct for a sequence of ros2_plugin_proto__srv__RosRpcWrapper_Request.
typedef struct ros2_plugin_proto__srv__RosRpcWrapper_Request__Sequence
{
  ros2_plugin_proto__srv__RosRpcWrapper_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ros2_plugin_proto__srv__RosRpcWrapper_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'serialization_type'
// already included above
// #include "rosidl_runtime_c/string.h"
// Member 'data'
// already included above
// #include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in srv/RosRpcWrapper in the package ros2_plugin_proto.
typedef struct ros2_plugin_proto__srv__RosRpcWrapper_Response
{
  int64_t code;
  rosidl_runtime_c__String serialization_type;
  rosidl_runtime_c__octet__Sequence data;
} ros2_plugin_proto__srv__RosRpcWrapper_Response;

// Struct for a sequence of ros2_plugin_proto__srv__RosRpcWrapper_Response.
typedef struct ros2_plugin_proto__srv__RosRpcWrapper_Response__Sequence
{
  ros2_plugin_proto__srv__RosRpcWrapper_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ros2_plugin_proto__srv__RosRpcWrapper_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ROS2_PLUGIN_PROTO__SRV__DETAIL__ROS_RPC_WRAPPER__STRUCT_H_
