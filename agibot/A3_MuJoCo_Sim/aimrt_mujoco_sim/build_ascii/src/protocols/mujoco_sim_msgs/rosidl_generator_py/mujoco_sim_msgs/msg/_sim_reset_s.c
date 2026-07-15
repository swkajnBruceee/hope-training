// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from mujoco_sim_msgs:msg/SimReset.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "mujoco_sim_msgs/msg/detail/sim_reset__struct.h"
#include "mujoco_sim_msgs/msg/detail/sim_reset__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__header__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool geometry_msgs__msg__pose__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * geometry_msgs__msg__pose__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool geometry_msgs__msg__twist__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * geometry_msgs__msg__twist__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool sensor_msgs__msg__joint_state__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * sensor_msgs__msg__joint_state__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool mujoco_sim_msgs__msg__sim_reset__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[40];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("mujoco_sim_msgs.msg._sim_reset.SimReset", full_classname_dest, 39) == 0);
  }
  mujoco_sim_msgs__msg__SimReset * ros_message = _ros_message;
  {  // header
    PyObject * field = PyObject_GetAttrString(_pymsg, "header");
    if (!field) {
      return false;
    }
    if (!std_msgs__msg__header__convert_from_py(field, &ros_message->header)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // mode
    PyObject * field = PyObject_GetAttrString(_pymsg, "mode");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->mode = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // keyframe_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "keyframe_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->keyframe_id = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // set_base
    PyObject * field = PyObject_GetAttrString(_pymsg, "set_base");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->set_base = (Py_True == field);
    Py_DECREF(field);
  }
  {  // pelvis_pose
    PyObject * field = PyObject_GetAttrString(_pymsg, "pelvis_pose");
    if (!field) {
      return false;
    }
    if (!geometry_msgs__msg__pose__convert_from_py(field, &ros_message->pelvis_pose)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // set_base_twist
    PyObject * field = PyObject_GetAttrString(_pymsg, "set_base_twist");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->set_base_twist = (Py_True == field);
    Py_DECREF(field);
  }
  {  // pelvis_twist
    PyObject * field = PyObject_GetAttrString(_pymsg, "pelvis_twist");
    if (!field) {
      return false;
    }
    if (!geometry_msgs__msg__twist__convert_from_py(field, &ros_message->pelvis_twist)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // set_joints
    PyObject * field = PyObject_GetAttrString(_pymsg, "set_joints");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->set_joints = (Py_True == field);
    Py_DECREF(field);
  }
  {  // joint_state
    PyObject * field = PyObject_GetAttrString(_pymsg, "joint_state");
    if (!field) {
      return false;
    }
    if (!sensor_msgs__msg__joint_state__convert_from_py(field, &ros_message->joint_state)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // zero_all_velocities
    PyObject * field = PyObject_GetAttrString(_pymsg, "zero_all_velocities");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->zero_all_velocities = (Py_True == field);
    Py_DECREF(field);
  }
  {  // clear_ctrl
    PyObject * field = PyObject_GetAttrString(_pymsg, "clear_ctrl");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->clear_ctrl = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * mujoco_sim_msgs__msg__sim_reset__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of SimReset */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("mujoco_sim_msgs.msg._sim_reset");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "SimReset");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  mujoco_sim_msgs__msg__SimReset * ros_message = (mujoco_sim_msgs__msg__SimReset *)raw_ros_message;
  {  // header
    PyObject * field = NULL;
    field = std_msgs__msg__header__convert_to_py(&ros_message->header);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "header", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // mode
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->mode);
    {
      int rc = PyObject_SetAttrString(_pymessage, "mode", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // keyframe_id
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->keyframe_id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "keyframe_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // set_base
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->set_base ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "set_base", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pelvis_pose
    PyObject * field = NULL;
    field = geometry_msgs__msg__pose__convert_to_py(&ros_message->pelvis_pose);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "pelvis_pose", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // set_base_twist
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->set_base_twist ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "set_base_twist", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pelvis_twist
    PyObject * field = NULL;
    field = geometry_msgs__msg__twist__convert_to_py(&ros_message->pelvis_twist);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "pelvis_twist", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // set_joints
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->set_joints ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "set_joints", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // joint_state
    PyObject * field = NULL;
    field = sensor_msgs__msg__joint_state__convert_to_py(&ros_message->joint_state);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "joint_state", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // zero_all_velocities
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->zero_all_velocities ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "zero_all_velocities", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // clear_ctrl
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->clear_ctrl ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "clear_ctrl", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
