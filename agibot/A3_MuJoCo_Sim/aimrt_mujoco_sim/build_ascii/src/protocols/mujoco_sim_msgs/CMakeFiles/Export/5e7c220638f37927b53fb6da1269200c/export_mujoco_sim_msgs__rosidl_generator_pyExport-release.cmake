#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "mujoco_sim_msgs::mujoco_sim_msgs__rosidl_generator_py" for configuration "Release"
set_property(TARGET mujoco_sim_msgs::mujoco_sim_msgs__rosidl_generator_py APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mujoco_sim_msgs::mujoco_sim_msgs__rosidl_generator_py PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "mujoco_sim_msgs::mujoco_sim_msgs__rosidl_generator_c;Python::Python"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmujoco_sim_msgs__rosidl_generator_py.so"
  IMPORTED_SONAME_RELEASE "libmujoco_sim_msgs__rosidl_generator_py.so"
  )

list(APPEND _cmake_import_check_targets mujoco_sim_msgs::mujoco_sim_msgs__rosidl_generator_py )
list(APPEND _cmake_import_check_files_for_mujoco_sim_msgs::mujoco_sim_msgs__rosidl_generator_py "${_IMPORT_PREFIX}/lib/libmujoco_sim_msgs__rosidl_generator_py.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
