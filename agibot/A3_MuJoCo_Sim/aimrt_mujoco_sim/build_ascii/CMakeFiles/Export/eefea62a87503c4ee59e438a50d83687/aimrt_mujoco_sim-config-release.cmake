#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "aimrt_mujoco_sim::mujoco_sim_pkg" for configuration "Release"
set_property(TARGET aimrt_mujoco_sim::mujoco_sim_pkg APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(aimrt_mujoco_sim::mujoco_sim_pkg PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmujoco_sim_pkg.so"
  IMPORTED_SONAME_RELEASE "libmujoco_sim_pkg.so"
  )

list(APPEND _cmake_import_check_targets aimrt_mujoco_sim::mujoco_sim_pkg )
list(APPEND _cmake_import_check_files_for_aimrt_mujoco_sim::mujoco_sim_pkg "${_IMPORT_PREFIX}/lib/libmujoco_sim_pkg.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
