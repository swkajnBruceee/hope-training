# Bootstrap ROS package lookup for bare `colcon build` runs in this workspace.
#
# When the shell has not sourced `/opt/ros/humble/setup.bash`, CMake cannot find
# `ament_cmake`. This helper prepends a known ROS installation prefix so each
# package can still resolve its ROS dependencies.

if(DEFINED HOPE_ROS_BOOTSTRAP_INCLUDED)
  return()
endif()
set(HOPE_ROS_BOOTSTRAP_INCLUDED TRUE)

set(_hope_ros_prefix "$ENV{HOPE_ROS_PREFIX}")
if(NOT _hope_ros_prefix)
  set(_hope_ros_prefix "/opt/ros/humble")
endif()

set(_hope_ament_config "${_hope_ros_prefix}/share/ament_cmake/cmake/ament_cmakeConfig.cmake")
if(EXISTS "${_hope_ament_config}")
  list(PREPEND CMAKE_PREFIX_PATH "${_hope_ros_prefix}")
  if(DEFINED ENV{AMENT_PREFIX_PATH} AND NOT "$ENV{AMENT_PREFIX_PATH}" STREQUAL "")
    set(ENV{AMENT_PREFIX_PATH} "${_hope_ros_prefix}:$ENV{AMENT_PREFIX_PATH}")
  else()
    set(ENV{AMENT_PREFIX_PATH} "${_hope_ros_prefix}")
  endif()

  file(GLOB _hope_ros_python_site_paths "${_hope_ros_prefix}/lib/python*/site-packages")
  file(GLOB _hope_ros_python_dist_paths "${_hope_ros_prefix}/local/lib/python*/dist-packages")
  set(_hope_ros_python_paths
    ${_hope_ros_python_site_paths}
    ${_hope_ros_python_dist_paths}
  )
  foreach(_hope_ros_python_path IN LISTS _hope_ros_python_paths)
    if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
      set(ENV{PYTHONPATH} "${_hope_ros_python_path}:$ENV{PYTHONPATH}")
    else()
      set(ENV{PYTHONPATH} "${_hope_ros_python_path}")
    endif()
  endforeach()

  set(_hope_ros_python_wrapper "${CMAKE_CURRENT_LIST_DIR}/hope_ros_python3")
  if(EXISTS "${_hope_ros_python_wrapper}")
    set(Python3_EXECUTABLE "${_hope_ros_python_wrapper}" CACHE FILEPATH
      "Workspace-local Python wrapper with ROS package paths" FORCE)
    set(PYTHON_EXECUTABLE "${_hope_ros_python_wrapper}" CACHE FILEPATH
      "Workspace-local Python wrapper with ROS package paths" FORCE)
  endif()
endif()

unset(_hope_ament_config)
unset(_hope_ros_python_dist_paths)
unset(_hope_ros_python_path)
unset(_hope_ros_python_paths)
unset(_hope_ros_python_wrapper)
unset(_hope_ros_python_site_paths)
unset(_hope_ros_prefix)
