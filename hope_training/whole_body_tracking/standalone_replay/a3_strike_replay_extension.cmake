# Injected with -DCMAKE_PROJECT_INCLUDE=<this file>; it leaves vendor sources
# untouched while building an executable against the exact same backend target.
set(A3_STRIKE_REPLAY_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "A3 strike replay extension source")

function(a3_strike_add_robotio_replay)
  if(TARGET a3_strike_robotio_replay)
    return()
  endif()
  if(NOT TARGET a3_deploy_shared)
    message(FATAL_ERROR "a3_deploy_shared is unavailable; this extension must be used with a3_deploy_example")
  endif()
  add_executable(a3_strike_robotio_replay
    "${A3_STRIKE_REPLAY_SOURCE_DIR}/a3_strike_robotio_replay.cpp")
  target_sources(a3_strike_robotio_replay PRIVATE
    "${PROJECT_SOURCE_DIR}/src/a3/a3_deploy_onnx_ref/src/robot_io/backend_factory.cpp"
    "${PROJECT_SOURCE_DIR}/src/a3/a3_deploy_onnx_ref/src/robot_io/layouts.cpp"
    "${PROJECT_SOURCE_DIR}/src/a3/a3_deploy_onnx_ref/src/robot_io/a3_layout_extra.cpp"
    "${PROJECT_SOURCE_DIR}/src/a3/a3_deploy_onnx_ref/src/robot_io/a3_aimrt_backend.cpp")
  target_include_directories(a3_strike_robotio_replay PRIVATE
    "${PROJECT_SOURCE_DIR}/src/a3/a3_deploy_onnx_ref/include"
    "${PROJECT_SOURCE_DIR}/src/a3/a3_deploy_onnx_ref/src")
  target_link_libraries(a3_strike_robotio_replay PRIVATE
    a3_deploy_shared unitree_sdk2 ZLIB::ZLIB onnxruntime::onnxruntime)
  if(TARGET aimrt::runtime::core)
    target_compile_definitions(a3_strike_robotio_replay PRIVATE ENABLE_A3_AIMRT_BACKEND=1)
    target_link_libraries(a3_strike_robotio_replay PRIVATE
      aimrt::interface::aimrt_module_cpp_interface
      aimrt::interface::aimrt_module_protobuf_interface
      aimrt::interface::aimrt_module_ros2_interface
      aimrt::runtime::core)
  endif()
  set_target_properties(a3_strike_robotio_replay PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${GS_RUNTIME_OUTPUT_DIR}"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib"
    INSTALL_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib")
endfunction()

# CMAKE_PROJECT_INCLUDE runs during project(), before the vendor's targets and
# backend variables exist. Defer target creation until the root list is done.
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL a3_strike_add_robotio_replay)
