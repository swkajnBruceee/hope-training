cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(a3_deploy VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")

option(ENABLE_A3_AIMRT_BACKEND "Build the AimRT/Iceoryx robot backend" ON)
option(ENABLE_A3_ROS_MSGS "Build vendored A3 ROS messages" ON)
option(ENABLE_RKNN_INFERENCE "Build the optional RKNN backend" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(BUILD_MC_TESTS "Build the large offline test suite" OFF)
set(GS_PACKAGE_ARCH_NAME rockchip CACHE STRING "Package architecture")
set(GS_RUNTIME_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dist/${GS_PACKAGE_ARCH_NAME}" CACHE PATH "Runtime output directory")

add_library(a3_deploy INTERFACE)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_ort_tar "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime/onnxruntime_gpu_1.19.2_aarch64.tar.gz")
  set(_ort_root "${CMAKE_CURRENT_BINARY_DIR}/_deps/onnxruntime_aarch64/onnxruntime_gpu_1.19.2")
  if(EXISTS "${_ort_tar}")
    if(NOT EXISTS "${_ort_root}/lib/cmake/onnxruntime/onnxruntimeConfig.cmake")
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/_deps/onnxruntime_aarch64")
      execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${_ort_tar}"
                      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/_deps/onnxruntime_aarch64"
                      RESULT_VARIABLE _ort_rc)
      if(NOT _ort_rc EQUAL 0)
        message(FATAL_ERROR "Cannot extract packaged ONNX Runtime")
      endif()
    endif()
    list(PREPEND CMAKE_PREFIX_PATH "${_ort_root}")
    set(onnxruntime_DIR "${_ort_root}/lib/cmake/onnxruntime" CACHE PATH "Packaged ONNX Runtime" FORCE)
    set(GS_BUNDLED_ONNXRUNTIME_LIB_DIR "${_ort_root}/lib" CACHE INTERNAL "Packaged ONNX Runtime libs")
  endif()
endif()
find_package(onnxruntime REQUIRED)
find_package(yaml-cpp REQUIRED)

if(ENABLE_RKNN_INFERENCE)
  set(_rknn_root "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rknn_runtime/2.3.2")
  add_library(rknnrt::rknnrt SHARED IMPORTED GLOBAL)
  set_target_properties(rknnrt::rknnrt PROPERTIES
    IMPORTED_LOCATION "${_rknn_root}/lib/aarch64/librknnrt.so"
    INTERFACE_INCLUDE_DIRECTORIES "${_rknn_root}/include")
endif()

set(BUILD_EXAMPLES OFF CACHE BOOL "Disable Unitree SDK examples" FORCE)
add_subdirectory(thirdparty/unitree_sdk2)
target_include_directories(a3_deploy INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/unitree_sdk2/include")
target_link_libraries(a3_deploy INTERFACE unitree_sdk2)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/a3_deploy_onnx_ref.cmake")
