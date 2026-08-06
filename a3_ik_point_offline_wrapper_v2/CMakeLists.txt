cmake_minimum_required(VERSION 3.16)
project(a3_ik_point_offline_wrapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MC_ROOT "" CACHE PATH "Path to the extracted mc source directory")
if(NOT MC_ROOT)
  message(FATAL_ERROR "Set -DMC_ROOT=/absolute/path/to/mc")
endif()
get_filename_component(MC_ROOT "${MC_ROOT}" ABSOLUTE)

foreach(required
    arm/include/ik_point_arm_source.hpp
    arm/src/ik_point_arm_source.cpp
    include/control_source.hpp
    include/fk.hpp
    src/fk.cpp
    include/math_utils.hpp
    include/policy_parameters.hpp
    include/xml.h
    include/robot_io/robot_io_backend.hpp)
  if(NOT EXISTS "${MC_ROOT}/${required}")
    message(FATAL_ERROR "Missing ${MC_ROOT}/${required}")
  endif()
endforeach()

# Eigen is header-only. This accepts standard distro install locations and a
# caller-provided EIGEN3_INCLUDE_DIR.
find_path(EIGEN3_INCLUDE_DIR Eigen/Core
  HINTS
    ${EIGEN3_INCLUDE_DIR}
    /usr/include/eigen3
    /usr/local/include/eigen3
  REQUIRED)

# Prefer the source archive's vendored yaml-cpp to keep this wrapper isolated.
set(_vendored_yaml "${MC_ROOT}/deps/fetchcontent/yaml-cpp_src")
if(EXISTS "${_vendored_yaml}/CMakeLists.txt")
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
  add_subdirectory("${_vendored_yaml}" "${CMAKE_BINARY_DIR}/yaml-cpp" EXCLUDE_FROM_ALL)
  set(YAML_CPP_TARGET yaml-cpp)
else()
  find_package(yaml-cpp REQUIRED)
  if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
  elseif(TARGET yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp)
  else()
    message(FATAL_ERROR "yaml-cpp package found but no CMake target is available")
  endif()
endif()

add_executable(a3_generate_strike_reference
  src/generate_strike_reference.cpp
  "${MC_ROOT}/arm/src/ik_point_arm_source.cpp"
  "${MC_ROOT}/src/fk.cpp")

target_include_directories(a3_generate_strike_reference PRIVATE
  "${MC_ROOT}/include"
  "${MC_ROOT}/arm/include"
  "${EIGEN3_INCLUDE_DIR}")

target_link_libraries(a3_generate_strike_reference PRIVATE ${YAML_CPP_TARGET})

target_compile_options(a3_generate_strike_reference PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

install(TARGETS a3_generate_strike_reference RUNTIME DESTINATION bin)
