cmake_minimum_required(VERSION 3.8)
project(common)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hope_ros_bootstrap.cmake)
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

add_library(common SHARED
  src/quaternion_utils.cpp
)

target_include_directories(common PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(common
  PUBLIC
    Eigen3
)
target_link_libraries(common PUBLIC Eigen3::Eigen)

install(TARGETS common
  EXPORT export_common
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

# Export targets to downstream packages.
ament_export_targets(export_common HAS_LIBRARY_TARGET)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_constants test/test_constants.cpp)
  target_link_libraries(test_constants common)
  ament_add_gtest(test_quaternion_utils test/test_quaternion_utils.cpp)
  target_link_libraries(test_quaternion_utils common)
endif()

ament_package()
