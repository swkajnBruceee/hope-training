cmake_minimum_required(VERSION 3.8)
project(solver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hope_ros_bootstrap.cmake)
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(common REQUIRED)
find_package(trajectory REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(msgs REQUIRED)

add_library(solver_lib SHARED
  src/hit_plan_solver.cpp
  src/landing_error_corrector.cpp
  src/racket_target_solver.cpp
  src/solver_pipeline.cpp
)

target_include_directories(solver_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(solver_lib
  PUBLIC
    Eigen3
    common
    trajectory
    rclcpp
    geometry_msgs
    std_msgs
    diagnostic_msgs
    msgs
)

add_executable(solver_node src/solver_node.cpp)
target_include_directories(solver_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(solver_node PUBLIC solver_lib)
ament_target_dependencies(solver_node
  PUBLIC
    rclcpp
    geometry_msgs
    std_msgs
    diagnostic_msgs
    msgs
    common
    trajectory
)

install(TARGETS solver_lib
  EXPORT export_solver
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS solver_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_export_targets(export_solver HAS_LIBRARY_TARGET)
ament_export_dependencies(
  Eigen3
  common
  trajectory
  rclcpp
  geometry_msgs
  std_msgs
  diagnostic_msgs
  msgs
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_racket_target_solver test/test_racket_target_solver.cpp)
  target_link_libraries(test_racket_target_solver solver_lib)
  ament_add_gtest(test_hit_plan_solver test/test_hit_plan_solver.cpp)
  target_link_libraries(test_hit_plan_solver solver_lib)
  ament_add_gtest(test_landing_error_corrector test/test_landing_error_corrector.cpp)
  target_link_libraries(test_landing_error_corrector solver_lib)
endif()

ament_package()
