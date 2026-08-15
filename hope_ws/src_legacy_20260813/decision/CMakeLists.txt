cmake_minimum_required(VERSION 3.8)
project(decision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hope_ros_bootstrap.cmake)
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(common REQUIRED)
find_package(msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(trajectory REQUIRED)
find_package(solver REQUIRED)
find_package(diagnostic_msgs REQUIRED)

add_library(decision SHARED
  src/landing_decision/landing_decision.cpp
  src/target_selector.cpp
)

target_include_directories(decision PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(decision
  PUBLIC
    Eigen3
    common
    trajectory
    solver
)
target_link_libraries(decision PUBLIC solver::solver_lib)

add_executable(decision_node src/decision_node.cpp)
target_include_directories(decision_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(decision_node decision)
ament_target_dependencies(decision_node
  msgs
  rclcpp
  trajectory
  diagnostic_msgs
)

install(TARGETS decision
  EXPORT export_decision
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS decision_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_decision HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 common trajectory solver)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_target_selector test/test_target_selector.cpp)
  target_link_libraries(test_target_selector decision)
  ament_add_gtest(test_landing_decision test/test_landing_decision.cpp)
  target_link_libraries(test_landing_decision decision)
endif()

ament_package()
