cmake_minimum_required(VERSION 3.8)
project(trajectory)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hope_ros_bootstrap.cmake)
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(common REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(msgs REQUIRED)
find_package(rclcpp REQUIRED)

add_library(trajectory SHARED
  src/ball_state_estimator.cpp
  src/ball_trajectory_predictor.cpp
  src/incoming_fitter.cpp
)

target_include_directories(trajectory PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(trajectory
  PUBLIC
    Eigen3
    common
)
target_link_libraries(trajectory PUBLIC Eigen3::Eigen common::common)

add_executable(trajectory_overlay_udp_node src/trajectory_overlay_udp_node.cpp)
target_include_directories(trajectory_overlay_udp_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(trajectory_overlay_udp_node trajectory)
ament_target_dependencies(trajectory_overlay_udp_node
  geometry_msgs
  rclcpp
)

add_executable(strike_prediction_node src/strike_prediction_node.cpp)
target_include_directories(strike_prediction_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(strike_prediction_node trajectory)
ament_target_dependencies(strike_prediction_node
  geometry_msgs
  msgs
  rclcpp
)

install(TARGETS trajectory
  EXPORT export_trajectory
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS trajectory_overlay_udp_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS strike_prediction_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_trajectory HAS_LIBRARY_TARGET)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_ball_state_estimator test/test_ball_state_estimator.cpp)
  target_link_libraries(test_ball_state_estimator trajectory)
  ament_add_gtest(test_ball_trajectory_predictor test/test_ball_trajectory_predictor.cpp)
  target_link_libraries(test_ball_trajectory_predictor trajectory)
endif()

ament_package()
