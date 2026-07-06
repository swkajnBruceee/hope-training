cmake_minimum_required(VERSION 3.8)
project(bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hope_ros_bootstrap.cmake)
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(msgs REQUIRED)

add_executable(avatar_pro_vrpn_relay src/avatar_pro_vrpn_relay.cpp)
ament_target_dependencies(avatar_pro_vrpn_relay
  geometry_msgs
  rclcpp
  tf2_ros
)

add_executable(ball_truth_udp_bridge src/ball_truth_udp_bridge.cpp)
ament_target_dependencies(ball_truth_udp_bridge
  geometry_msgs
  rclcpp
)

add_executable(hit_state_udp_bridge src/hit_state_udp_bridge.cpp)
ament_target_dependencies(hit_state_udp_bridge
  geometry_msgs
  msgs
  rclcpp
)

add_executable(landing_candidates_udp_bridge src/landing_candidates_udp_bridge.cpp)
ament_target_dependencies(landing_candidates_udp_bridge
  msgs
  rclcpp
)

add_executable(landing_feedback_sim_node src/landing_feedback_sim_node.cpp)
target_link_libraries(landing_feedback_sim_node Eigen3::Eigen)
ament_target_dependencies(landing_feedback_sim_node
  msgs
  rclcpp
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  avatar_pro_vrpn_relay
  ball_truth_udp_bridge
  hit_state_udp_bridge
  landing_candidates_udp_bridge
  landing_feedback_sim_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  set(ament_cmake_xmllint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
