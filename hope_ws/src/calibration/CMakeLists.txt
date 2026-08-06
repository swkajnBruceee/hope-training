cmake_minimum_required(VERSION 3.8)
project(calibration)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hope_ros_bootstrap.cmake)
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(common REQUIRED)

add_library(calibration SHARED
  src/calibration.cpp
  src/split_calibration_csv.cpp
)

target_include_directories(calibration PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(calibration
  PUBLIC
    Eigen3
    common
)

add_executable(calibrate_ball_physics src/calibration_cli.cpp)
target_link_libraries(calibrate_ball_physics calibration)

add_executable(split_calibration_csv src/split_calibration_csv_cli.cpp)
target_link_libraries(split_calibration_csv calibration)

install(TARGETS calibration
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS calibrate_ball_physics split_calibration_csv
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_calibration test/test_calibration.cpp)
  target_link_libraries(test_calibration calibration)
  ament_add_gtest(test_split_calibration_csv test/test_split_calibration_csv.cpp)
  target_link_libraries(test_split_calibration_csv calibration)
endif()

ament_package()
