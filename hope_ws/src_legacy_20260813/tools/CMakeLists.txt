cmake_minimum_required(VERSION 3.8)
project(tools)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hope_ros_bootstrap.cmake)
find_package(ament_cmake REQUIRED)

add_library(tools_lib SHARED src/bag_to_csv.cpp)
target_include_directories(tools_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
install(TARGETS tools_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

add_executable(bag_to_csv src/bag_to_csv_cli.cpp)
target_link_libraries(bag_to_csv tools_lib)
install(TARGETS bag_to_csv RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_package()
