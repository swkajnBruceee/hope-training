#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "aimrt::runtime::core" for configuration "Release"
set_property(TARGET aimrt::runtime::core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(aimrt::runtime::core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaimrt_runtime_core.a"
  )

list(APPEND _cmake_import_check_targets aimrt::runtime::core )
list(APPEND _cmake_import_check_files_for_aimrt::runtime::core "${_IMPORT_PREFIX}/lib/libaimrt_runtime_core.a" )

# Import target "aimrt::runtime::main" for configuration "Release"
set_property(TARGET aimrt::runtime::main APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(aimrt::runtime::main PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/aimrt_main"
  )

list(APPEND _cmake_import_check_targets aimrt::runtime::main )
list(APPEND _cmake_import_check_files_for_aimrt::runtime::main "${_IMPORT_PREFIX}/bin/aimrt_main" )

# Import target "aimrt::tools::protoc_plugin_cpp_gen_aimrt_cpp_rpc" for configuration "Release"
set_property(TARGET aimrt::tools::protoc_plugin_cpp_gen_aimrt_cpp_rpc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(aimrt::tools::protoc_plugin_cpp_gen_aimrt_cpp_rpc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/protoc_plugin_cpp_gen_aimrt_cpp_rpc"
  )

list(APPEND _cmake_import_check_targets aimrt::tools::protoc_plugin_cpp_gen_aimrt_cpp_rpc )
list(APPEND _cmake_import_check_files_for_aimrt::tools::protoc_plugin_cpp_gen_aimrt_cpp_rpc "${_IMPORT_PREFIX}/bin/protoc_plugin_cpp_gen_aimrt_cpp_rpc" )

# Import target "aimrt::protocols::common_pb_gencode" for configuration "Release"
set_property(TARGET aimrt::protocols::common_pb_gencode APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(aimrt::protocols::common_pb_gencode PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaimrt_protocols_common_pb_gencode.a"
  )

list(APPEND _cmake_import_check_targets aimrt::protocols::common_pb_gencode )
list(APPEND _cmake_import_check_files_for_aimrt::protocols::common_pb_gencode "${_IMPORT_PREFIX}/lib/libaimrt_protocols_common_pb_gencode.a" )

# Import target "aimrt::protocols::geometry_pb_gencode" for configuration "Release"
set_property(TARGET aimrt::protocols::geometry_pb_gencode APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(aimrt::protocols::geometry_pb_gencode PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaimrt_protocols_geometry_pb_gencode.a"
  )

list(APPEND _cmake_import_check_targets aimrt::protocols::geometry_pb_gencode )
list(APPEND _cmake_import_check_files_for_aimrt::protocols::geometry_pb_gencode "${_IMPORT_PREFIX}/lib/libaimrt_protocols_geometry_pb_gencode.a" )

# Import target "aimrt::protocols::sensor_pb_gencode" for configuration "Release"
set_property(TARGET aimrt::protocols::sensor_pb_gencode APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(aimrt::protocols::sensor_pb_gencode PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaimrt_protocols_sensor_pb_gencode.a"
  )

list(APPEND _cmake_import_check_targets aimrt::protocols::sensor_pb_gencode )
list(APPEND _cmake_import_check_files_for_aimrt::protocols::sensor_pb_gencode "${_IMPORT_PREFIX}/lib/libaimrt_protocols_sensor_pb_gencode.a" )

# Import target "aimrt::protocols::actuator_pb_gencode" for configuration "Release"
set_property(TARGET aimrt::protocols::actuator_pb_gencode APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(aimrt::protocols::actuator_pb_gencode PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaimrt_protocols_actuator_pb_gencode.a"
  )

list(APPEND _cmake_import_check_targets aimrt::protocols::actuator_pb_gencode )
list(APPEND _cmake_import_check_files_for_aimrt::protocols::actuator_pb_gencode "${_IMPORT_PREFIX}/lib/libaimrt_protocols_actuator_pb_gencode.a" )

# Import target "aimrt::protocols::time_manipulator_plugin_pb_gencode" for configuration "Release"
set_property(TARGET aimrt::protocols::time_manipulator_plugin_pb_gencode APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(aimrt::protocols::time_manipulator_plugin_pb_gencode PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaimrt_protocols_time_manipulator_plugin_pb_gencode.a"
  )

list(APPEND _cmake_import_check_targets aimrt::protocols::time_manipulator_plugin_pb_gencode )
list(APPEND _cmake_import_check_files_for_aimrt::protocols::time_manipulator_plugin_pb_gencode "${_IMPORT_PREFIX}/lib/libaimrt_protocols_time_manipulator_plugin_pb_gencode.a" )

# Import target "aimrt::protocols::time_manipulator_plugin_aimrt_rpc_gencode" for configuration "Release"
set_property(TARGET aimrt::protocols::time_manipulator_plugin_aimrt_rpc_gencode APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(aimrt::protocols::time_manipulator_plugin_aimrt_rpc_gencode PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaimrt_protocols_time_manipulator_plugin_aimrt_rpc_gencode.a"
  )

list(APPEND _cmake_import_check_targets aimrt::protocols::time_manipulator_plugin_aimrt_rpc_gencode )
list(APPEND _cmake_import_check_files_for_aimrt::protocols::time_manipulator_plugin_aimrt_rpc_gencode "${_IMPORT_PREFIX}/lib/libaimrt_protocols_time_manipulator_plugin_aimrt_rpc_gencode.a" )

# Import target "aimrt::protocols::ros2_plugin_proto_aimrt_rpc_gencode" for configuration "Release"
set_property(TARGET aimrt::protocols::ros2_plugin_proto_aimrt_rpc_gencode APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(aimrt::protocols::ros2_plugin_proto_aimrt_rpc_gencode PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaimrt_protocols_ros2_plugin_proto_aimrt_rpc_gencode.a"
  )

list(APPEND _cmake_import_check_targets aimrt::protocols::ros2_plugin_proto_aimrt_rpc_gencode )
list(APPEND _cmake_import_check_files_for_aimrt::protocols::ros2_plugin_proto_aimrt_rpc_gencode "${_IMPORT_PREFIX}/lib/libaimrt_protocols_ros2_plugin_proto_aimrt_rpc_gencode.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
