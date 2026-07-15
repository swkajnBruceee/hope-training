#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "irobot_events_executor::irobot_events_executor" for configuration "Release"
set_property(TARGET irobot_events_executor::irobot_events_executor APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(irobot_events_executor::irobot_events_executor PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libirobot_events_executor.so"
  IMPORTED_SONAME_RELEASE "libirobot_events_executor.so"
  )

list(APPEND _cmake_import_check_targets irobot_events_executor::irobot_events_executor )
list(APPEND _cmake_import_check_files_for_irobot_events_executor::irobot_events_executor "${_IMPORT_PREFIX}/lib/libirobot_events_executor.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
