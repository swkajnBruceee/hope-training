# Install script for directory: /home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/bruce/hopett_sim/build_ascii/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/iceoryx_hoofs" TYPE FILE FILES "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/LICENSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "bin" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/bruce/hopett_sim/build_ascii/libiceoryx_hoofs.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/iceoryx/v" TYPE DIRECTORY FILES
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/buffer/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/cli/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/concurrent/buffer/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/concurrent/sync/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/container/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/design/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/filesystem/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/functional/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/memory/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/primitives/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/reporting/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/time/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/utility/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/vocabulary/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/posix/auth/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/posix/design/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/posix/ipc/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/posix/filesystem/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/posix/sync/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/posix/time/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/posix/utility/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/posix/vocabulary/include/"
    "/home/bruce/hopett_sim/_deps/iceoryx-src/iceoryx_hoofs/legacy/include/"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/iceoryx_hoofs" TYPE FILE FILES
    "/home/bruce/hopett_sim/build_ascii/hoofs/iceoryx_hoofsConfigVersion.cmake"
    "/home/bruce/hopett_sim/build_ascii/hoofs/iceoryx_hoofsConfig.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/iceoryx_hoofs/iceoryx_hoofsTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/iceoryx_hoofs/iceoryx_hoofsTargets.cmake"
         "/home/bruce/hopett_sim/build_ascii/hoofs/CMakeFiles/Export/2df1df3350d90c929624816c3cc2d98f/iceoryx_hoofsTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/iceoryx_hoofs/iceoryx_hoofsTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/iceoryx_hoofs/iceoryx_hoofsTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/iceoryx_hoofs" TYPE FILE FILES "/home/bruce/hopett_sim/build_ascii/hoofs/CMakeFiles/Export/2df1df3350d90c929624816c3cc2d98f/iceoryx_hoofsTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/iceoryx_hoofs" TYPE FILE FILES "/home/bruce/hopett_sim/build_ascii/hoofs/CMakeFiles/Export/2df1df3350d90c929624816c3cc2d98f/iceoryx_hoofsTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/iceoryx/v/iox" TYPE FILE FILES "/home/bruce/hopett_sim/build_ascii/generated/iceoryx_hoofs/include/iox/iceoryx_hoofs_deployment.hpp")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/bruce/hopett_sim/build_ascii/hoofs/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
