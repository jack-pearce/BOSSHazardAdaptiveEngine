cmake_minimum_required(VERSION 3.10)
project(BOSSHazardAdaptiveEngine)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

# Configure CCache if available
find_program(CCACHE_EXECUTABLE ccache)
mark_as_advanced(CCACHE_EXECUTABLE)
if(CCACHE_EXECUTABLE)
  foreach(LANG C CXX)
    if(NOT DEFINED CMAKE_${LANG}_COMPILER_LAUNCHER AND NOT CMAKE_${LANG}_COMPILER MATCHES ".*/ccache")
      message(STATUS "Enabling ccache for ${LANG}")
      set(CMAKE_${LANG}_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE} CACHE STRING "")
    endif()
  endforeach()
endif()

#################### Requirements for hazard-adaptive engine ####################

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Unsupported operating system: ${CMAKE_SYSTEM_NAME}")
endif ()

file(READ "/proc/sys/kernel/perf_event_paranoid" PARANOID_LEVEL_CONTENT)
if (NOT PARANOID_LEVEL_CONTENT)
    message(FATAL_ERROR "Failed to read /proc/sys/kernel/perf_event_paranoid")
endif()

if (NOT PARANOID_LEVEL_CONTENT LESS_EQUAL 1)
    message(FATAL_ERROR "Paranoid level is ${PARANOID_LEVEL_CONTENT}, it must be greater than 1.")
endif()

message(STATUS "Paranoid level: ${PARANOID_LEVEL_CONTENT}")


############################## Custom build modes ###############################

# Default flags for all build types
set(COMMON_FLAGS "-Wall -Wextra -Wpedantic -Werror -march=native")

# Debug build type
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -Og -g"
    CACHE STRING
    "Flags used by the C++ compiler during Debug builds."
    FORCE)
set(CMAKE_C_FLAGS_DEBUG "${COMMON_FLAGS} -Og -g"
    CACHE STRING
    "Flags used by the C compiler during Debug builds."
    FORCE)
set(CMAKE_EXE_LINKER_FLAGS_DEBUG
    ${CMAKE_EXE_LINKER_FLAGS_DEBUG} CACHE STRING
    "Flags used for linking binaries during Debug builds."
    FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG
    ${CMAKE_SHARED_LINKER_FLAGS_DEBUG} CACHE STRING
    "Flags used by the shared libraries linker during Debug builds."
    FORCE)

# Release build type
set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3"
    CACHE STRING
    "Flags used by the C++ compiler during Release builds."
    FORCE)
set(CMAKE_C_FLAGS_RELEASE "${COMMON_FLAGS} -O3"
    CACHE STRING
    "Flags used by the C compiler during Release builds."
    FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELEASE
    ${CMAKE_EXE_LINKER_FLAGS_RELEASE} CACHE STRING
    "Flags used for linking binaries during Release builds."
    FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE
    ${CMAKE_SHARED_LINKER_FLAGS_RELEASE} CACHE STRING
    "Flags used by the shared libraries linker during Release builds."
    FORCE)

# Sanitize build type
set(CMAKE_CXX_FLAGS_SANITIZE "${COMMON_FLAGS} -fno-omit-frame-pointer -fsanitize=address,signed-integer-overflow,null,alignment,bounds,function,return,vla-bound -O0 -g"
    CACHE STRING
    "Flags used by the C++ compiler during Sanitize builds."
    FORCE)
set(CMAKE_C_FLAGS_SANITIZE "${COMMON_FLAGS} -fno-omit-frame-pointer -fsanitize=address,signed-integer-overflow,null,alignment,bounds,function,return,vla-bound -O0 -g"
    CACHE STRING
    "Flags used by the C compiler during Sanitize builds."
    FORCE)
set(CMAKE_EXE_LINKER_FLAGS_SANITIZE
    ${CMAKE_EXE_LINKER_FLAGS_DEBUG} CACHE STRING
    "Flags used for linking binaries during Sanitize builds."
    FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_SANITIZE
    ${CMAKE_SHARED_LINKER_FLAGS_DEBUG} CACHE STRING
    "Flags used by the shared libraries linker during Sanitize builds."
    FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_DEBUG		    CMAKE_EXE_LINKER_FLAGS_DEBUG
    CMAKE_C_FLAGS_DEBUG		        CMAKE_SHARED_LINKER_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE		    CMAKE_EXE_LINKER_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELEASE		    CMAKE_SHARED_LINKER_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_SANITIZE		CMAKE_EXE_LINKER_FLAGS_SANITIZE
    CMAKE_C_FLAGS_SANITIZE		    CMAKE_SHARED_LINKER_FLAGS_SANITIZE
  )

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
  "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Sanitize."
  FORCE)

############################### External Projects ###############################

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE) # make sure all dependent libraries are compiled with that option

if(MSVC)
    set(BOSS_LINK_LIBRARY_PREFIX ${CMAKE_LINK_LIBRARY_PREFIX})
    set(BOSS_LINK_LIBRARY_SUFFIX ${CMAKE_LINK_LIBRARY_SUFFIX})
else()
    set(BOSS_LINK_LIBRARY_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
    set(BOSS_LINK_LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
endif(MSVC)

include(ExternalProject)

# Core BOSS
if(NOT DEFINED BOSS_SOURCE_REPOSITORY)
    if(NOT GITHUB_TOKEN)
        message( FATAL_ERROR "You need to pass -DGITHUB_TOKEN=... to pass a github token to fetch the boss release or -DBOSS_SOURCE_REPOSITORY= for a non-github clone" )
    endif()
    set(BOSS_SOURCE_REPOSITORY https://little-big-h:${GITHUB_TOKEN}@github.com/symbol-store/BOSS.git)
endif()
if(NOT DEFINED BOSS_SOURCE_BRANCH)
    set(BOSS_SOURCE_BRANCH main)
endif()
message("getting boss sources from ${BOSS_SOURCE_REPOSITORY}, branch ${BOSS_SOURCE_BRANCH}")
ExternalProject_Add(BOSS
        GIT_REPOSITORY ${BOSS_SOURCE_REPOSITORY}
        GIT_TAG ${BOSS_SOURCE_BRANCH}
        GIT_SHALLOW ON
        CMAKE_ARGS -DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE} -DCMAKE_INSTALL_PREFIX=${BOSSHazardAdaptiveEngine_BINARY_DIR}/deps $<IF:$<CONFIG:>,,-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}>
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        BUILD_BYPRODUCTS ${BOSSHazardAdaptiveEngine_BINARY_DIR}/deps/lib/${BOSS_LINK_LIBRARY_PREFIX}BOSS${BOSS_LINK_LIBRARY_SUFFIX}
)

# PAPI
ExternalProject_Add(
        papi
        PREFIX ${BOSSHazardAdaptiveEngine_BINARY_DIR}/papi
        URL https://icl.utk.edu/projects/papi/downloads/papi-7.0.1.tar.gz
        URL_MD5 14bb2b09dab28232911f929ef4e4b98b
        SOURCE_DIR ${BOSSHazardAdaptiveEngine_BINARY_DIR}/papi/source
        BINARY_DIR ${BOSSHazardAdaptiveEngine_BINARY_DIR}/papi/source/src
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E chdir ${BOSSHazardAdaptiveEngine_BINARY_DIR}/papi/source/src ${BOSSHazardAdaptiveEngine_BINARY_DIR}/papi/source/src/configure --prefix=${BOSSHazardAdaptiveEngine_BINARY_DIR}/papi/install
        BUILD_COMMAND make
        INSTALL_COMMAND make install
)

# Tessil robin-map
ExternalProject_Add(
        robin-map
        GIT_REPOSITORY https://github.com/Tessil/robin-map.git
        GIT_TAG d37a410
        PREFIX ${BOSSHazardAdaptiveEngine_BINARY_DIR}/robin-map
        INSTALL_COMMAND ""
)

# JsonCpp
ExternalProject_Add(
        jsoncpp
        GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
        GIT_TAG 5defb4e
        PREFIX ${BOSSHazardAdaptiveEngine_BINARY_DIR}/jsoncpp
        INSTALL_COMMAND ""
)


#################################### Targets ####################################

if(MSVC)
  # not making any difference on Windows
  # and helps Visual Studio to parse it correctly as a target
  set(LibraryType SHARED)
else()
  set(LibraryType MODULE)
endif(MSVC)

if(MSVC)
    set(pluginInstallDir bin)
else()
    set(pluginInstallDir lib)
endif(MSVC)

set(ImplementationFiles
        Source/HazardAdaptiveEngine.cpp
        Source/operators/select.cpp
        Source/utilities/papiWrapper.cpp
        Source/utilities/systemInformation.cpp
        Source/constants/machineConstants.cpp
)

# Plugin
add_library(BOSSHazardAdaptiveEngine MODULE ${ImplementationFiles})
target_include_directories(BOSSHazardAdaptiveEngine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Source)

set_property(TARGET BOSSHazardAdaptiveEngine PROPERTY CXX_STANDARD 20) ## the core is c++ 17 but the engines may want to use 20
target_include_directories(BOSSHazardAdaptiveEngine SYSTEM PUBLIC ${BOSSHazardAdaptiveEngine_BINARY_DIR}/deps/include)
target_link_directories(BOSSHazardAdaptiveEngine PUBLIC ${BOSSHazardAdaptiveEngine_BINARY_DIR}/deps/lib)
add_dependencies(BOSSHazardAdaptiveEngine BOSS)

target_include_directories(BOSSHazardAdaptiveEngine SYSTEM PRIVATE ${BOSSHazardAdaptiveEngine_BINARY_DIR}/papi/install/include)
target_link_libraries(BOSSHazardAdaptiveEngine PRIVATE ${BOSSHazardAdaptiveEngine_BINARY_DIR}/papi/install/lib/libpapi.so)
add_dependencies(BOSSHazardAdaptiveEngine papi)

target_include_directories(BOSSHazardAdaptiveEngine PRIVATE ${BOSSHazardAdaptiveEngine_BINARY_DIR}/robin-map/src/robin-map/include)
add_dependencies(BOSSHazardAdaptiveEngine robin-map)

target_include_directories(BOSSHazardAdaptiveEngine PRIVATE ${BOSSHazardAdaptiveEngine_BINARY_DIR}/jsoncpp/src/jsoncpp/include)
target_link_libraries(BOSSHazardAdaptiveEngine PRIVATE ${BOSSHazardAdaptiveEngine_BINARY_DIR}/jsoncpp/src/jsoncpp-build/lib/libjsoncpp.so)
add_dependencies(BOSSHazardAdaptiveEngine jsoncpp)

set_target_properties(BOSSHazardAdaptiveEngine PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS BOSSHazardAdaptiveEngine LIBRARY DESTINATION lib)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
