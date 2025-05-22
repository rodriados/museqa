# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file CMake configuration file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
cmake_minimum_required(VERSION 3.29)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(HelperFunctions)

# Discovers the project's version and makes it readily available in a variable.
# The version is sourced from the code itself.
get_project_version(MUSEQA_VERSION "src/museqa/version.h")

# Determines whether the current project is the master target or it is being built
# as a dependency, potentially as a thirdparty, to an external project.
if(NOT DEFINED MUSEQA_MASTER_PROJECT)
  set(MUSEQA_MASTER_PROJECT OFF)
  if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(MUSEQA_MASTER_PROJECT ON)
    message(STATUS "Project: Museqa ${MUSEQA_VERSION}")
    message(STATUS "CMake version: ${CMAKE_VERSION}")
  endif()
endif()

# Command line options to control the conditions and generation of the targets.
# The targets may be enabled on demand, depending on the user or environment needs.
option(MUSEQA_INSTALL "Generate the install target." ON)

cmake_dependent_option(MUSEQA_DEVELOPMENT_BUILD "Build tests, enable warnings and errors." ON "BUILD_TESTING" OFF)
cmake_dependent_option(MUSEQA_BUILD_TESTING "Generate the test targets." ON "BUILD_TESTING" OFF)
cmake_dependent_option(MUSEQA_ENABLE_PEDANTIC "Errors on warnings." ON "BUILD_TESTING" OFF)

# Discovers the project's current version and initializes it. The version is retrieved
# from a source file where the version should be explicitly set.
project(
  Museqa
    LANGUAGES CXX
    VERSION ${MUSEQA_VERSION}
    HOMEPAGE_URL "https://github.com/rodriados/museqa"
    DESCRIPTION "Multiple Sequence Aligner using hybrid parallel computing.")

include(GNUInstallDirs)

# Installation, configuration and targets variables. These variables are used to
# define where the project must be installed on and where to store its config files.
set(MUSEQA_CMAKE_CONFIG_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Museqa")
set(MUSEQA_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/museqa")
set(MUSEQA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(MUSEQA_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/MuseqaConfigVersion.cmake")
set(MUSEQA_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/MuseqaConfig.cmake")
set(MUSEQA_CONFIG_FILE_TEMPLATE "${CMAKE_CURRENT_LIST_DIR}/cmake/MuseqaConfig.cmake.in")
set(MUSEQA_TARGETS_EXPORT_NAME "MuseqaTargets")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

# Declaration of the project's targets. The targets are the main building blocks
# of the project and they are defined according to the its structure and requirements.
file(GLOB_RECURSE MUSEQA_SRC_CPPFILES  "${MUSEQA_SOURCE_DIR}/museqa/*.cpp")
file(GLOB_RECURSE MUSEQA_SRC_CUDAFILES "${MUSEQA_SOURCE_DIR}/museqa/*.cu")

add_executable(Museqa src/museqa.cpp)
add_library(MuseqaLib SHARED ${MUSEQA_SRC_CPPFILES})

target_compile_features(MuseqaLib INTERFACE cxx_std_17)
target_include_directories(
  MuseqaLib INTERFACE
    $<BUILD_INTERFACE:${MUSEQA_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(Museqa PUBLIC MuseqaLib)

# Determine whether optional external or third-party dependencies shall be used
# for the current project. Dependencies may be skipped if they are not needed.
if(NOT DEFINED MUSEQA_AVOID_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(MuseqaLib INTERFACE MPI::MPI_CXX)
  if(CMAKE_CXX_STANDARD LESS 17)
    set(MUSEQA_AVOID_THIRDPARTY_MPIWCPP17 ON)
  endif()
else()
  set(MUSEQA_AVOID_THIRDPARTY_MPIWCPP17 ON)
endif()

if(NOT DEFINED MUSEQA_AVOID_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(MuseqaLib PRIVATE ${MUSEQA_SRC_CUDAFILES})
  target_link_libraries(MuseqaLib INTERFACE CUDA::cudart)
  target_link_libraries(MuseqaLib INTERFACE CUDA::cudart_static)
  set_target_properties(MuseqaLib PROPERTIES CUDA_ARCHITECTURES native)
  set(CACHE CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
else()
  set(MUSEQA_AVOID_THIRDPARTY_CUDAWRAPPERS ON)
endif()

if(DEFINED MUSEQA_AVOID_THIRDPARTY)
  set(MUSEQA_AVOID_THIRDPARTY_CUDAWRAPPERS ON)
  set(MUSEQA_AVOID_THIRDPARTY_FMTLIB ON)
  set(MUSEQA_AVOID_THIRDPARTY_MPIWCPP17 ON)
  set(MUSEQA_AVOID_THIRDPARTY_REFLECTOR ON)
  set(MUSEQA_AVOID_THIRDPARTY_SUPERTUPLE ON)
endif()

# As the project has multiple dependencies, each one must be included and linked
# into the project according to each dependency's configuration and requirements.
if(NOT DEFINED MUSEQA_AVOID_THIRDPARTY_CUDAWRAPPERS)
  find_package(CudaWrappers REQUIRED)
  target_link_libraries(MuseqaLib INTERFACE cuda-api-wrappers::runtime-and-driver)
endif()

if(NOT DEFINED MUSEQA_AVOID_THIRDPARTY_FMTLIB)
  find_package(Fmtlib REQUIRED)
  target_link_libraries(MuseqaLib INTERFACE fmt::fmt)
endif()

if(NOT DEFINED MUSEQA_AVOID_THIRDPARTY_MPIWCPP17)
  find_package(MPIwCpp17 REQUIRED)
  target_link_libraries(MuseqaLib INTERFACE MPIwCpp17)
endif()

if(NOT DEFINED MUSEQA_AVOID_THIRDPARTY_REFLECTOR)
  find_package(Reflector REQUIRED)
  target_link_libraries(MuseqaLib INTERFACE Reflector)
endif()

if(NOT DEFINED MUSEQA_AVOID_THIRDPARTY_SUPERTUPLE)
  find_package(SuperTuple REQUIRED)
  target_link_libraries(MuseqaLib INTERFACE SuperTuple)
endif()

# Sets the compiler flags for a pedantic compilation. We mostly assume that the
# used compiler is GNU-compatible, although some special cases might be cared for.
if(MUSEQA_ENABLE_PEDANTIC)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-pedantic-errors -Wall -Wextra -pedantic -Werror)
  endif()
endif()

# Declaration of target for bundling and packing the project's source code to make
# it possible to include the whole library as a single header-only file.
if(MUSEQA_MASTER_PROJECT)
  add_subdirectory(src)
endif()

# Installs the library by copying all needed files to their corresponding destination
# folders. The library should be installed globally in the user system.
if(MUSEQA_INSTALL)
  install(
    TARGETS MuseqaLib
    EXPORT MuseqaTargets
    RUNTIME_DEPENDENCY_SET MuseqaDependencies
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  export(
    TARGETS MuseqaLib
    NAMESPACE Museqa::
    FILE "${PROJECT_BINARY_DIR}/MuseqaTargets.cmake")

  install(
    EXPORT MuseqaTargets
    NAMESPACE Museqa::
    DESTINATION ${MUSEQA_CMAKE_CONFIG_DESTINATION})

  install(
    DIRECTORY "${MUSEQA_SOURCE_DIR}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

  configure_package_config_file(
    ${MUSEQA_CONFIG_FILE_TEMPLATE}
    ${MUSEQA_CONFIG_FILE}
    INSTALL_DESTINATION ${MUSEQA_CMAKE_CONFIG_DESTINATION})

  write_basic_package_version_file(
    ${MUSEQA_VERSION_FILE}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  install(
    FILES
      ${MUSEQA_CONFIG_FILE}
      ${MUSEQA_VERSION_FILE}
    DESTINATION ${MUSEQA_CMAKE_CONFIG_DESTINATION})
endif()

# Declaration of targets for testing. These targets are only available when this
# project is root, therefore not used as dependency for any other.
if(BUILD_TESTING AND MUSEQA_MASTER_PROJECT AND MUSEQA_BUILD_TESTING)
  include(CTest)
  enable_testing()
  add_subdirectory(test)
endif()
