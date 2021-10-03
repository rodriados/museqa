/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Environment configuration and macro values.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

/**
 * Information about the source code version used to compile the project. The version
 * allows the developers to identify when an error was introduced to the codebase,
 * whenever a new bug is reported. Also, it allows the control of feature deprecation
 * and introduction.
 */
#define MUSEQA_VERSION 1000 // The version number is (major * 1000 + minor)

/*
 * The software's authorship information. If you need to get in touch with this
 * software's author, please use the info available below.
 */
#define MUSEQA_APPNAME "Museqa: Multiple Sequence Aligner using hybrid parallel computing."
#define MUSEQA_AUTHOR  "Rodrigo Albuquerque de Oliveira Siqueira"
#define MUSEQA_EMAIL   "rodriados at gmail dot com"

/*
 * Indicates the target environment mode to which the code must be compiled to.
 * The mode may affect some features' availability and performance.
 */
#if defined(DEBUG)
  #define MUSEQA_DEBUG "debug"
  #define MUSEQA_ENV 1
#elif defined(TESTING)
  #define MUSEQA_TESTING "testing"
  #define MUSEQA_ENV 2
#elif defined(PRODUCTION)
  #define MUSEQA_PRODUCTION "production"
  #define MUSEQA_ENV 3
#else
  #define MUSEQA_DEV "development"
  #define MUSEQA_ENV 4
#endif

/*
 * Checks the current compiler's C++ language level. As the majority of this software's
 * codebase is written in C++, we must check whether its available or not.
 */
#if !defined(__cplusplus) || __cplusplus < 201703L
  #error "This software requires at least a C++17 enabled compiler."
#else
  #define MUSEQA_CPP __cplusplus
#endif

/*
 * Finds the version of compiler being used. Some features might change or be unavailable
 * depending on the current compiler configuration.
 */
#if defined(__GNUC__)
  #if !defined(__clang__)
    #define MUSEQA_COMPILER_GCC (__GNUC__ * 100 + __GNUC_MINOR__)
    #define MUSEQA_COMPILER "gcc"
  #else
    #define MUSEQA_COMPILER_CLANG (__clang_major__ * 100 + __clang_minor__)
    #define MUSEQA_COMPILER "clang"
  #endif
  #define MUSEQA_COMPILER_GNUC 1
#endif

#if defined(__NVCC__) || defined(__CUDACC__)
  #define MUSEQA_COMPILER_NVCC (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
  #if defined(MUSEQA_COMPILER)
    #undef MUSEQA_COMPILER
  #endif
  #define MUSEQA_COMPILER "nvcc"
#endif

#if defined(__INTEL_COMPILER) || defined(__ICL)
  #define MUSEQA_COMPILER_ICC __INTEL_COMPILER
  #define MUSEQA_COMPILER "icc"
#endif

#if defined(_MSC_VER)
  #define MUSEQA_COMPILER_MSC _MSC_VER
  #define MUSEQA_COMPILER "msc"
#endif

#if !defined(MUSEQA_COMPILER)
  #define MUSEQA_COMPILER_UNKNOWN 1
  #define MUSEQA_COMPILER "unknown"
#endif

/* 
 * Discovers about the environment in which the software is being compiled. Some
 * conditional compiling may take place depending on the environment.
 */
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
  #if defined(__linux__) || defined(__gnu_linux__)
    #define MUSEQA_OS_LINUX 1
    #define MUSEQA_OS "linux"
  #else
    #define MUSEQA_OS_APPLE 1
    #define MUSEQA_OS "apple"
  #endif
  #define MUSEQA_OS_UNIX 1
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32)
  #define MUSEQA_OS_WINDOWS 1
  #define MUSEQA_OS "windows"
#endif

#if !defined(MUSEQA_OS)
  #define MUSEQA_OS_UNKNOWN 1
  #define MUSEQA_OS "unknown"
#endif

/*
 * Checks whether the current compiler is compatible with the software's prerequisites.
 * Should it not be compatible, then we emit a warning but try compiling anyway.
 */
#if defined(MUSEQA_COMPILER_UNKNOWN)
  #if defined(MUSEQA_PRODUCTION)
    #warning "This software has not been tested with the current compiler."
  #endif
#else
  #define MUSEQA_COMPILER_TESTED 1
#endif

/*
 * Checks whether the software has been tested in the current compilation environment.
 * Were it not tested, we emit a warning but try compiling anyway.
 */
#if defined(MUSEQA_OS_UNKNOWN)
  #if defined(MUSEQA_PRODUCTION)
    #warning "This software has not been tested under the current environment."
  #endif
#else
  #define MUSEQA_OS_TESTED 1
#endif

/*
 * Discovers about the runtime environment. As this software uses GPUs to run calculations
 * on, we need to know at times whether the code is being executed in CPU or GPU.
 */
#if defined(MUSEQA_COMPILER_NVCC) && defined(__CUDA_ARCH__)
  #define MUSEQA_RUNTIME_DEVICE 1
  #define MUSEQA_RUNTIME (0x01)
#else
  #define MUSEQA_RUNTIME_HOST 1
  #define MUSEQA_RUNTIME (0x10)
#endif

/*
 * Determines whether the software should run in unsafe mode. By default, safe mode
 * is turned off in production builds in exchange to performance.
 */
#if defined(MUSEQA_PRODUCTION) || defined(MUSEQA_RUNTIME_DEVICE)
  #if !defined(MUSEQA_UNSAFE)
    #define MUSEQA_UNSAFE 1
  #endif
#endif

/*
 * Determines features that must be disabled depending on the current language version
 * available for compilation.
 */
#if MUSEQA_CPP < 201402L
  #if !defined(MUSEQA_AVOID_REFLECTION)
    #define MUSEQA_AVOID_REFLECTION
  #endif
#endif
