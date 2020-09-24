/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Environment configuration and macro values.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

/*
 * The software version is in the form (major * 10000 + minor * 100 + patch).
 */
#define __museqa_version 101

/*
 * The software's authorship information. If you need to get in touch with this
 * software's author, please use the info available below.
 */
#define __museqa_appname "Museqa: Multiple Sequence Aligner using hybrid parallel computing."
#define __museqa_author  "Rodrigo Albuquerque de Oliveira Siqueira"
#define __museqa_email   "rodriados at gmail dot com"

/*
 * Indicates the environment mode to which the code is being compiled to. The mode
 * may affect some features' availability and performance.
 */
#if defined(DEBUG)
  #define __museqa_debug
  #define __museqa_environment 1
#elif defined(TESTING)
  #define __museqa_testing
  #define __museqa_environment 2
#elif defined(PRODUCTION)
  #define __museqa_production
  #define __museqa_environment 3
#else
  #define __museqa_dev
  #define __museqa_environment 4
#endif

/*
 * Checks the current compiler's C++ language level. As the majority of this software's
 * codebase is written in C++, we must check whether its available or not.
 */
#if defined(__cplusplus)
  #define __museqa_cpp __cplusplus
#endif

#if !defined(__museqa_cpp) || __museqa_cpp < 201103L
  #error "This software requires a C++11 enabled compiler"
#endif

/*
 * Finds the version of compiler being used. Some features might change or be unavailable
 * depending on the current compiler configuration.
 */
#if defined(__GNUC__)
  #if !defined(__clang__)
    #define __museqa_compiler_gcc
    #define __museqa_compiler_version __GNUC__
  #else
    #define __museqa_compiler_clang
    #define __museqa_compiler_version __clang__
  #endif
  #define __museqa_compiler_gnuc
#endif

#if defined(__NVCC__) || defined(__CUDACC__)
  #define __museqa_compiler_nvcc
  #define __museqa_compiler_version __CUDACC__
#endif

#if defined(__INTEL_COMPILER) || defined(__ICL)
  #define __museqa_compiler_icl
  #define __museqa_compiler_version __ICL
#endif

#if defined(_MSC_VER)
  #define __museqa_compiler_msc
  #define __museqa_compiler_version _MSC_VER
#endif

/* 
 * Discovers about the environment in which the software is being compiled. Some
 * conditional compiling may take place depending on the environment.
 */
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
  #if defined(__linux__) || defined(__gnu_linux__)
    #define __museqa_os_linux
  #else
    #define __museqa_os_apple
  #endif
  #define __museqa_os_unix
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32)
  #define __museqa_os_windows
#endif

/*
 * Checks whether the current compiler is compatible with the software's prerequisites.
 * Should it not be compatible, then we emit a warning and try compiling anyway.
 */
#if !defined(__museqa_compiler_gcc) && !defined(__museqa_compiler_nvcc)
  #if defined(__museqa_production)
    #warning this software has not been tested with the current compiler
  #endif
#else
  #define __museqa_compiler_tested
#endif

/*
 * Checks whether the software has been tested in the current compilation environment.
 * Were it not tested, we emit a warning and try compiling anyway.
 */
#if !defined(__museqa_os_linux)
  #if defined(__museqa_production)
    #warning this software has not been tested under the current environment
  #endif
#else
  #define __museqa_os_tested
#endif

/*
 * Discovers about the runtime environment. As this software uses GPUs to run calculations
 * on, we need to know, at times, whether the code is being executed in CPU or GPU.
 */
#if defined(__museqa_compiler_nvcc) && defined(__CUDA_ARCH__)
  #define __museqa_runtime_device
#else
  #define __museqa_runtime_host
#endif

#if defined(__museqa_testing) || defined(CYTHON_ABI)
  #define __museqa_runtime_cython
#endif
