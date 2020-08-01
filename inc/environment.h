/**
 * Multiple Sequence Alignment environment header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#pragma once

/*
 * The software version is in the form (major * 10000 + minor * 100 + patch).
 */
#define __msa_version 101

/*
 * The software's authorship information. If you need to get in touch with this
 * software's author, please use the info available below.
 */
#define __msa_appname "Multiple Sequence Alignment using hybrid parallel computing"
#define __msa_author  "Rodrigo Albuquerque de Oliveira Siqueira"
#define __msa_email   "rodriados at gmail dot com"

/*
 * Indicates the environment mode in which the compilation is taking place. The
 * compilation mode may affect some features' availability and performance.
 */
#if defined(DEBUG)
  #define __msa_debug
  #define __msa_environment 1
#elif defined(TESTING)
  #define __msa_testing
  #define __msa_environment 2
#elif defined(PRODUCTION)
  #define __msa_production
  #define __msa_environment 3
#else
  #define __msa_dev
  #define __msa_environment 4
#endif

/*
 * Finds the version of compiler being used. Some features might change or be unavailable
 * depending on the compiler configuration.
 */
#if defined(__GNUC__)
  #if !defined(__clang__)
    #define __msa_compiler_gcc
  #else
    #define __msa_compiler_clang
  #endif
  #define __msa_compiler_gnuc
#endif

#if defined(__NVCC__) || defined(__CUDACC__)
  #define __msa_compiler_nvcc
#endif

#if defined(__INTEL_COMPILER) || defined(__ICL)
  #define __msa_compiler_icc
#endif

#if defined(_MSC_VER)
  #define __msa_compiler_msc
#endif

/* 
 * Discovers about the environment in which the software is being compiled. Some
 * conditional compiling may take place depending on the environment.
 */
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
  #if defined(__linux__) || defined(__gnu_linux__)
    #define __msa_os_linux
  #else
    #define __msa_os_apple
  #endif
  #define __msa_os_unix
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
  #define __msa_os_windows
#endif

/*
 * Checks whether the current compiler is compatible with the software's prerequisites.
 * Should it not be compatible, then we emit a warning and try compiling anyway.
 */
#if !defined(__msa_compiler_gcc) && !defined(__msa_compiler_nvcc)
  #if defined(__msa_production)
    #warning this software has not been tested with the current compiler
  #endif
#else
  #define __msa_compiler_tested
#endif

/*
 * Checks whether the software has been tested in the current compilation environment.
 * Were it not tested, we emit a warning and try compiling anyway.
 */
#if !defined(__msa_os_linux)
  #if defined(__msa_production)
    #warning this software has not been tested under the current environment
  #endif
#else
  #define __msa_os_tested
#endif

/*
 * Discovers about the runtime environment. As this software uses GPUs to run calculations
 * on, we need to know, at times, whether the code is being executed in CPU or GPU.
 */
#if defined(__msa_compiler_nvcc) && defined(__CUDA_ARCH__)
  #define __msa_runtime_device
#else
  #define __msa_runtime_host
#endif

#if defined(__msa_testing) || defined(CYTHON_ABI)
  #define __msa_runtime_cython
#endif
