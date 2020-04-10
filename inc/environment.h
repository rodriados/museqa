/**
 * Multiple Sequence Alignment environment header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

/*
 * The software's authorship information. If you need to get in touch with this
 * software's author, please use the info available below.
 */
#define __msa_appname "Multiple Sequence Alignment using hybrid parallel computing"
#define __msa_author  "Rodrigo Albuquerque de Oliveira Siqueira"
#define __msa_email   "rodriados at gmail dot com"

/*
 * The software version is in the form (major * 10000 + minor * 100 + patch).
 */
#define __msa_version 00101

/*
 * Finds the version of compiler being used. Some features might change or be unavailable
 * depending on the compiler configuration.
 */
#if defined(__GNUC__)
  #define __msa_compiler_gnuc 1
  #if !defined(__clang__)
    #define __msa_compiler_gcc 1
    #define __msa_compiler_clang 0
    #define __msa_compiler_tested 1
  #else
    #define __msa_compiler_clang 1
    #define __msa_compiler_gcc 0
  #endif
#else
  #define __msa_compiler_gcc 0
  #define __msa_compiler_gnuc 0
  #define __msa_compiler_clang 0
#endif

#if defined(__NVCC__) || defined(__CUDACC__)
  #define __msa_compiler_nvcc 1
  #define __msa_compiler_tested 1
#else
  #define __msa_compiler_nvcc 0
#endif

#if defined(__INTEL_COMPILER) || defined(__ICL)
  #define __msa_compiler_icc 1
#else
  #define __msa_compiler_icc 0
#endif

#if defined(_MSC_VER)
  #define __msa_compiler_msc 1
#else
  #define __msa_compiler_msc 0
#endif

/*
 * Checks whether the current compiler is compatible with the software's prerequisites.
 * Should it not be compatible, then we emit a warning and try compiling anyway.
 */
#if !defined(__msa_compiler_tested)
  #warning this software has not been tested with the current compiler
#endif

/* 
 * Discovers about the environment in which the software is being compiled. Some
 * conditional compiling may take place depending on the environment.
 */
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
  #define __msa_env_unix 1
  #if defined(__linux__) || defined(__gnu_linux__)
    #define __msa_env_apple 0
    #define __msa_env_linux 1
    #define __msa_env_tested 1
  #else
    #define __msa_env_apple 1
    #define __msa_env_linux 0
  #endif
#else
  #define __msa_env_apple 0
  #define __msa_env_linux 0
  #define __msa_env_unix 0
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
  #define __msa_env_windows 1
#else
  #define __msa_env_windows 0
#endif

/*
 * Checks whether the software has been tested in the current compilation environment.
 * Were it not tested, we emit a warning and try compiling anyway.
 */
#if !defined(__msa_env_tested)
  #warning this software has not been tested under the current environment
#endif

/*
 * Discovers about the runtime environment. As this software uses GPUs to run calculations
 * on, we need to know, at times, whether the code is being executed in CPU or GPU.
 */
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
  #define __msa_runtime_cython 0
  #define __msa_runtime_device 1
  #define __msa_runtime_host 0
#elif defined(CYTHON_ABI) || defined(msa_target_cython)
  #define __msa_runtime_cython 1
  #define __msa_runtime_device 0
  #define __msa_runtime_host 1
#else
  #define __msa_runtime_cython 0
  #define __msa_runtime_device 0
  #define __msa_runtime_host 1
#endif

/**
 * Gets the value of one of the detected environment macros.
 * @param value The name of requested environment macro.
 * @since 0.1.1
 */
#define __msa_concat_value(value)                                                   \
    __msa_##value

/**
 * Gets a macro organized in one group.
 * @param group The group name to find value macro into.
 * @param value The name of the value macro to find.
 * @since 0.1.1
 */
#define __msa_concat_group(group, value)                                            \
    __msa_concat_value(group##_##value)

/**
 * Effectively implements optional macro arguments and picks concatenator.
 * @param zero An ignored argument, so we guarantee we have at least one argument.
 * @param opt The variadic argument, which if present shifts all following arguments.
 * @param tgt The macro concatenator to be picked.
 * @since 0.1.1
 */
#define __msa_optional_arg(zero, opt, tgt, ...)                                     \
    tgt

/**
 * Allows easy access to any of the detected environment macros.
 * @param m The requested macro name or group.
 * @since 0.1.1
 */
#define __msa(m, ...)                                                               \
    __msa_optional_arg(m, ##__VA_ARGS__, __msa_concat_group, __msa_concat_value, )  \
        (m, ##__VA_ARGS__)
