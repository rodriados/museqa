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
#define msa_appname() "Multiple Sequence Alignment using hybrid parallel computing"
#define msa_author()  "Rodrigo Albuquerque de Oliveira Siqueira"
#define msa_email()   "rodriados at gmail dot com"

/*
 * The software version is in the form (major * 10000 + minor * 100 + patch).
 */
#define msa_version() 101

/*
 * Indicates the environment mode in which the compilation is taking place. The
 * compilation mode may affect some features' availability and performance.
 */
#if defined(DEBUG)
  #define msa_debug() 1
#else
  #define msa_debug() 0
#endif

#if defined(TESTING)
  #define msa_testing() 1
#else
  #define msa_testing() 0
#endif

#if !defined(DEBUG) && !defined(TESTING)
  #define msa_production() 1
#else
  #define msa_production() 0
#endif

/*
 * Finds the version of compiler being used. Some features might change or be unavailable
 * depending on the compiler configuration.
 */
#if defined(__GNUC__)
  #if !defined(__clang__)
    #define msa_compiler_gcc() 1
    #define msa_compiler_clang() 0
  #else
    #define msa_compiler_gcc() 0
    #define msa_compiler_clang() 1
  #endif
  #define msa_compiler_gnuc() 1
#else
  #define msa_compiler_gcc() 0
  #define msa_compiler_clang() 0
  #define msa_compiler_gnuc() 0
#endif

#if defined(__NVCC__) || defined(__CUDACC__)
  #define msa_compiler_nvcc() 1
#else
  #define msa_compiler_nvcc() 0
#endif

#if defined(__INTEL_COMPILER) || defined(__ICL)
  #define msa_compiler_icc() 1
#else
  #define msa_compiler_icc() 0
#endif

#if defined(_MSC_VER)
  #define msa_compiler_msc() 1
#else
  #define msa_compiler_msc() 0
#endif

/* 
 * Discovers about the environment in which the software is being compiled. Some
 * conditional compiling may take place depending on the environment.
 */
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
  #if defined(__linux__) || defined(__gnu_linux__)
    #define msa_os_linux() 1
    #define msa_os_apple() 0
  #else
    #define msa_os_linux() 0
    #define msa_os_apple() 1
  #endif
  #define msa_os_unix() 1
#else
  #define msa_os_linux() 0
  #define msa_os_apple() 0
  #define msa_os_unix() 0
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
  #define msa_os_windows() 1
#else
  #define msa_os_windows() 0
#endif

/*
 * Checks whether the current compiler is compatible with the software's prerequisites.
 * Should it not be compatible, then we emit a warning and try compiling anyway.
 */
#if !msa_compiler_gcc() && !msa_compiler_nvcc()
  #define msa_compiler_tested() 0
  #warning this software has not been tested with the current compiler
#else
  #define msa_compiler_tested() 1
#endif

/*
 * Checks whether the software has been tested in the current compilation environment.
 * Were it not tested, we emit a warning and try compiling anyway.
 */
#if !msa_os_linux()
  #define msa_os_tested() 0
  #warning this software has not been tested under the current environment
#else
  #define msa_os_tested() 1
#endif

/*
 * Discovers about the runtime environment. As this software uses GPUs to run calculations
 * on, we need to know, at times, whether the code is being executed in CPU or GPU.
 */
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
  #define msa_runtime_device() 1
  #define msa_runtime_host() 0
#else
  #define msa_runtime_device() 0
  #define msa_runtime_host() 1
#endif

#if msa_testing() || defined(CYTHON_ABI)
  #define msa_runtime_cython() 1
#else
  #define msa_runtime_cython() 0
#endif

/**
 * Gets the value of one of the detected environment macros.
 * @param value The name of requested environment macro.
 * @since 0.1.1
 */
#define __msa_concat_value(value)                                                   \
    msa_##value

/**
 * Gets a macro organized in one group.
 * @param group The group name to find value macro into.
 * @param value The name of the value macro to find.
 * @since 0.1.1
 */
#define __msa_concat_group(group, value)                                            \
    __msa_concat_value(group##_##value)()

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
