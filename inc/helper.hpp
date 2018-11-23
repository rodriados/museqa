/** 
 * Multiple Sequence Alignment helper functions header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef HELPER_HPP_INCLUDED
#define HELPER_HPP_INCLUDED

#pragma once

#include <cstdint>
#include <cstdio>

#include "colors.h"
#include "config.h"

#ifdef msa_compile_cython
  #include <stdexcept>
#else
  #include "node.hpp"
#endif

/*
 * Prints out general information to console. When compiling for Python though,
 * no text shall be printed. The info tag will not affect the watchdog.
 */ 
#ifdef msa_compile_cython
  #define info(...)
#else
  #define info(format, ...)                                                     \
    printf(s_bold "[info] " s_reset format "\n", ##__VA_ARGS__);                \
    fflush(stdout);
#endif

/*
 * Reports an error and exits the software execution. When compiling for Python,
 * throw an exception informing about the error. The error report will make
 * the watchdog forceably kill the process and finish all execution.
 */
#ifdef msa_compile_cython
  #define error(format, ...) {                                                  \
    char string[1024];                                                          \
    sprintf(string, format, ##__VA_ARGS__);                                     \
    throw std::logic_error(string);                                             \
  }
#else
[[noreturn]] extern void error(const char *, ...);
#endif

/*
 * Print a warning to console. Again, no text shall be printed for Python. The
 * warning tag causes no reaction to the watchdog. Thus, its purpose is solely
 * warning the user about an eventual error.
 */
#ifdef msa_compile_cython
  #define warning(...)
#else
  #define warning(format, ...)                                                  \
    printf("[warning] " format "\n", ##__VA_ARGS__);                            \
    fflush(stdout);
#endif

/*
 * Informs execution progress for the watchdog. No text is shown for Python. This
 * tag is responsible for keeping the watchdog updated about execution progress.
 */
#ifdef msa_compile_cython
  #define watchdog(...)
#else
  #define watchdog(task, done, total, format, ...)                              \
    printf("[watchdog] %s %u %u %u %u " format "\n",                            \
        task, cluster::rank, cluster::size, done, total, ##__VA_ARGS__);        \
    fflush(stdout);
#endif

extern void version();

#endif