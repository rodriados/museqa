/** 
 * Multiple Sequence Alignment helper functions header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef HELPER_HPP_INCLUDED
#define HELPER_HPP_INCLUDED

#pragma once

#include <cstdio>

#include "colors.h"
#include "node.hpp"

#ifdef msa_compile_cython
  #include <stdexcept>
#endif

/*
 * Prints out general information to console. When compiling for Python though,
 * no text shall be printed. The info tag will not affect the watchdog.
 */ 
#ifndef msa_compile_cython
  #define info(fmt, ...) printf(s_bold "[info] " s_reset fmt "\n", ##__VA_ARGS__)
#else
  #define info(...)
#endif

/*
 * Print a warning to console. Again, no text shall be printed for Python. The
 * warning tag causes no reaction to the watchdog. Thus, its purpose is solely
 * warning the user about an eventual error.
 */
#ifndef msa_compile_cython
  #define warning(fmt, ...) printf(s_bold c_blue_fg "[warning] " s_reset fmt "\n", ##__VA_ARGS__)
#else
  #define warning(...)
#endif

/*
 * Reports an error and exits the software execution. When compiling for Python,
 * throw an exception informing about the error. The error report will make
 * the watchdog forceably kill the process and finish all execution.
 */
#ifndef msa_compile_cython
  #define error(fmt, ...) {                                                     \
    printf("[error] " fmt "\n", ##__VA_ARGS__);                                 \
    quit(1);                                                                    \
  }
#else
  #define error(fmt, ...) {                                                     \
    char string[1024];                                                          \
    sprintf(string, fmt, ##__VA_ARGS__);                                        \
    throw std::logic_error(string);                                             \
  }
#endif

/*
 * Informs execution progress for the watchdog. No text is shown for Python. This
 * tag is responsible for keeping the watchdog updated about execution progress.
 */
#ifndef msa_compile_cython
  #define watchdog(task, done, total, fmt, ...)                                 \
    printf("[watchdog] %s %u %u %u %u " fmt "\n",                               \
        task, cluster::rank, cluster::size, done, total, ##__VA_ARGS__)
#else
  #define watchdog(...) 
#endif

#ifndef msa_compile_cython
extern void quit [[noreturn]] (uint8_t = 0);
extern void version [[noreturn]] ();
#endif

#endif