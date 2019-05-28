/** 
 * Multiple Sequence Alignment main header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef MSA_HPP_INCLUDED
#define MSA_HPP_INCLUDED

/*
 * Leave uncommented if compiling in debug mode. This may affect many aspects
 * of the software, such as error reporting.
 */
#define msa_debug 1

/*
 * The software's information. Any of the provided information piece can be printed
 * from the command line as an argument.
 */
#define msa_appname "msa"
#define msa_author  "Rodrigo Albuquerque de Oliveira Siqueira"
#define msa_email   "rodriados@gmail.com"

/*
 * The software version is in the form major * 10000 + minor * 100 + patch.
 */
#define msa_version 00101

/*
 * Finds the version of compiler being used. Some features might change depending on
 * the version of compiler being used.
 */
#if defined(__GNUC__) && !defined(__clang__)
  #define msa_gcc_version (__GNUC__ * 100 + __GNUC_MINOR__)
  #define msa_gcc 1
#else
  #define msa_gcc_version 0
#endif

#ifdef __clang__
  #define msa_clang_version (__clang_major__ * 100 + __clang_minor__)
  #define msa_clang 1
#else
  #define msa_clang_version 0
#endif

#ifdef __INTEL_COMPILER
  #define msa_icc_version __INTEL_COMPILER
  #define msa_icc 1
#elif defined(__ICL)
  #define msa_icc_version __ICL
  #define msa_icc 1
#else
  #define msa_icc_version 0
#endif

#ifdef _MSC_VER
  #define msa_msc_version _MSC_VER
  #define msa_msc 1
#else
  #define msa_msc_version 0
#endif

#ifdef __NVCC__
  #define msa_cuda_version (__CUDA_VER_MAJOR__ * 100 + __CUDA_VER_MINOR__)
  #define msa_cuda 1
#else
  #define msa_cuda_version 0
#endif

/* 
 * Checks whether the system we are compiling in is POSIX compatible. If it
 * is not POSIX compatible, some conditional compiling may take place.
 */
#if defined(unix) || defined(__unix__) || defined(__unix) || defined(__linux__)
  #define msa_posix
  #define msa_unix
#elif defined(__APPLE__) && defined(__MACH__)
  #define msa_posix
  #define msa_apple
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32)
  #define msa_windows
#endif

/*
 * Likeliness annotations are useful in the rare cases the author knows better
 * than the compiler whether a branch condition is overwhelmingly likely to take
 * a specific value. Also, it allows the author to show the compiler which code
 * paths are designed as the fast path, so they can be optimized.
 */
#if defined(msa_gcc) && !defined(likely)
  #define likely(x)   (__builtin_expect((x), 1))
  #define unlikely(x) (__builtin_expect((x), 0))
#else
  #define likely(x)   (x)
  #define unlikely(x) (x)
#endif

#include <string>
#include <cstdarg>
#include <cstddef>
#include <cstdint>

#include "node.hpp"
#include "exception.hpp"

namespace msa
{
#if !defined(msa_compile_cython)
    extern void halt(uint8_t = 0);
#endif

    template <typename ...T>
    inline void task(const char *, const char *, T&&...) noexcept;

    template <typename ...T>
    inline void info(const char *, T&&...) noexcept;

    template <typename ...T>
    inline void error(const char *, T&&...);

    template <typename ...T>
    inline void warning(const char *, T&&...) noexcept;

    inline void report(const char *, double) noexcept;
};

#if defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wformat-security"
#endif

/**
 * Informs the watchdog about the task we are currently processing.
 * @param taskname The name of the task being processed.
 * @param fstr The message formatting string.
 * @param args The message parts to be printed.
 */
template <typename ...T>
inline void msa::task(const char *taskname, const char *fstr, T&&... args) noexcept
{
#ifndef msa_compile_cython
    printf("[task] %s ", taskname); printf(fstr, args...); putchar('\n');
#endif
}

/**
 * Prints an informative log message.
 * @tparam T The types of message arguments.
 * @param fstr The message formating string.
 * @param args The message parts to be printed.
 */
template <typename ...T>
inline void msa::info(const char *fstr, T&&... args) noexcept
{
#ifndef msa_compile_cython
    puts("[info] "); printf(fstr, args...); putchar('\n');
#endif
}

/**
 * Prints an error log message and halts execution.
 * @tparam T The types of message arguments.
 * @param fstr The message formating string.
 * @param args The message parts to be printed.
 */
template <typename ...T>
inline void msa::error(const char *fstr, T&&... args)
{
#ifndef msa_compile_cython
    puts("[error] "); printf(fstr, args...); putchar('\n');
    msa::halt(1);
#else
    throw Exception(fstr, args...);
#endif
}

/**
 * Prints a warning log message.
 * @tparam T The types of message arguments.
 * @param fstr The message formating string.
 * @param args The message parts to be printed.
 */
template <typename ...T>
inline void msa::warning(const char *fstr, T&&... args) noexcept
{
#ifndef msa_compile_cython
    puts("[warning] "); printf(fstr, args...); putchar('\n');
#endif
}

/**
 * Prints a time report for given task.
 * @param taskname The name of completed task.
 * @param seconds The duration in seconds of given task.
 */
inline void msa::report(const char *taskname, double seconds) noexcept
{
#ifndef msa_compile_cython
    onlymaster printf("[report] %s in %lf seconds\n", taskname, seconds);
#endif
}

#if defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

#endif
