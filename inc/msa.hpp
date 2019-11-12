/**
 * Multiple Sequence Alignment using hybrid parallel computing
 *
 * Copyright (c) 2018 - present, Rodrigo Siqueira
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once

#ifndef MSA_HPP_INCLUDED
#define MSA_HPP_INCLUDED

/*
 * Leave uncommented if compiling in debug mode. This may affect many aspects
 * of the software, such as error reporting.
 */
#define DEBUG 1

/*
 * The software's authorship information. If you need to get in touch with this
 * software's author, please use the info available below.
 */
#define MSA_APPNAME "Multiple Sequence Alignment using hybrid parallel computing"
#define MSA_AUTHOR  "Rodrigo Albuquerque de Oliveira Siqueira"
#define MSA_EMAIL   "rodrigo dot siqueira at ufms dot br"

/*
 * The software version is in the form (major * 10000 + minor * 100 + patch).
 */
#define MSA_VERSION 00101

/*
 * Finds the version of compiler being used. Some features might change or be unavailable
 * depending on the compiler configuration.
 */
#if defined(__GNUC__) && !defined(__clang__)
  #define MSA_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
  #define MSA_GCC 1
#else
  #define MSA_GCC_VERSION 0
#endif

#if defined(__NVCC__)
  #define MSA_NVCC_VERSION (__CUDA_VER_MAJOR__ * 100 + __CUDA_VER_MINOR__)
  #define MSA_NVCC 1
#else
  #define MSA_NVCC_VERSION 0
#endif

#if defined(__clang__)
  #define MSA_CLANG_VERSION (__clang_major__ * 100 + __clang_minor__)
  #define MSA_CLANG 1
#else
  #define MSA_CLANG_VERSION 0
#endif

#if defined(__INTEL_COMPILER)
  #define MSA_ICC_VERSION __INTEL_COMPILER
  #define MSA_ICC 1
#elif defined(__ICL)
  #define MSA_ICC_VERSION __ICL
  #define MSA_ICC 1
#else
  #define MSA_ICC_VERSION 0
#endif

#if defined(_MSC_VER)
  #define MSA_MSC_VERSION _MSC_VER
  #define MSA_MSC 1
#else
  #define MSA_MSC_VERSION 0
#endif

/* 
 * Discovers about the environment in which the software is being compiled. Some
 * conditional compiling may take place depending on the environment.
 */
#if defined(unix) || defined(__unix__) || defined(__unix) || defined(__linux__)
  #define MSA_UNIX 1
  #define MSA_POSIX 1
#elif defined(__APPLE__) && defined(__MACH__)
  #define MSA_APPLE 1
  #define MSA_POSIX 1
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32)
  #define MSA_WINDOWS 1
#endif

/*
 * Checks whether the software has been previously tested in the current environment
 * and compiler configurations.
 */
#if !defined(MSA_GCC) && !defined(MSA_NVCC)
  #warning this software has not been tested with the current compiler
#endif

#if !defined(MSA_POSIX) || defined(MSA_APPLE)
  #warning this software has not been tested in the current environment
#endif

#include <string>
#include <cstddef>
#include <cstdint>

#include <node.hpp>
#include <utils.hpp>
#include <format.hpp>
#include <exception.hpp>

namespace msa
{
    #if !defined(onlycython)
        /**
         * Halts the whole software's execution and exits with given code.
         * @param code The exit code.
         */
        extern void halt(uint8_t = 0) noexcept;
    #endif

    /**
     * Prints an error log message and halts execution.
     * @tparam T The types of message arguments.
     * @param fmtstr The message formating string.
     * @param args The message parts to be printed.
     */
    template <typename ...T>
    inline void error(const std::string& fmtstr, T&&... args)
    {
        #if !defined(onlycython)
            fmt::print("[error] " + fmtstr + '\n', args...);
            halt(1);
        #else
            throw exception {fmtstr, args...};
        #endif
    }

    /**
     * Prints an informative log message.
     * @tparam T The message arguments' types.
     * @param fmtstr The message format template.
     * @param args The message arguments.
     */
    template <typename ...T>
    inline void info(const std::string& fmtstr, T&&... args) noexcept
    {
        #if !defined(onlycython)
            fmt::print("[info] " + fmtstr + '\n', args...);
        #endif
    }

    /**
     * Prints a time report for given task.
     * @param taskname The name of completed task.
     * @param seconds The duration in seconds of given task.
     */
    inline void report(const char *taskname, double seconds) noexcept
    {
        #if !defined(onlycython)
            onlymaster fmt::print("[report] %s in %lf seconds\n", taskname, seconds);
        #endif
    }

    /**
     * Informs the watchdog about the task we are currently processing.
     * @param taskname The name of the task being processed.
     * @param fmtstr The message formatting string.
     * @param args The message parts to be printed.
     */
    template <typename ...T>
    inline void task(const char *taskname, const std::string& fmtstr, T&&... args) noexcept
    {
        #if !defined(onlycython)
            fmt::print("[task] %s " + fmtstr + '\n', taskname, args...);
        #endif
    }

    /**
     * Prints a warning log message.
     * @tparam T The types of message arguments.
     * @param fmtstr The message formating string.
     * @param args The message parts to be printed.
     */
    template <typename ...T>
    inline void warning(const std::string& fmtstr, T&&... args) noexcept
    {
        #if !defined(onlycython)
            fmt::print("[warning] " + fmtstr + '\n', args...);
        #endif
    }
};

#endif
