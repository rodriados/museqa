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
#else
  #define msa_gcc_version 0
#endif

#ifdef __clang__
  #define msa_clang_version (__clang_major__ * 100 + __clang_minor__)
#else
  #define msa_clang_version 0
#endif

#ifdef __INTEL_COMPILER
  #define msa_icc_version __INTEL_COMPILER
#elif defined(__ICL)
  #define msa_icc_version __ICL
#else
  #define msa_icc_version 0
#endif

#ifdef _MSC_VER
  #define msa_msc_version _MSC_VER
#else
  #define msa_msc_version 0
#endif

#ifdef __NVCC__
  #define msa_cuda_version (__CUDA_VER_MAJOR__ * 100 + __CUDA_VER_MINOR__)
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

#include <iostream>
#include <cstdint>

namespace msa
{
    /**#@+
     * Formats a log message, split in the arguments to an output stream.
     * @param os The output stream to put the message into.
     * @param obj The object to be currently formatted to output.
     * @param rest The other objects in queue to be formatted to output.
     */
    template <typename T>
    inline void log(std::ostream& stream, const T& obj)
    {
        stream << (obj) << std::endl;
    }

    template <typename T, typename ...U>
    inline void log(std::ostream& stream, const T& obj, const U&... rest)
    {
        stream << (obj) << ' ';
        log(stream, rest...);
    }
    /**#@-*/

#if !defined(msa_compile_cython)
    extern void halt(uint8_t = 0);
#endif
};

#include <cstddef>
#include <string>

#include "colors.h"
#include "node.hpp"
#include "utils.hpp"
#include "exception.hpp"

/**
 * Prints an informative log message.
 * @tparam T The types of message arguments.
 * @param args The message parts to be printed.
 */
template <typename ...T>
inline void info(const T&... args)
{
#if !defined(msa_compile_cython)
    msa::log(std::cout, s_bold "[info]" s_reset, args...);
#endif
}

/**
 * Prints an error log message and halts execution.
 * @tparam T The types of message arguments.
 * @param args The message parts to be printed.
 */
template <typename ...T>
inline void error(const T&... args)
{
#if !defined(msa_compile_cython)
    msa::log(std::cout, "[error]", args...);
    msa::halt(1);
#else
    throw Exception(args...);
#endif
}

/**
 * Prints a warning log message.
 * @tparam T The types of message arguments.
 * @param args The message parts to be printed.
 */
template <typename ...T>
inline void warning(const T&... args)
{
#if !defined(msa_compile_cython)
    msa::log(std::cout, s_bold "[warning]" s_reset, args...);
#endif
}

/**
 * Prints a watchdog progress log message.
 * @tparam T The types of message arguments.
 * @param task The task being currently processed.
 * @param done The number of subtasks already processed.
 * @param total The total number of subtasks to process.
 * @param nodes The number of nodes working on the process.
 * @param args The message parts to be printed.
 */
template <typename ...T>
inline void watchdog(const char *task, size_t done, size_t total, int nodes, const T&... args)
{
#if !defined(msa_compile_cython)
    msa::log(std::cout, "[watchdog]", task, node::rank, nodes, done, total, args...);
#endif
}

#endif
