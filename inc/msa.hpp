/** 
 * Multiple Sequence Alignment main header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef MSA_HPP_INCLUDED
#define MSA_HPP_INCLUDED

#pragma once

/*
 * Leave uncommented if compiling in debug mode. This may affect many aspects
 * of the software, such as error reporting.
 */
#define msa_debug

/*
 * The software's information. Any of the provided information piece can be printed
 * from the command line as an argument.
 */
#define msa_appname "msa"
#define msa_version "0.1.1"
#define msa_author  "Rodrigo Albuquerque de Oliveira Siqueira"
#define msa_email   "rodriados@gmail.com"

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
  #error MSA is not compatible with Windows as yet.
#endif

#include <cstdint>
#include <cstddef>
#include <exception>
#include <string>

#include "colors.h"
#include "node.hpp"

/**
 * Holds an error message so it can be propagated through the code.
 * @since 0.1.1
 */
struct Exception : public std::exception
{
    std::string msg(128);       /// The exception message.

    /**
     * Builds a new exception instance.
     * @param msg The exception message.
     */
    explicit Exception(const std::string& msg)
    : msg(msg) {}

    /**
     * Builds a new exception instance.
     * @tparam T The format parameters' types.
     * @param fmt The message format.
     * @param args The format's parameters.
     */
    template <typename ...T>
    explicit Exception(const std::string& fmt, T... args)
    {
        sprintf(msg.data(), fmt.data(), args...);
    }

    Exception(const Exception&) = default;
    Exception(Exception&&) = delete;

    Exception& operator=(const Exception&) = default;
    Exception& operator=(Exception&&) = delete;

    /**
     * Returns an explanatory string.
     * @return The explanatory string.
     */
    inline const char *what() const noexcept
    {
        return msg.data();
    }
}

#ifndef msa_compile_cython
  extern void halt [[noreturn]] (uint8_t = 0);
  extern void version [[noreturn]] ();
#endif

/**
 * Prints an informative log message.
 * @tparam T The types of message arguments.
 * @param fmt The message format.
 * @param args The format values.
 */
template <typename ...T>
inline void info(const std::string& fmt, T... args)
{
#ifndef msa_compile_cython
    puts(s_bold "[info]" s_reset ": ");
    printf(fmt.data(), args...);
    putchar('\n');
#endif
}

/**
 * Prints an error log message and halts execution.
 * @tparam T The types of message arguments.
 * @param fmt The message format.
 * @param args The format values.
 */
template <typename ...T>
inline void error(const std::string& fmt, T... args)
{
#ifndef msa_compile_cython
    puts("[error]: ");
    printf(fmt.data(), args...);
    putchar('\n');
    halt(1);
#else
    throw Exception(fmt, args...);
#endif
}

/**
 * Prints a warning log message.
 * @tparam T The types of message arguments.
 * @param fmt The message format.
 * @param args The format values.
 */
template <typename ...T>
inline void warning(const std::string& fmt, T... args)
{
#ifndef msa_compile_cython
    puts(s_bold "[warning]" s_reset ": ");
    printf(fmt.data(), args...);
    putchar('\n');
#endif
}

/**
 * Prints a watchdog progress log message.
 * @tparam T The types of message arguments.
 * @param fmt The message format.
 * @param args The format values.
 */
template <typename ...T>
inline void watchdog(const std::string& task, uint32_t done, uint32_t total, const std::string& fmt, T... args)
{
#ifndef msa_compile_cython
    puts("[watchdog]: ");
    printf("%s %u %u %u %u ", task.data(), cluster::rank, cluster::size, done, total);
    printf(fmt.data(), args...);
    putchar('\n');
#endif
}

#endif
