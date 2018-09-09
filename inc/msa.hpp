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
#define MSA_DEBUG

/*
 * The software's information. Any of the provided information piece can be printed
 * from the command line as an argument.
 */
#define MSA         "msa"
#define MSA_VERSION "0.1.alpha"
#define MSA_AUTHOR  "Rodrigo Albuquerque de Oliveira Siqueira"
#define MSA_EMAIL   "rodriados@gmail.com"

/* 
 * Checks whether the system we are compiling in is POSIX compatible. If it
 * is not POSIX compatible, some conditional compiling may take place.
 */
#if defined(unix) || defined(__unix__) || defined(__unix) || defined(__linux__)
#  define MSA_POSIX
#  define MSA_UNIX
#elif defined(__APPLE__) && defined(__MACH__)
#  define MSA_POSIX
#  define MSA_APPLE
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#  define MSA_WINDOWS
#endif

#include <cstdio>
#include <string>
#include <cstdint>

#include "colors.h"
#include "node.hpp"

/*
 * Defines some debug macro functions. These functions should be used to
 * print out debugging information.
 */ 
#ifdef MSA_DEBUG
#  define pdebug(msg, ...) fprintf(stderr, MSA style(bold, " [debug]: ") msg "\n", ##__VA_ARGS__)
#else
#  ifndef __CUDA_ARCH__
#    define pdebug(msg, ...) if(verbose) { fprintf(stderr, MSA style(bold, " [debug]: ") msg "\n", ##__VA_ARGS__); }
#  else
#    define pdebug(msg, ...) // do not print anything from device, please.
#  endif
#endif

/**
 * This struct handles error messages so they can be easily accessible.
 * This struct shall be inherited by any module willing to keep its errors.
 * @since 0.1.alpha
 */
struct Error
{
    std::string msg;
    Error() : msg("") {}
    Error(const std::string& msg) : msg(msg) {}
    static const Error success() { return Error(); };
};

/*
 * Declaring global variables and functions.
 */
extern bool verbose;
[[noreturn]] extern void finalize(Error);

#endif