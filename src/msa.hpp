/** 
 * Multiple Sequence Alignment main header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _MSA_HPP_
#define _MSA_HPP_

/*
 * Leave uncommented if compiling in debug mode. This may affect many aspects
 * of the software, such as error reporting.
 */
#define DEBUG

/*
 * The software's name and version. This information can be printed from
 * command line as an argument.
 */
#define MSA     "msa"
#define VERSION "0.1.alpha"

/* 
 * Checks whether the system we are compiling in is POSIX compatible. If it
 * is not POSIX compatible, some conditional compiling may take place.
 */
#if defined(unix) || defined(__unix__) || defined(__unix) || defined(__linux__)
#  define __msa_posix__
#  define __msa_unix__
#elif defined(__APPLE__) && defined(__MACH__)
#  define __msa_posix__
#  define __msa_apple__
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#  define __msa_windows__
#endif

#include <cstdint>
#include <cstdio>

#include "node.h"

/**
 * Enumerates errors so their message can be easily printed.
 * @since 0.1.alpha
 */
enum class ErrorCode : uint8_t
{
    SUCCESS = 0
,   NOFILE
,   INVALIDFILE
,   INVALIDARG
,   NOGPU
,   CUDAERROR
};

/*
 * Declaring global variables and functions.
 */
extern bool verbose;
extern void finalize(ErrorCode);

/*
 * Defines some debug macro functions. These functions should be used to
 * print out debugging information.
 */ 
#ifdef DEBUG
#  define __debugh(msg, ...) printf("[  msa:host] " msg "\n", ##__VA_ARGS__)
#  define __debugd(msg, ...) printf("[msa:device] " msg "\n", ##__VA_ARGS__)
#else
#  define __debugh(msg, ...) if(verbose) {printf("[  msa:host] " msg "\n", ##__VA_ARGS__);}
#  define __debugd(msg, ...) if(verbose) {printf("[msa:device] " msg "\n", ##__VA_ARGS__);}
#endif

#endif