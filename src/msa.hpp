/*! \file msa.hpp
 * \brief Parallel Multiple Sequence Alignment main header file.
 * \author Rodrigo Siqueira <rodriados@gmail.com>
 * \copyright 2018 Rodrigo Siqueira
 */
#ifndef _MSA_HPP
#define _MSA_HPP

#define DEBUG
#define VERSION "0.1-alpha"

#if !defined(unix) && !defined(__unix__) && !defined(__unix)                 \
    && !defined(__linux__) && !(defined(__APPLE__) && defined(__MACH__))
#  error The host system is not POSIX compatible.
#endif

#include <iostream>

#ifdef DEBUG
#  define __hdebug__ (std::cerr << "[  msa:host] ")
#  define __ddebug__ (std::cerr << "[msa:device] ")
#else
#  define __hdebug__
#  define __ddebug__
#endif

extern struct msadata {
    int rank;
    int nproc;
    char *fname = NULL;
} gldata;

extern void finish(int = 0);

#endif