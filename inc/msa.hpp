/** 
 * Multiple Sequence Alignment main header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef MSA_HPP_INCLUDED
#define MSA_HPP_INCLUDED

#pragma once

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

#include <cstdint>
#include <cstddef>

#include "config.h"
#include "helper.h"

#include "node.hpp"

#endif
