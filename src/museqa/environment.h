/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Environment configuration and macro values.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/version.h>

#include <museqa/environment/compiler.h>
#include <museqa/environment/language.h>
#include <museqa/environment/runtime.h>
#include <museqa/environment/annotations.h>
#include <museqa/environment/namespace.hpp>

/*
 * Enumerates all possible target environment modes to which the code might be compiled
 * to. The environment mode may affect some features' availability and performace.
 */
#define MUSEQA_BUILD_DEV        0
#define MUSEQA_BUILD_DEBUG      1
#define MUSEQA_BUILD_TESTING    2
#define MUSEQA_BUILD_PRODUCTION 3

/*
 * Discovers and explicits the target environment mode to which the code must be
 * currently compiled to. The mode may affect some features' availability and performance.
 */
#if defined(DEBUG) || defined(_DEBUG)
  #define MUSEQA_BUILD MUSEQA_BUILD_DEBUG
  #define MUSEQA_ENVIRONMENT "Debug"
#elif defined(TESTING)
  #define MUSEQA_BUILD MUSEQA_BUILD_TESTING
  #define MUSEQA_ENVIRONMENT "Testing"
#elif defined(PRODUCTION)
  #define MUSEQA_BUILD MUSEQA_BUILD_PRODUCTION
  #define MUSEQA_ENVIRONMENT "Production"
#else
  #define MUSEQA_BUILD MUSEQA_BUILD_DEV
  #define MUSEQA_ENVIRONMENT "Development"
#endif

/*
 * Checks the current compiler's C++ language level. As the majority of this software's
 * codebase is written in C++, we must check whether its available or not.
 */
#if !defined(MUSEQA_IGNORE_CPP_DIALECT)
  #if !defined(MUSEQA_CPP_DIALECT) || MUSEQA_CPP_DIALECT < 2017
    #error This software requires at least a C++17 enabled compiler.
  #endif
#endif

/*
 * Checks whether the current compiler is compatible with the software's prerequisites.
 * Should it not be compatible, then we emit a warning but try compiling anyway.
 */
#if !defined(MUSEQA_IGNORE_COMPILER_CHECK) && MUSEQA_BUILD == MUSEQA_BUILD_PRODUCTION
  #if MUSEQA_HOST_COMPILER == MUSEQA_HOST_COMPILER_UNKNOWN
    #warning This software has not been tested with the current compiler.
  #endif
#endif

/*
 * Determines whether the software should run in unsafe mode. By default, safe mode
 * is turned off in production builds in exchange to better performance.
 */
#if MUSEQA_BUILD == MUSEQA_BUILD_PRODUCTION || MUSEQA_RUNTIME_DEVICE == 1
  #if !defined(MUSEQA_MODE_UNSAFE)
    #define MUSEQA_MODE_UNSAFE 1
  #endif
#endif

/*
 * Determines features that must be disabled depending on the current language version
 * available for compilation.
 */
#if MUSEQA_CPP_DIALECT < 2014 || MUSEQA_HOST_COMPILER != MUSEQA_HOST_COMPILER_GCC
  #if !defined(MUSEQA_AVOID_REFLECTION)
    #define MUSEQA_AVOID_REFLECTION
  #endif
#endif
