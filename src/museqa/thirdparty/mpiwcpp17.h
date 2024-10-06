/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Configuration and inclusion of the mpiwcpp17 third party library.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>

#if !defined(MUSEQA_AVOID_MPI)
  #if !defined(MUSEQA_AVOID_MPIWCPP17)
    #if MUSEQA_CPP_DIALECT >= 2017
      #include <mpiwcpp17.h>
    #else
      #error library MPIwCPP17 requires C++17 or later
    #endif
  #endif
#endif
