/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Configuration and inclusion of the mpiwcpp17 third party library.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>

#ifndef MUSEQA_AVOID_MPI
  #ifndef MUSEQA_AVOID_THIRDPARTY_MPIWCPP17
    #ifdef MUSEQA_OVERRIDE_MPIWCPP17
      #include MUSEQA_OVERRIDE_MPIWCPP17
    #elif __has_include(<rodriados/mpiwcpp17.h>)
      #include <rodriados/mpiwcpp17.h>
    #elif __has_include(<mpiwcpp17/api.h>)
      #include <mpiwcpp17/api.h>
    #else
      #include <mpiwcpp17.h>
    #endif
  #endif
#endif

#ifndef MUSEQA_AVOID_THIRDPARTY_MPIWCPP17
  #define MUSEQA_MPI_ONLY_COMPUTE   if (mpi::rank != mpi::process::root)
  #define MUSEQA_MPI_ONLY_ROOT      if (mpi::rank == mpi::process::root)
#else
  #define MUSEQA_MPI_ONLY_COMPUTE   if (false)
  #define MUSEQA_MPI_ONLY_ROOT      if (true)
#endif
