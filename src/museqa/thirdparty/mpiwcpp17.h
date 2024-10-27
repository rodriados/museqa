/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Configuration and inclusion of the mpiwcpp17 third party library.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>

#ifdef MUSEQA_AVOID_MPI
  #ifndef MUSEQA_AVOID_MPIWCPP17
    #define MUSEQA_AVOID_MPIWCPP17
  #endif
#else
  #include <mpi.h>
#endif

#ifndef MUSEQA_AVOID_MPIWCPP17
#if MUSEQA_CPP_DIALECT >= 2017
  #include <mpiwcpp17.h>
#else
  #error library MPIwCPP17 requires C++17 or later
#endif

MPIWCPP17_BEGIN_NAMESPACE

/**
 * Evaluates whether the current process is the one with the given rank in communicator.
 * @param rank The rank to check whether corresponds to the current process.
 * @param comm The communicator to check for process.
 * @return Does the current process has the given rank in the communicator?
 */
MPIWCPP17_INLINE bool is_process(process_t rank, communicator_t comm = world) noexcept
{
    return global::rank == communicator::rank(comm);
}

/**
 * Evaluates whether the current process is the root process in communicator.
 * @param comm The communicator to check for process.
 * @return Is the current process the communicator's root process?
 */
MPIWCPP17_INLINE bool is_root(communicator_t comm = world) noexcept
{
    return is_process(process::root, comm);
}

MPIWCPP17_END_NAMESPACE

#endif
