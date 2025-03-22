/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Global algorithm parameters definition.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/thirdparty/mpiwcpp17.h>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::algorithm
{
    /**
     * Defines the parameters that are globally revelant and must be shared between
     * every execution step of a heuristic pipeline.
     * @since 1.0
     */
    struct parameters_t
    {
      #ifndef MUSEQA_AVOID_MPI
        mpi::communicator_t comm = mpi::world;
        mpi::process_t root = mpi::process::root;
      #endif
    };
}

MUSEQA_END_NAMESPACE
