/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file MPI wrapper global variables and functions definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#if !defined(MUSEQA_AVOID_MPI)

#include <map>
#include <cstdint>

#include <museqa/node.hpp>
#include <museqa/mpi/lambda.hpp>

namespace museqa
{
    /**
     * Informs the current node's global rank within the instantiated MPI topology.
     * @see museqa::mpi::init
     */
    node::id node::rank;

    /**
     * Informs the total number of nodes within the current MPI topology.
     * @see museqa::mpi::init
     */
    int32_t node::count;

    /**
     * Informs the currently active MPI operator function. This is necessary to recover
     * a wrapped function from within MPI execution.
     * @since 1.0
     */
    mpi::function::id mpi::function::active;

    /**
     * Maps a function identifier to its actual user-defined implementation. Unfortunately,
     * this is maybe the only reliable way to inject a wrapped function into the
     * operator actually called by MPI.
     * @since 1.0
     */
    std::map<mpi::function::id, void*> mpi::function::fmapper;
}

#endif
