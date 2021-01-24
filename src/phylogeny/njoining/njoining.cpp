/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the phylogeny module's neighbor-joining algorithm.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#include "museqa.hpp"

#include "mpi.hpp"
#include "environment.h"

#include "phylogeny/phylogeny.cuh"
#include "phylogeny/njoining/njoining.cuh"

namespace museqa
{
    namespace phylogeny
    {
        namespace njoining
        {
            /**
             * The operator for reducing a list of join pair candidates. This operator
             * will always return the candidate with the closest nodes.
             * @param a The first join pair candidate to compare.
             * @param b The second join pair candidate to compare.
             * @return The candidate with the minimum distance.
             */
            auto closest(const joinable& a, const joinable& b) -> joinable
            {
                return a.distance > b.distance ? a : b;
            }

            /**
             * Reduces join pair candidates from all nodes and returns the one with
             * the minimum distance to master and all working nodes.
             * @param candidate The current working node's join pair candidate.
             * @return The globally best join pair candidate.
             */
            auto algorithm::reduce(joinable& candidate) const -> joinable
            {
                #if !defined(__museqa_runtime_cython)
                    static auto mpiop = mpi::op::create<joinable>(closest);
                    return mpi::allreduce(candidate, mpiop);
                #else
                    return candidate;
                #endif
            }

            /**
             * Picks the module's default algorithm according to the program's global
             * state conditions and devices availability.
             * @return The picked algorithm instance.
             */
            auto best() -> phylogeny::algorithm *
            {
                if (node::count > 1 || global_state.mpi_running) {
                    return global_state.use_devices
                        ? njoining::hybrid_symmetric()
                        : njoining::sequential_symmetric();
                } else {
                    return global_state.local_devices > 0
                        ? njoining::hybrid_symmetric()
                        : njoining::sequential_symmetric();
                }
            }
        }
    }
}
