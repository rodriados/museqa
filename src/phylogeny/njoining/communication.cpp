/**
 * Multiple Sequence Alignment parallel needleman communication file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#include <mpi.hpp>

#include <environment.h>

#include <phylogeny/phylogeny.cuh>
#include <phylogeny/algorithm/njoining.cuh>

namespace msa
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
                const auto d1 = a.distance;
                const auto d2 = b.distance;
                return d1 < d2 ? a : b;
            }

            /**
             * Reduces join pair candidates from all nodes and returns the one with
             * the minimum distance to master and all working nodes.
             * @param candidate The current working node's join pair candidate.
             * @return The globally best join pair candidate.
             */
            auto algorithm::reduce(joinable& candidate) -> joinable
            {
                #if !__msa(runtime, cython)
                    static auto mpiop = mpi::op::create<joinable>(closest);
                    return mpi::allreduce(candidate, mpiop);
                #else
                    return candidate;
                #endif
            }
        }
    }
}