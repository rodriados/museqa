/**
 * Multiple Sequence Alignment parallel needleman communication file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#include <mpi.hpp>
#include <utils.hpp>
#include <buffer.hpp>

#include <environment.h>

#include <pairwise/pairwise.cuh>
#include <pairwise/algorithm/needleman.cuh>

namespace msa
{
    namespace pairwise
    {
        /**
         * Scatters generated workpairs from master to all other processes.
         * @return The pairs the current node is responsible for processing.
         */
        auto needleman::algorithm::scatter(buffer<pair>& pairs) const -> buffer<pair>
        {
            onlymaster return pairs;

            #if !defined(__msa_runtime_cython)
                const size_t total    = pairs.size();

                const size_t quotient   = total / (node::count - 1);
                const size_t remainder  = total % (node::count - 1);
                const size_t rank       = node::rank - 1;

                const size_t count    = quotient + (remainder > rank);
                const ptrdiff_t displ = quotient * rank + utils::min(rank, remainder);

                return slice_buffer<pair> {pairs, displ, count};
            #else
                return pairs;
            #endif
        }

        /**
         * Gathers all calculated scores from all processes to master.
         * @param input The buffer with the current node's results.
         * @return The gathered score from all processes.
         */
        auto needleman::algorithm::gather(buffer<score>& input) const -> buffer<score>
        {
            #if !defined(__msa_runtime_cython)
                return mpi::allgather(input);
            #else
                return input;
            #endif
        }
    }
}
