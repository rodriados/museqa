/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the pairwise module's needleman-wunsch algorithm.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include "museqa.hpp"

#include "mpi.hpp"
#include "oeis.hpp"
#include "utils.hpp"
#include "buffer.hpp"
#include "exception.hpp"
#include "environment.h"

#include "pairwise/pairwise.cuh"
#include "pairwise/needleman/needleman.cuh"

namespace museqa
{
    namespace pairwise
    {
        namespace needleman
        {
            /**
             * Generates all working pairs that will be processed by the current
             * node rank for a given number of total sequences.
             * @param num The total number of sequences.
             * @return The generated sequence pairs.
             */
            auto algorithm::generate(size_t num) const -> buffer<pair>
            {
                #if !defined(__museqa_runtime_cython)
                    enforce(node::rank >= 1, "master node must not generate pairs");

                    const auto total = utils::nchoose(num);
                    const auto range = utils::partition(total, node::count - 1, node::rank - 1);

                    auto pairs = buffer<pair>::make(range.total);
                    
                    size_t i = oeis::a002024(range.offset + 1);
                    size_t j = range.offset - utils::nchoose(i);

                    for(size_t c = 0; c < range.total; ++i, j = 0)
                        while(c < range.total && j < i)
                            pairs[c++] = pair {seqref(i), seqref(j++)};

                    return pairs;
                #else
                    return pairwise::algorithm::generate(num);
                #endif
            }

            /**
             * Gathers all calculated scores from all processes to master.
             * @param input The buffer with the current node's results.
             * @return The gathered score from all processes.
             */
            auto algorithm::gather(buffer<score>& input) const -> buffer<score>
            {
                #if !defined(__museqa_runtime_cython)
                    return mpi::allgather(input);
                #else
                    return input;
                #endif
            }

            /**
             * Picks the default needleman algorithm instance according to the executions's
             * global state conditions and devices availability.
             * @return The picked algorithm instance.
             */
            auto best() -> pairwise::algorithm *
            {
                if (node::count > 1 || global_state.mpi_running) {
                    return global_state.use_devices
                        ? needleman::hybrid()
                        : needleman::sequential();
                } else {
                    return global_state.local_devices > 0
                        ? needleman::hybrid()
                        : needleman::sequential();
                }
            }
        }
    }
}
