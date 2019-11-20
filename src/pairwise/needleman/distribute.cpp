/**
 * Multiple Sequence Alignment parallel needleman distribution file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <mpi.hpp>
#include <utils.hpp>
#include <buffer.hpp>

#include <pairwise/pairwise.cuh>
#include <pairwise/needleman.cuh>

using namespace pairwise;

/**
 * Scatters generated workpairs from master to all other processes.
 * @return The pairs the current node is responsible for processing.
 */
auto needleman::algorithm::scatter() -> buffer<pair>
{
    buffer<pair>& tgt = this->pairs;

    #if !defined(onlycython)
        const size_t npairs = tgt.size();

        const size_t quo = npairs / (node::count - 1);
        const size_t rem = npairs % (node::count - 1);
        const size_t rank = node::rank - 1;

        const size_t count = quo + (rem > rank);
        const ptrdiff_t displ = quo * rank + utils::min(rank, rem);

        tgt = slice_buffer<pair> {tgt, displ, count};
    #endif

    return tgt;
}

/**
 * Gathers all calculated scores from all processes to master.
 * @param input The buffer with the current node's results.
 * @return The gathered score from all processes.
 */
auto needleman::algorithm::gather(buffer<score>& input) const -> buffer<score>
{
    buffer<score> output = input;

    #if !defined(onlycython)
        mpi::allgather(input, output);
    #endif

    return output;
}
