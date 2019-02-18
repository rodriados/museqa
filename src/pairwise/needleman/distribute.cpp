/**
 * Multiple Sequence Alignment parallel needleman distribution file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <algorithm>

#include "mpi.hpp"
#include "buffer.hpp"

#include "pairwise/pairwise.cuh"
#include "pairwise/needleman.cuh"

using namespace pairwise;

/**
 * Scatters generated workpairs from master to all other processes.
 * @param origin All pairs to be scatter. Significant only at master.
 * @return The pairs the current node is responsible for processing.
 */
Buffer<Pair> pairwise::needleman::scatter(Buffer<Pair>& origin)
{
    Buffer<Pair> destiny;

    size_t count = 0;
    size_t displ = 0;

    size_t npair = origin.getSize();
    mpi::broadcast(npair);

    onlyslaves {
        const size_t quo = npair / (node::size - 1);
        const size_t rem = npair % (node::size - 1);

        const size_t rank = node::rank - 1;

        count = quo + (rem > rank);
        displ = quo * rank + std::min(rank, rem);
    }

    mpi::scatter(origin, destiny, count, displ);

    return destiny;
}

/**
 * Gathers all calculated scores from all processes to master.
 * @param origin The scores to be sent from current process to master.
 * @return The gathered score from all processes.
 */
Buffer<Score> pairwise::needleman::gather(Buffer<Score>& origin)
{
    Buffer<Score> destiny;
    mpi::gather(origin, destiny);
    return destiny;
}
