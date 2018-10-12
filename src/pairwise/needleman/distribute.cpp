/**
 * Multiple Sequence Alignment pairwise parallelization distribution file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>
#include <vector>

#include "msa.hpp"
#include "buffer.hpp"
#include "cluster.hpp"
#include "pairwise/needleman.cuh"

/**
 * Scatters the workload through the working nodes.
 */
void pairwise::Needleman::scatter()
{
    std::vector<int> sendcount(cluster::size, 0);
    std::vector<int> senddispl(cluster::size, 0);

    int total = this->pair.size();
    cluster::broadcast(total);

    int each = total / (cluster::size - 1);
    int addt = total % (cluster::size - 1);

    for(uint16_t i = 1; i < cluster::size; ++i) {
        sendcount[i] = each + (addt >= i);
        senddispl[i] = senddispl[i - 1] + sendcount[i - 1];
    }

    std::vector<Workpair> buffer;    
    cluster::scatter(this->pair, buffer, sendcount, senddispl);
    
    onlyslaves this->pair = buffer;
    this->score = {new Score[this->pair.size()], this->pair.size()};
}

/**
 * Gathers the resulting data of all working nodes.
 */
void pairwise::Needleman::gather()
{
    cluster::sync();
}