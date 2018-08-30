/**
 * Multiple Sequence Alignment pairwise parallelization distribution file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>
#include <vector>

#include "msa.hpp"
#include "cluster.hpp"
#include "pairwise/needleman.cuh"

void pairwise::Needleman::scatter()
{
    std::vector<uint32_t> sendCount(cluster::size, 0);
    std::vector<uint32_t> sendDispl(cluster::size, 0);

    onlymaster {
        uint32_t each = this->pwise.getCount() / (cluster::size - 1);
        uint32_t more = this->pwise.getCount() % (cluster::size - 1);

        for(uint16_t i = 1; i < cluster::size; ++i) {
            sendCount[i] = each + (more >= i);
            sendDispl[i] = sendDispl[i - 1] + sendCount[i - 1];
        }
    }

    cluster::broadcast(sendCount.data(), cluster::size);
    cluster::broadcast(sendDispl.data(), cluster::size);
    cluster::sync();

    onlyslaves {
        this->pairs.resize(sendCount[cluster::rank]);
    }

    //cluster::scatterv<uint32_t>(this->pairs.data(), sendCount, sendDispl, this->pairs.data(), sendCount[cluster::size]);
    cluster::sync();
}


void pairwise::Needleman::gather() const
{

}