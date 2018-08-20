/**
 * Multiple Sequence Alignment needleman file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <vector>

#include "msa.hpp"
#include "pairwise.hpp"
#include "pairwise/needleman.cuh"

void pairwise::Needleman::generate()
{
    for(uint16_t i = 0, n = this->pwise.getCount(); i < n; ++i)
        for(uint16_t j = i + 1; j < n; ++j)
            this->pairs.push_back({i, j});

    __onlymaster {
        __debugh("generated %d sequence pairs", this->pairs.size());
    }
}

void pairwise::Needleman::run()
{
    //pairwise::needleman<<<1,1>>>(in);
}

/** 
 * Performs the needleman sequence aligment algorithm in parallel.
 * @param in The input data requested by the algorithm.
 * @param out The output data produced by the algorithm.
 */
__global__ void
__launch_bounds__(__pw_threads_per_block__)
pairwise::needleman(pairwise::dNeedleman in, pairwise::Score *out)
{}