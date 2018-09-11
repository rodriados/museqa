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
    uint16_t t = 0;

    for(uint16_t i = 0, n = this->pwise.getList().getCount(); i < n; ++i)
        for(uint16_t j = i + 1; j < n; ++j)
            this->pairs.push_back({i, j});

    onlymaster debug("generated %d sequence pairs", this->pairs.size());
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
__launch_bounds__(pw_threads_per_block)
pairwise::needleman(pairwise::dNeedleman in, pairwise::Score *out)
{}