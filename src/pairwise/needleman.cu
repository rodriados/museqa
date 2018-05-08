/** @file needleman.cu
 * @brief Parallel Multiple Sequence Alignment needleman file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda.h>

#include "msa.h"
#include "gpu.hpp"
#include "pairwise.cuh"

/** @fn void pairwise::needleman(needleman_t, score_t[])
 * @brief Performs the parallel pairwise algorithm.
 * @param in The input data requested by the algorithm.
 * @param out The output data produced by the algorithm.
 */
__global__ void
__launch_bounds__(__msa_threads_per_block__)
pairwise::needleman(needleman_t in, score_t out[])
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= in.npair)
        return;
    
    __debugd("thread %2d: (%2ux%2u)", id, in.pair[id].seq[0], in.pair[id].seq[1]);
}
