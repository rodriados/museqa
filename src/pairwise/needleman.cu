/** @file needleman.cu
 * @brief Parallel Multiple Sequence Alignment needleman file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <stdio.h>
#include <cuda.h>

#include "msa.h"
#include "pairwise.hpp"
#include "needleman.cuh"

namespace pairwise
{
__global__ void needleman(char *data, position_t *seq, workpair_t *pair, score_t *score)
{
    int myId = blockIdx.x * blockDim.x + threadIdx.x;
    __debugd("block(%02d) - thread(%02d) > %d %d", blockIdx.x, threadIdx.x, pair[myId].seq[0], pair[myId].seq[1]);
}

}