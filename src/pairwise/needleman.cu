/** @file needleman.cu
 * @brief Parallel Multiple Sequence Alignment needleman file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>
#include <vector>
#include <cuda.h>

#include "msa.h"
#include "pairwise.cuh"

#define MAX_THREADS_PER_BLOCK 32

__global__ void
__launch_bounds__(MAX_THREADS_PER_BLOCK)
pairwise::needleman(needleman_t in, score_t *out)
{
    __debugd("__global__ pairwise::needleman(%d, %d)", sizeof(in), sizeof(out));
}

void needleman_t::alloc(std::vector<uint32_t>& pairs)
{

}

void needleman_t::free()
{

}