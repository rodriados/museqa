/** @file pairwise.cu
 * @brief Parallel Multiple Sequence Alignment pairwise file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda.h>

#include "msa.h"
#include "gpu.hpp"
#include "fasta.hpp"
#include "interface.hpp"

#include "pairwise.cuh"

static cudaDeviceProp dprop;

/** @fn N align<N>(N, N)
 * @brief Calculates the alignment for a given size.
 * @param size The size to be aligned.
 * @param align The alignment to use for given size.
 * @return The new aligned size.
 */
template <typename N>
inline N align(N size, N align = 4)
{
    return (size / align + !!(size % align)) * align;
}

/** @fn N max<N>(N, N)
 * @brief Returns the maximum between two values.
 * @param a First given value.
 * @param b Second given value.
 * @return The maximum value between a and b.
 */
template <typename N>
inline N max(N a, N b)
{
    return a > b ? a : b;
}

/** @fn pairwise_t::pairwise_t()
 * @brief Initializes the object.
 */
pairwise_t::pairwise_t()
    : nseq(0)
    , npair(0)
{
    this->seq = NULL;
    this->pair = NULL;
    this->score = NULL;
    this->seqchar = NULL;
}

/** @fn pairwise_t::~pairwise_t()
 * @brief Cleans up all dynamicaly allocated data for pairwise.
 */
pairwise_t::~pairwise_t()
{
    delete[] this->seq;
    delete[] this->pair;
    delete[] this->score;
    delete[] this->seqchar;
}

/** @fn void pairwise_t::pairwise()
 * @brief Commands pairwise execution.
 */
void pairwise_t::pairwise()
{
    int gpu;
    needleman_t in;

    __cudacheck(cudaSetDevice(gpu = gpu::assign()));
    __cudacheck(cudaGetDeviceProperties(&dprop, gpu));

    dprop.totalGlobalMem -= sizeof(uint8_t) * 625;

    this->scatter();
    this->blosum(in);
    this->run(in);

    __cudacheck(cudaFree(in.table));
}

/** @fn bool pairwise_t::filter(bool[], std::vector<uint32_t>&)
 * @brief Filters the sequences needed for processor.
 */
bool pairwise_t::filter(bool control[], std::vector<uint32_t>& pairs)
{
    uint32_t totalmem = dprop.totalGlobalMem;
    uint32_t pairmem, npair = 0, maxsize = 0;

    uint16_t *used = new uint16_t [this->nseq] ();

    for(int i = 0; i < this->npair; ++i)
        if(!control[i]) {
            // For each workpair to process, at least two sequences and an output score are needed.
            pairmem = sizeof(workpair_t) + 2*sizeof(position_t) + sizeof(score_t) + sizeof(char) *
               ((used[this->pair[i].seq[0]] ? 0 : align(this->seq[this->pair[i].seq[0]].length)) +
                (used[this->pair[i].seq[1]] ? 0 : align(this->seq[this->pair[i].seq[1]].length)));
            
            maxsize = max(maxsize, max(
                this->seq[this->pair[i].seq[0]].length + 1
            ,   this->seq[this->pair[i].seq[1]].length + 1
            ));

            if(pairmem + (npair + 1) * maxsize * sizeof(score_t) < totalmem) {
                used[this->pair[i].seq[0]] = 1;
                used[this->pair[i].seq[1]] = 1;

                totalmem -= pairmem;
                pairs.push_back(i);
                ++npair;
            }
        }

    delete[] used;
    return npair;
}

/** @fn void pairwise_t::run(needleman_t&)
 * @brief Executes the pairwise algorithm.
 * @param in The needleman input data.
 */
void pairwise_t::run(needleman_t& in)
{    
    bool *control = new bool [this->npair] ();
    std::vector<uint32_t> pairs;
    
    this->score = new score_t [this->npair] ();
    score_t *out;

#ifdef __msa_use_shared_mem_for_temp_storage__
    __cudacheck(cudaFuncSetCacheConfig(pairwise::needleman, cudaFuncCachePreferShared));
#else
    __cudacheck(cudaFuncSetCacheConfig(pairwise::needleman, cudaFuncCachePreferL1));
#endif

    while(this->filter(control, pairs)) {
        in.alloc(pairs);

        /*__cudacheck(cudaMalloc(&in.seq, sizeof(position_t) * in.nseq));
        __cudacheck(cudaMalloc(&in.pair, sizeof(workpair_t) * in.npair));
        __cudacheck(cudaMalloc(&out, sizeof(score_t) * in.npair));*/

        // Put sequences and pairs into GPU's global memory according to the available space.
        // Memcpy only the sequences for the selected pairs to execute.

        pairwise::needleman<<<1,1>>>(in, out);
        __cudacheck(cudaThreadSynchronize());

        /*__cudacheck(cudaFree(in.seq));
        __cudacheck(cudaFree(in.pair));
        __cudacheck(cudaFree(out));*/
        in.free();
    }

    delete[] control;
}
