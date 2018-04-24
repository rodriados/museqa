/** @file pairwise.cu
 * @brief Parallel Multiple Sequence Alignment pairwise file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>
#include <cstdlib>
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
inline N align(N size, N align)
{
    return (size / align + !!(size % align)) * align;
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
    needleman_t indata;

    __cudacheck(cudaSetDevice(gpu = gpu::assign()));
    __cudacheck(cudaGetDeviceProperties(&dprop, gpu));

    this->scatter();
    this->filter();
    this->blosum(indata);
    this->run(indata);

    __cudacheck(cudaFree(indata.table));
}

/** @fn void pairwise_t::filter()
 * @brief Filters the sequences needed for processor.
 */
void pairwise_t::filter()
{
    uint16_t *map = new uint16_t [this->nseq] ();

    uint32_t length = 0;
    uint16_t count = 0;

    for(int i = 0; i < this->npair; ++i) {
        map[this->pair[i].seq[0]] = 1;
        map[this->pair[i].seq[1]] = 1;
    }

    for(int i = 0; i < this->nseq; ++i)
        if(map[i] != 0) {
            this->seq[count].offset = this->seq[i].offset;
            this->seq[count].length = this->seq[i].length;
            map[i] = count++;
        }

    for(int i = 0; i < this->npair; ++i) {
        this->pair[i].seq[0] = map[this->pair[i].seq[0]];
        this->pair[i].seq[1] = map[this->pair[i].seq[1]];
    }

    delete[] map;

    this->nseq = count;
}

/** @fn void pairwise_t::run(needleman_t&)
 * @brief Executes the pairwise algorithm.
 */
void pairwise_t::run(needleman_t& indata)
{
    score_t *outdata;

#ifdef __msa_use_shared_mem_for_temp_storage__
    __cudacheck(cudaFuncSetCacheConfig(pairwise::needleman, cudaFuncCachePreferShared));
#else
    __cudacheck(cudaFuncSetCacheConfig(pairwise::needleman, cudaFuncCachePreferL1));
#endif

    indata.nseq = this->nseq;
    indata.npair = this->npair;

    __cudacheck(cudaMalloc(&indata.seq, sizeof(position_t) * indata.nseq));
    __cudacheck(cudaMalloc(&indata.pair, sizeof(workpair_t) * indata.npair));
    __cudacheck(cudaMalloc(&outdata, sizeof(score_t) * indata.npair));

    this->score = new score_t [this->npair] ();

    // Put sequences and pairs into GPU's global memory according to the available space.
    // Memcpy only the sequences for the selected pairs to execute.

    pairwise::needleman<<<1,1>>>(indata, outdata);
    __cudacheck(cudaThreadSynchronize());

    __cudacheck(cudaFree(indata.seq));
    __cudacheck(cudaFree(indata.pair));
    __cudacheck(cudaFree(outdata));
}
