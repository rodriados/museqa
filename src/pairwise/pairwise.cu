/** @file pairwise.cu
 * @brief Parallel Multiple Sequence Alignment pairwise file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cuda.h>

#include "msa.h"
#include "gpu.hpp"
#include "fasta.hpp"
#include "interface.hpp"

#include "pairwise.hpp"
#include "needleman.cuh"

pairwise_t pairdata;

extern clidata_t cli_data;

namespace pairwise
{
/** @fn void pairwise::prepare()
 * @brief Prepares all pairwise information to execution.
 */
void prepare()
{
    short *map = new short[pairdata.nseq]();
    short *off = new short[pairdata.nseq]();
    unsigned length = 0;
    short used = 0;

    char *d_data;

    for(int i = 0; i < pairdata.npair; ++i) {
        map[pairdata.pair[i].seq[0]] = 1;
        map[pairdata.pair[i].seq[1]] = 1;
    }

    for(int i = 0; i < pairdata.nseq; ++i)
        if(map[i]) {
            off[used] = pairdata.seq[i].offset;
            pairdata.seq[used].offset = length;
            pairdata.seq[used].length = pairdata.seq[i].length;

            length += gpu::align(pairdata.seq[used].length);
            map[i] = used++;
        }

    for(int i = 0; i < pairdata.npair; ++i) {
        pairdata.pair[i].seq[0] = map[pairdata.pair[i].seq[0]];
        pairdata.pair[i].seq[1] = map[pairdata.pair[i].seq[1]];
    }

    __cudacheck(cudaSetDevice(gpu::assign()));
    __cudacheck(cudaMalloc((void **)&d_data, sizeof(char) * length));

    for(int i = 0; i < used; ++i)
        __cudacheck(cudaMemcpy(
            &d_data[pairdata.seq[i].offset]
        ,   &pairdata.data[off[i]]
        ,   pairdata.seq[i].length
        ,   cudaMemcpyHostToDevice
        ));

    delete[] pairdata.data;
    delete[] map;
    delete[] off;

    pairdata.nseq = used;
    pairdata.data = d_data;
}

/** @fn score_t *pairwise::pairwise()
 * @brief Executes the pairwise step algorithm.
 * @return The score of all pairs compared.
 */
score_t *pairwise()
{
    score_t *d_score;
    position_t *d_seq;
    workpair_t *d_pair;

    unsigned batchsz = pairdata.npair > cli_data.batchsize
        ? cli_data.batchsize
        : pairdata.npair;

    __cudacheck(cudaMalloc((void **)&d_seq, sizeof(position_t) * pairdata.nseq));
    __cudacheck(cudaMalloc((void **)&d_pair, sizeof(workpair_t) * batchsz));
    __cudacheck(cudaMalloc((void **)&d_score, sizeof(score_t) * batchsz));
    __cudacheck(cudaMemcpy(d_seq, pairdata.seq, sizeof(position_t) * pairdata.nseq, cudaMemcpyHostToDevice));

    score_t *score = new score_t[pairdata.npair]();

#ifdef __msa_use_shared_mem_for_temp_storage__
    __cudacheck(cudaFuncSetCacheConfig(needleman, cudaFuncCachePreferShared));
#else
    __cudacheck(cudaFuncSetCacheConfig(needleman, cudaFuncCachePreferL1));
#endif

    for(int done = 0; done < pairdata.npair; ) {
        __cudacheck(cudaMemcpy(d_pair, pairdata.pair + done, sizeof(workpair_t) * batchsz, cudaMemcpyHostToDevice));
        __cudacheck(cudaMemset(d_score, 0, sizeof(score_t) * batchsz));

        needleman <<<1, batchsz>>> (
            pairdata.data
        ,   d_seq
        ,   d_pair
        ,   d_score
        );

        __cudacheck(cudaThreadSynchronize());
        __cudacheck(cudaMemcpy(score + done, d_score, sizeof(score_t) * batchsz, cudaMemcpyDeviceToHost));

        done = done + batchsz;
        batchsz = pairdata.npair - done < batchsz ? pairdata.npair - done : batchsz;
    }

    __cudacheck(cudaFree(d_seq));
    __cudacheck(cudaFree(d_pair));
    __cudacheck(cudaFree(d_score));

    return score;
}

/** @fn void pairwise::clean()
 * @brief Cleans up all dynamicaly allocated data for pairwise.
 */
void clean()
{
    delete[] pairdata.seq;
    delete[] pairdata.pair;
    __cudacheck(cudaFree(pairdata.data));
}

}