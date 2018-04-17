/** @file pairwise.cu
 * @brief Parallel Multiple Sequence Alignment pairwise file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cuda.h>

#include "msa.h"
#include "gpu.hpp"
#include "fasta.hpp"

#include "pairwise.hpp"

pairwise_t pairdata;

namespace pairwise
{
void prepare()
{
    short *map = new short[pairdata.nseq]();
    short *off = new short[pairdata.nseq]();
    unsigned length = 0;
    short used = 0;

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

    char *gpuseq;
    __cudacheck(cudaSetDevice(gpu::assign()));
    __cudacheck(cudaMalloc((void **)&gpuseq, sizeof(char) * length));

    for(int i = 0; i < used; ++i)
        __cudacheck(cudaMemcpy(
            &gpuseq[pairdata.seq[i].offset],
            &pairdata.data[off[i]],
            pairdata.seq[i].length,
            cudaMemcpyHostToDevice
        ));

    delete[] pairdata.data;
    delete[] map;
    delete[] off;

    pairdata.nseq = used;
    pairdata.data = gpuseq;
}

void pairwise()
{

}

}