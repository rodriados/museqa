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
    
    free(this->seqchar);
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

    this->generate();
    this->blosum(in);
    this->run(in);

    __cudacheck(cudaFree(in.table));
}

/** @fn void pairwise_t::generate()
 * @brief Generates all sequence workpairs for slave nodes.
 */
void pairwise_t::generate()
{
    int i, wp, count = 0;
    int nproc = mpi_data.size - 1;
    int nrank = mpi_data.rank - 1;

    int div = this->npair / nproc;
    int rem = this->npair % nproc;

    this->npair = div + (nrank < rem);
    this->pair  = new workpair_t [this->npair];

    for(i = 0, wp = 0; i < this->nseq; ++i)
        for(int j = i + 1; j < this->nseq; ++j, ++wp)
            if(wp % nproc == nrank) {
                this->pair[count].seq[0] = i;
                this->pair[count].seq[1] = j;
                ++count;
            }
}

/** @fn bool pairwise_t::select(bool[], std::vector<uint32_t>&) const
 * @brief Selects sequences to be processed based on memory constraints.
 * @param control Indicates which pairs have already been processed.
 * @param pairs Stores the selected pairs to process.
 */
bool pairwise_t::select(bool control[], std::vector<uint32_t>& pairs) const
{
    uint32_t pairmem, npair = 0, maxsize = 0, msz;
    uint32_t totalmem = dprop.totalGlobalMem;

    uint16_t *used = new uint16_t [this->nseq] ();

    for(int i = 0; i < this->npair; ++i)
        if(!control[i]) {
            // For each workpair to process, at least two sequences and an output score are needed.
            pairmem = sizeof(workpair_t) + 2*sizeof(position_t) + sizeof(score_t) + sizeof(char) *
               ((used[this->pair[i].seq[0]] ? 0 : align(this->seq[this->pair[i].seq[0]].length)) +
                (used[this->pair[i].seq[1]] ? 0 : align(this->seq[this->pair[i].seq[1]].length)));
            
            msz = max(maxsize, max(
                this->seq[this->pair[i].seq[0]].length + 1
            ,   this->seq[this->pair[i].seq[1]].length + 1
            ));

            if(pairmem + (npair + 1) * msz * sizeof(score_t) < totalmem) {
                used[this->pair[i].seq[0]] = 1;
                used[this->pair[i].seq[1]] = 1;

                totalmem -= pairmem;
                pairs.push_back(i);
                maxsize = msz;

                control[i] = true;
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
    score_t *out;

    this->score = new score_t [this->npair] ();

#ifdef __msa_prefer_shared_mem__
    __cudacheck(cudaFuncSetCacheConfig(pairwise::needleman, cudaFuncCachePreferShared));
#else
    __cudacheck(cudaFuncSetCacheConfig(pairwise::needleman, cudaFuncCachePreferL1));
#endif

    while(this->select(control, pairs)) {
        this->alloc(in, pairs);

        __cudacheck(cudaMalloc(&out, sizeof(score_t) * in.npair));

        short blocks  = divceil(in.npair, 32);
        short threads = min(in.npair, 32);

        pairwise::needleman<<<blocks, threads>>>(in, out);
        __cudacheck(cudaThreadSynchronize());

        this->destroy(in);
        pairs.clear();
        __cudacheck(cudaFree(out));
    }

    delete[] control;

    daemon::destroy();
}

/** @fn void pairwise_t::alloc(needleman_t&, std::vector<uint32_t>&)
 * @brief Allocates all memory needed for needleman's execution.
 * @param in The object owning all allocated pointers.
 * @param pairs The selected pairs' identifiers.
 */
void pairwise_t::alloc(needleman_t& in, std::vector<uint32_t>& pairs)
{
    std::vector<uint16_t> slist;
    uint16_t *used = new uint16_t [this->nseq] ();

    for(const uint32_t& pair : pairs) {
        if(!used[this->pair[pair].seq[0]]) {
            slist.push_back(this->pair[pair].seq[0]);
            used[this->pair[pair].seq[0]] = ++in.nseq;
        }
        
        if(!used[this->pair[pair].seq[1]]) {
            slist.push_back(this->pair[pair].seq[1]);
            used[this->pair[pair].seq[1]] = ++in.nseq;
        }

        ++in.npair;
    }

    workpair_t *wp = new workpair_t [in.npair] ();

    for(int i = 0; i < in.npair; ++i) {
        wp[i].seq[0] = used[this->pair[pairs.at(i)].seq[0]] - 1;
        wp[i].seq[1] = used[this->pair[pairs.at(i)].seq[1]] - 1;
    }

    __cudacheck(cudaMalloc(&in.pair, sizeof(workpair_t) * in.npair));
    __cudacheck(cudaMemcpy(in.pair, wp, sizeof(workpair_t) * in.npair, cudaMemcpyHostToDevice));

    this->allocseq(in, slist);

    delete[] used;
    delete[] wp;
}

/** @fn void pairwise_t::allocseq(needleman_t&, std::vector<uint16_t>&)
 * @brief Allocates memory needed for storing sequences and copies them.
 * @param in The object owning all allocated pointers.
 * @param slist The list of sequences to copy.
 */
void pairwise_t::allocseq(needleman_t& in, std::vector<uint16_t>& slist)
{
    uint32_t length = 0;
    std::vector<uint16_t> request;

    for(int i = 0; i < in.nseq; ++i) {
        length += align(this->seq[slist[i]].length);

        if(this->seq[slist[i]].offset == ~0)
        	request.push_back(slist[i]);
    }

    if(request.size())
    	this->request(request);

    char *data = new char [length] ();
    position_t *pos = new position_t [in.nseq] ();

    for(int i = 0, length = 0; i < in.nseq; ++i) {
        pos[i].offset = length;
        pos[i].length = this->seq[slist[i]].length;

        memcpy(data + length, this->seqchar + this->seq[slist[i]].offset, sizeof(char) * pos[i].length);
        length += align(pos[i].length);
    }

    __cudacheck(cudaMalloc(&in.seqchar, sizeof(char) * length));
    __cudacheck(cudaMalloc(&in.seq, sizeof(position_t) * in.nseq));

    __cudacheck(cudaMemcpy(in.seqchar, data, sizeof(char) * length, cudaMemcpyHostToDevice));
    __cudacheck(cudaMemcpy(in.seq, pos, sizeof(position_t) * in.nseq, cudaMemcpyHostToDevice));

    delete[] pos;
    delete[] data;
}

/** @fn void pairwise_t::request(std::vector<uint16_t>&)
 * @brief Requests the sequences needed from master node.
 * @param seqlist The list of needed sequences.
 */
void pairwise_t::request(std::vector<uint16_t>& seqlist)
{
    int bfsize;

    position_t *position = new position_t [seqlist.size()];
    char *buffer = daemon::request(seqlist, position, bfsize);

    this->seqchar = (char *)realloc(this->seqchar, sizeof(char) * (this->clength + bfsize));
    memcpy(this->seqchar + this->clength, buffer, bfsize);

    for(int i = 0; i < nseq; ++i) {
        this->seq[seqlist.at(i)].offset = this->clength + position[i].offset;
        this->seq[seqlist.at(i)].length = position[i].length;
    }

    this->clength += bfsize;

    delete[] buffer;
    delete[] position;
}

/** @fn void pairwise_t::destroy(needleman_t&) const
 * @brief Frees all alloc'd memory for executing needleman.
 * @param in The object containing the references to clean.
 */
void pairwise_t::destroy(needleman_t& in) const
{
    __cudacheck(cudaFree(in.seq));
    __cudacheck(cudaFree(in.pair));
    __cudacheck(cudaFree(in.seqchar));

    in.nseq = in.npair = 0;
}