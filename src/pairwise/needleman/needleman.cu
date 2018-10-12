/**
 * Multiple Sequence Alignment needleman file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */ 
#include <set>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "msa.hpp"
#include "buffer.hpp"
#include "device.cuh"
#include "pointer.hpp"
#include "pairwise/pairwise.hpp"
#include "pairwise/sequence.cuh"
#include "pairwise/needleman.cuh"

namespace needleman = pairwise::needleman;

/**
 * Executes one step of Longest Common Subsequence algorithm.
 * @param target The sequence to be inspected.
 * @param line The current LCS table line.
 * @param size The size of LCS line.
 * @param table The scoring table to use.
 * @param letter The letter to update the LCS line.
 */
__device__ void linestep
    (   const pairwise::dSequenceSlice& target
    ,   pairwise::Score *line
    ,   size_t size
    ,   int8_t table[25][25]
    ,   uint8_t letter                          )
{
    pairwise::Score top, left, done = line[0];
    pairwise::Score gap = table[24][0];

    line[0] += gap;

    for(size_t b = 0, i = 1; b < size; ++b) {
        Block block = target.getBlock(b);

        #pragma unroll
        for(uint8_t j = 0; j < 6; ++i, ++j) {
            top  = line[i];
            left = line[i - 1];

            line[i] = max(
                done + table[pairwise::decode(block, j)][letter],
                max(top + gap, left + gap)
            );

            done = top;
        }
    }
}

/**
 * Aligns two sequences and returns the similarity between them.
 * @param one The first sequence to align.
 * @param two The second sequence to align.
 * @param table The scoring table to use.
 * @return The similarity between sequences.
 */
__device__ pairwise::Score align
    (   const pairwise::dSequenceSlice& one
    ,   const pairwise::dSequenceSlice& two
    ,   int8_t table[25][25]                    )
{
    size_t lineLength = one.getLength();
    pairwise::Score *line = new pairwise::Score[lineLength + 1]();

    for(size_t i = 1; i <= lineLength; ++i)
        line[i] = line[i - 1] + table[0][24];

    for(size_t i = 0, n = two.getSize(); i < n; ++i) {
        Block block = two.getBlock(i);

        #pragma unroll
        for(uint8_t j = 0; j < 6; ++j)
            linestep(one, line, one.getSize(), table, pairwise::decode(block, j));
    }

    pairwise::Score maximum = ~0;

    for(size_t i = 1; i <= lineLength; ++i)
        if(line[i] > maximum)
            maximum = line[i];

    return maximum;
}

/** 
 * Performs the needleman sequence aligment algorithm in parallel.
 * @param in The input data requested by the algorithm.
 * @param out The output data produced by the algorithm.
 */
__launch_bounds__(pw_threads_per_block)
__global__ void needleman::run(needleman::Input in, Buffer<pairwise::Score> out)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= in.pair.getSize())
        return;

    size_t first  = in.sequence[in.pair[idx].first ].getSize();
    size_t second = in.sequence[in.pair[idx].second].getSize();

    out.getBuffer()[idx] = align(
        in.sequence[first < second ? in.pair[idx].first : in.pair[idx].second]
    ,   in.sequence[first < second ? in.pair[idx].second : in.pair[idx].first]
    ,   in.table.getRaw()
    );
}

/**
 * Selects pairs to be processed in device.
 * @param missing The pairs still available to be processed.
 * @return The selected pair indeces.
 */
std::set<ptrdiff_t> pairwise::Needleman::select(std::vector<pairwise::Workpair>& missing) const
{
    size_t totalmem = device::properties().totalGlobalMem * .75 - 1000;
    size_t pairmem, maxsize = 0;

    std::set<ptrdiff_t> selectedPairs;
    std::vector<bool> usedSequence(this->list.getCount(), false);

    for(size_t i = 0, n = missing.size(); i < n; ++i) {
        pairmem = sizeof(pairwise::Workpair) + sizeof(pairwise::Score) + sizeof(Block) * (
            (usedSequence[missing[i].first ] ? 0 : this->list[missing[i].first ].getSize())
        +   (usedSequence[missing[i].second] ? 0 : this->list[missing[i].second].getSize())
        );

        maxsize = std::max(maxsize, std::max(
            this->list[missing[i].first ].getSize()
        ,   this->list[missing[i].second].getSize()
        ));

        if(pairmem + (selectedPairs.size() + 1) * maxsize * sizeof(pairwise::Score) < totalmem) {
            usedSequence[missing[i].first ] = true;
            usedSequence[missing[i].second] = true;
            selectedPairs.insert(i);
            totalmem -= pairmem;
        }
    }

    return selectedPairs;
}

/**
 * Loads input data to device.
 * @param list The list of available sequences.
 * @param pair The list of all pairs to process.
 * @param selected The indeces of pairs to process at the moment.
 * @param input The input structure.
 */
void gpuLoad
    (   const pairwise::SequenceList& list
    ,   const std::vector<pairwise::Workpair>& pair
    ,   const std::set<ptrdiff_t>& selected
    ,   needleman::Input& input
    ,   Buffer<pairwise::Score>& output                   )
{
    std::set<ptrdiff_t> usedSequence;


    for(ptrdiff_t i : selected) {
        usedSequence.insert(pair[i].first );
        usedSequence.insert(pair[i].second);
    }

    pairwise::Score *gpuScore;
    pairwise::Workpair *gpuPairs;
    std::vector<pairwise::Workpair> pairList;
    std::vector<ptrdiff_t> sequences(usedSequence.begin(), usedSequence.end());

    for(ptrdiff_t i : selected)
        pairList.push_back({*usedSequence.find(pair[i].first), *usedSequence.find(pair[i].second)});

    cudacall(cudaMalloc(&gpuScore, sizeof(pairwise::Score) * pairList.size()));
    cudacall(cudaMalloc(&gpuPairs, sizeof(pairwise::Workpair) * pairList.size()));
    cudacall(cudaMemcpy(gpuPairs, pairList.data(), sizeof(pairwise::Workpair) * pairList.size(), cudaMemcpyHostToDevice));

    output = {gpuScore, pairList.size(), device::deleter<pairwise::Score>};
    input.pair = {gpuPairs, pairList.size(), device::deleter<pairwise::Workpair>};
    input.sequence = list.select(sequences).compress().toDevice();
}

/**
 * Executes the needleman algorithm for the pairwise step. This method
 * is responsible for distributing and gathering workload from different
 * cluster nodes.
 */
void pairwise::Needleman::run()
{
    std::set<ptrdiff_t> current;
    std::vector<pairwise::Workpair> missing = this->pair;

#ifdef pw_prefer_shared_mem
    cudacall(cudaFuncSetCacheConfig(needleman::run, cudaFuncCachePreferShared));
#else
    cudacall(cudaFuncSetCacheConfig(needleman::run, cudaFuncCachePreferL1));
#endif

    while(!missing.empty()) {
        Buffer<pairwise::Score> output;
        needleman::Input input;

        input.table = this->table;
        current = this->select(missing);        
        gpuLoad(this->list, this->pair, current, input, output);

        size_t blocks = ceil(current.size() / 32.0);
        needleman::run<<<blocks, 32>>>(input, output);

        for(auto i = current.rbegin(); i != current.rend(); ++i)
            missing.erase(missing.begin() + *i);

        cudacall(cudaThreadSynchronize());
        //this->recover(current, output);
    }
}
