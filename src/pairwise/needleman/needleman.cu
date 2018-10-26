/**
 * Multiple Sequence Alignment needleman file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */ 
#include <set>
#include <vector>

#include "msa.hpp"
#include "buffer.hpp"
#include "device.cuh"
#include "pointer.hpp"
#include "pairwise/pairwise.hpp"
#include "pairwise/sequence.cuh"
#include "pairwise/needleman.cuh"

namespace needleman = pairwise::needleman;

/**
 * Alias for scoring table type. This helps increasing the code readability.
 * @since 0.1.alpha
 */
using ScoringTable = int8_t[25][25];

/*
 * Dynamically allocated shared memory pointer. This variable has its contents
 * allocated dynamic at kernel call runtime.
 */
extern __shared__ volatile pairwise::Score line[];

/*
 * Declaring constant variables.
 */
constexpr const uint8_t batch = pw_batches_per_block;
constexpr const uint8_t warp = pw_threads_per_block;

/*
 * Declaring global variables.
 */
__device__ uint32_t blockId;
__device__ uint8_t threadId;

/**
 * Aligns a slice of the sequences.
 * @param decoded The decoded target sequence to align.
 * @param table The scoring table to use.
 * @param penalty The gap penalty.
 * @param letter The letter to update the LCS line.
 */
__device__ void slice
    (   const uint8_t *decoded
    ,   const ScoringTable table
    ,   const int8_t penalty
    ,   const uint8_t letter        )
{
    pairwise::Score prev, actual;

    for(uint8_t i = -threadId; i < batch * warp; ++i) {
        if(i < 0) continue;

        actual = (i == 0)
            ? line[i] + penalty
            : max(prev + table[letter][decoded[i]], max(line[i - 1] + penalty, line[i] + penalty));

        prev = line[i];
        line[i] = actual;
    }

    __syncthreads();
}

/**
 * Aligns two sequences and returns the similarity between them.
 * @param one The first sequence to align.
 * @param two The second sequence to align.
 * @param table The scoring table to use.
 * @param penalty The gap penalty.
 * @return The similarity between sequences.
 */
__device__ pairwise::Score align
    (   const pairwise::dSequenceSlice& one
    ,   const pairwise::dSequenceSlice& two
    ,   const ScoringTable table
    ,   const int8_t penalty                        )
{
    __shared__ uint8_t decoded[batch * warp];

    #pragma unroll
    for(uint16_t i = threadId; i < batch * warp; i += warp) {
        decoded[i] = one.getLength() > i
            ? one[i]
            : pairwise::endl;
        line[i] = penalty * i;
    }

    __syncthreads();

    for(size_t i = threadId, n = two.getLength(); i < n; i += warp)
        slice(decoded, table, penalty, two[i]);

    __syncthreads();

    return line[0];
}

/** 
 * Performs the needleman sequence aligment algorithm in parallel.
 * @param input The input data requested by the algorithm.
 * @param output The output data produced by the algorithm.
 */
__launch_bounds__(pw_threads_per_block)
__global__ void needleman::run(needleman::Input input, Buffer<pairwise::Score> output)
{
    blockId = blockIdx.y * gridDim.x + blockIdx.x;
    threadId = threadIdx.y * blockDim.x + threadIdx.x;

    output.getBuffer()[blockId] = align(
        input.sequence[input.pair[blockId].first ]
    ,   input.sequence[input.pair[blockId].second]
    ,   input.table.getRaw()
    ,   input.penalty
    );
}

/**
 * Selects pairs to be processed in device.
 * @param missing The pairs still available to be processed.
 * @return The selected pair indeces.
 */
std::set<ptrdiff_t> pairwise::Needleman::select(std::vector<pairwise::Workpair>& missing) const
{
    const DeviceProperties& dprop = device::properties();
    size_t pairmem, totalmem = dprop.totalGlobalMem - 2048;
    size_t maxpairs = dprop.maxGridSize[0];

    std::set<ptrdiff_t> selectedPairs;
    std::vector<bool> usedSequence(this->list.getCount(), false);

    for(size_t i = 0, n = missing.size(); i < n && selectedPairs.size() < maxpairs; ++i) {
        pairmem = sizeof(pairwise::Workpair) + sizeof(pairwise::Score) + sizeof(Block) * (
            (usedSequence[missing[i].first ] ? 0 : this->list[missing[i].first ].getSize())
        +   (usedSequence[missing[i].second] ? 0 : this->list[missing[i].second].getSize())
        );

        if(pairmem < totalmem) {
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
Buffer<pairwise::Score> gpuLoad
    (   const pairwise::SequenceList& list
    ,   const std::vector<pairwise::Workpair>& pair
    ,   const std::set<ptrdiff_t>& selected
    ,   needleman::Input& input                         )
{
    std::set<ptrdiff_t> usedSequence;

    for(ptrdiff_t i : selected) {
        usedSequence.insert(pair[i].first );
        usedSequence.insert(pair[i].second);
    }

    pairwise::Score *gpuScore;
    pairwise::Workpair *gpuPairs;
    std::vector<pairwise::Workpair> pairList;

    for(ptrdiff_t i : selected)
        pairList.push_back({*usedSequence.find(pair[i].first), *usedSequence.find(pair[i].second)});

    cudacall(cudaMalloc(&gpuScore, sizeof(pairwise::Score) * pairList.size()));
    cudacall(cudaMalloc(&gpuPairs, sizeof(pairwise::Workpair) * pairList.size()));
    cudacall(cudaMemcpy(gpuPairs, pairList.data(), sizeof(pairwise::Workpair) * pairList.size(), cudaMemcpyHostToDevice));

    input.pair = {gpuPairs, pairList.size(), device::deleter<pairwise::Workpair>};
    input.sequence = list.select(usedSequence).compress().toDevice();

    return {gpuScore, pairList.size(), device::deleter<pairwise::Score>};
}

/**
 * Executes the needleman algorithm for the pairwise step. This method
 * is responsible for distributing and gathering workload from different
 * cluster nodes.
 */
void pairwise::Needleman::run()
{
    std::vector<pairwise::Workpair> missing = this->pair;

#ifdef pw_prefer_shared_mem
    cudacall(cudaFuncSetCacheConfig(needleman::run, cudaFuncCachePreferShared));
#else
    cudacall(cudaFuncSetCacheConfig(needleman::run, cudaFuncCachePreferL1));
#endif

    while(!missing.empty()) {
        std::set<ptrdiff_t> current = this->select(missing);
        needleman::Input input = {this->table, -5};
        Buffer<pairwise::Score> output = gpuLoad(this->list, this->pair, current, input);

        debug("round of %d", input.pair.getSize());

        // We set out Needleman-Wunsch line to exactly 480 positions, because this is a 
        // *magic* number, as it ends up being a multiple of both the number of characters
        // in a block and the number of threads in a warp.
        needleman::run<<<current.size(), warp, sizeof(pairwise::Score) * batch * warp>>>(input, output);

        for(auto i = current.rbegin(); i != current.rend(); ++i)
            missing.erase(missing.begin() + *i);

        cudacall(cudaThreadSynchronize());
        //this->recover(current, output);
    }
}
