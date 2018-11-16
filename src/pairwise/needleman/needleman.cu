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
#include "pairwise/needleman.cuh"

namespace needleman = pairwise::needleman;

/**
 * Alias for scoring table type. This helps increasing the code readability.
 * @since 0.1.alpha
 */
using ScoringTable = int8_t[25][25];

/*
 * Declaring constant variables.
 */
constexpr const uint16_t blockSize = pw_threads_per_block;
constexpr const uint16_t batchSize = pw_alignment_batches * blockSize;
constexpr const uint16_t colSize = 2 * batchSize;

/*
 * Dynamically allocated shared memory pointer. This variable has its contents
 * allocated dynamic at kernel call runtime.
 */
extern __shared__ pairwise::Score line[];

/**
 * Aligns a slice of a sequence.
 * @param decoded The decoded target sequence slice to align.
 * @param table The scoring table to use.
 * @param penalty The gap penalty.
 * @param done The previous value in line.
 * @param left The previous value in column.
 * @param letter The letter to update the LCS line with.
 * @return The value calculated in the last column.
 */
__device__ pairwise::Score sliceAlignment
    (   const uint8_t *decoded
    ,   const ScoringTable table
    ,   const int8_t penalty
    ,   pairwise::Score done
    ,   pairwise::Score left
    ,   const uint8_t letter                )
{
    threaddecl(blockId, threadId);

    pairwise::Score val;

    // To align the slice with a block of the other sequence, we use the whole warp to
    // calculate part of its line. The threads work in a diagonal fashion upon the line.
    // The batch size shall be large enough so the waiting times at the start and finish
    // of the method is worth it.
    for(int32_t sliceOffset = -threadId; sliceOffset < batchSize; ++sliceOffset)
        if(sliceOffset >= 0) {
            uint8_t actual = decoded[sliceOffset];
            bool active = (actual != pairwise::endl) || (letter != pairwise::endl);

            // The new value of the line is obtained from the maximum value between the
            // previous line or column plus the corresponding penalty or gain.
            val = max(  active ? done              + table[actual][letter] : done
            ,     max(  active ? left              - penalty               : left
            ,           active ? line[sliceOffset] - penalty               : line[sliceOffset]
            ));

            done = line[sliceOffset];
            left = line[sliceOffset] = val;
        }

    __syncthreads();

    return val;
}

__device__ uint16_t offsetLine(const pairwise::Score column[])
{
    return 240;
}

/**
 * Aligns two sequences and returns the similarity between them.
 * @param one The first sequence to align.
 * @param two The second sequence to align.
 * @param table The scoring table to use.
 * @param penalty The gap penalty.
 * @return The similarity between sequences.
 */
__device__ pairwise::Score fullAlignment
    (   const pairwise::dSequenceSlice& one
    ,   const pairwise::dSequenceSlice& two
    ,   const ScoringTable table
    ,   const int8_t penalty                    )
{
    threaddecl(blockId, threadId);

    __shared__ pairwise::Score column[colSize];
    __shared__ uint8_t decoded[batchSize];
    
    pairwise::Score done, left;
    size_t columnOffset = 0, lineOffset = 0, lineOffsetDiff = 0;

    // The 0-th column and line of the alignment matrix must be initialized by using
    // successive gap penalties. As the columns will not be recalculated each iteration,
    // we must then manually calculate its initial state.
    #pragma unroll
    for(size_t i = threadId; i < colSize; i += blockSize)
        column[i] = - penalty * (i + 1);

    __syncthreads();

    while(columnOffset < one.getLength()) {
        // Every slice will take into account the values obtained by the previous slice.
        // This will allow us to simulate parts of the alignment matrix without actually
        // calculating all of its parts. Thus saving processing time and memory.
        done = lineOffsetDiff > 0
            ? column[lineOffsetDiff - 1]
            : column[0] + penalty;

        // For each slice, we need to set up the first row of the alignment matrix. We
        // will achieve this by using the value obtained from the line offset of the last
        // slice. From this value, the penalty gap is applied to complete the row. Also,
        // here we decode the slice of sequence that will be processed in this iteration.
        #pragma unroll
        for(size_t i = threadId; i < batchSize; i += blockSize) {
            decoded[i] = columnOffset + i < one.getLength()
                ? one[columnOffset + i]
                : pairwise::endl;
            line[i] = done - penalty * (i + 1);
        }

        __syncthreads();

        // For each line we must first calculate the value of its 0-th column. After the
        // first iteration, the line might be offset, so we use the last known column
        // value to calculate the values for unknown lines.
        #pragma unroll
        for(size_t i = threadId; i < colSize; i += blockSize) {
            done = lineOffsetDiff + i <= 0
                ? column[0] + penalty
                : lineOffsetDiff + i - 1 < colSize
                    ? column[lineOffsetDiff + i - 1]
                    : column[colSize - 1] - penalty * (lineOffsetDiff + i - colSize);

            left = lineOffsetDiff + i < colSize
                ? column[lineOffsetDiff + i]
                : column[colSize - 1] - penalty * (lineOffsetDiff + i - colSize + 1);

            uint8_t letter = lineOffset + i < two.getLength()
                ? two[lineOffset + i]
                : pairwise::endl;

            column[i] = sliceAlignment(decoded, table, penalty, done, left, letter);
        }

        __syncthreads();

        // Finally, we must recalculate the offsets. The line offset may not vary in a constant
        // fashion. We try to keep the best alignment values in the middle of the of out window.
        lineOffsetDiff = offsetLine(column);
        columnOffset += batchSize;
        lineOffset += lineOffsetDiff;
    }

    __syncthreads();

    // The final result is the result obtained by the last column of the last line in the
    // alignment matrix. If the shorter sequence has not been thoroughly processed, we add up
    // gap penalties to its total, according to the number of unprocessed characters.
    int32_t gaps = two.getLength() - lineOffset - colSize;
    return column[colSize - 1] - penalty * max(gaps, 0);
}

/** 
 * Performs the needleman sequence aligment algorithm in parallel.
 * @param in The input data requested by the algorithm.
 * @param out The output data produced by the algorithm.
 */
__launch_bounds__(pw_threads_per_block, 4)
__global__ void needleman::run(needleman::Input in, Buffer<pairwise::Score> out)
{
    threaddecl(blockId, threadId);

    if(blockId < in.pair.getSize()) {
        const dSequenceSlice& first  = in.sequence[in.pair[blockId].first ];
        const dSequenceSlice& second = in.sequence[in.pair[blockId].second];

        // We must make sure that, if the sequences have different lenghts, the first
        // sequence must be longer than the second. This will allow the alignment
        // heuristic used to be more accurate.
        pairwise::Score result = fullAlignment(
            first.getSize() > second.getSize() ? first  : second
        ,   first.getSize() > second.getSize() ? second : first
        ,   in.table.getRaw()
        ,   in.penalty
        );

        onlythread(0) out.getBuffer()[blockId] = result;
    }
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
        pairmem = 2 * sizeof(pairwise::dSequenceSlice) + sizeof(Block) * (
            (usedSequence[missing[i].first ] ? 0 : this->list[missing[i].first ].getSize())
        +   (usedSequence[missing[i].second] ? 0 : this->list[missing[i].second].getSize())
        ) + sizeof(pairwise::Workpair) + sizeof(pairwise::Score);

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
 * @param in The input structure.
 */
Buffer<pairwise::Score> deviceLoad
    (   const pairwise::SequenceList& list
    ,   const std::vector<pairwise::Workpair>& pair
    ,   const std::set<ptrdiff_t>& selected
    ,   needleman::Input& in                         )
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

    device::malloc(gpuScore, sizeof(pairwise::Score) * pairList.size());
    device::malloc(gpuPairs, sizeof(pairwise::Workpair) * pairList.size());
    device::memcpy(gpuPairs, pairList.data(), sizeof(pairwise::Workpair) * pairList.size());

    in.pair = {gpuPairs, pairList.size(), device::free<pairwise::Workpair>};
    in.sequence = list.select(usedSequence).compress().toDevice();

    return {gpuScore, pairList.size(), device::free<pairwise::Score>};
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
        std::set<ptrdiff_t> pair = this->select(missing);

        needleman::Input in = {this->penalty, this->table};
        Buffer<pairwise::Score> out = deviceLoad(this->list, this->pair, pair, in);

        // Here, we call our kernel and allocate the batch size to our Needleman-Wunsch line buffer.
        // We recommend that the batch size be exactly 480 as this is a *magic* number: it is a
        // multiple of both the number of characters in a sequence block and the number of threads
        // in a warp. Block size must not be higher than 32.
        needleman::run<<<pair.size(), blockSize, sizeof(pairwise::Score) * batchSize>>>(in, out);

        for(auto i = pair.rbegin(); i != pair.rend(); ++i)
            missing.erase(missing.begin() + *i);

        device::sync();
        //this->recover(pair, out);
    }
}
