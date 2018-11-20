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
constexpr const uint16_t batchSize = blockSize;//pw_alignment_batches * blockSize;

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
    ,   const uint8_t letter,bool t                )
{
    threaddecl(blockId, threadId);

    uint8_t actual;
    pairwise::Score val = left;

    // To align the slice with a block of the other sequence, we use the whole warp to
    // calculate part of its line. The threads work in a diagonal fashion upon the line.
    // The batch size shall be large enough so the waiting times at the start and finish
    // of the method is worth it.
    for(int32_t sliceOffset = -threadId; sliceOffset < batchSize; ++sliceOffset)
        if(sliceOffset >= 0) {

            // If the column to be processed at the moment represents the end of sequence,
            // then there is nothing left to do in here.
            if((actual = decoded[sliceOffset]) == pairwise::endl)
                break;

            // The new value of the line is obtained from the maximum value between the
            // previous line or column plus the corresponding penalty or gain. If the
            // line represents the end of sequence, the line will be simply copied.
            val = letter != pairwise::endl
                ? max(done + table[actual][letter], max(left - penalty, line[sliceOffset] - penalty))
                : line[sliceOffset];

            done = line[sliceOffset];
            left = line[sliceOffset] = val;
        }

    return val;
}

/**
 * Aligns two sequences and returns the similarity between them.
 * @param one The first sequence to align.
 * @param two The second sequence to align.
 * @param column A global memory buffer.
 * @param table The scoring table to use.
 * @param penalty The gap penalty.
 * @return The similarity between sequences.
 */
__device__ pairwise::Score fullAlignment
    (   const pairwise::dSequenceSlice& one
    ,   const pairwise::dSequenceSlice& two
    ,   pairwise::Score * const column
    ,   const ScoringTable table
    ,   const int8_t penalty                    )
{
    threaddecl(blockId, threadId);

    __shared__ uint8_t decoded[batchSize];

    // The 0-th column and line of the alignment matrix must be initialized by using
    // successive gap penalties. As the columns will not be recalculated each iteration,
    // we must then manually calculate its initial state.
    for(size_t lineOffset = threadId; lineOffset < two.getLength(); lineOffset += blockSize)
        column[lineOffset] = - penalty * (lineOffset + 1);

    __syncthreads();

    for(size_t columnOffset = 0; columnOffset < one.getLength(); columnOffset += batchSize) {
        // For each slice, we need to set up the 0-th line of the alignment matrix. We
        // achieve this by calculating the penalties represented in the line. Also, as
        // we are already iterating over the line, we decode the slice of sequence that
        // will be processed in this iteration.
        #pragma unroll
        for(size_t i = threadId; i < batchSize; i += blockSize) {
            decoded[i] = columnOffset + i < one.getLength()
                ? one[columnOffset + i]
                : pairwise::endl;
            line[i] = - penalty * (columnOffset + i + 1);
        }

        __syncthreads();

        // We will align each slice of the first sequence with the entirety of the second
        // sequence. The 0-th column of each slice will be obtained from the last column
        // of the previous slice. For the first slice, the 0-th column is pre-calculated.
        for(size_t lineOffset = threadId; lineOffset < two.getLength(); lineOffset += blockSize) {
            pairwise::Score done = lineOffset > 0
                ? column[lineOffset - 1]
                : - penalty * columnOffset;    
            pairwise::Score left = column[lineOffset];
            uint8_t letter = two[lineOffset];

            column[lineOffset] = sliceAlignment(decoded, table, penalty, done, left, letter, !lineOffset);
        }

        __syncthreads();
    }

    // The final result is the result obtained by the last column of the last line in the
    // alignment matrix. As the algorithm stops at sequence end, no extra penalties are needed.
    return column[two.getLength() - 1];
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
        size_t cacheSize = in.glcache.getSize() / in.pair.getSize();

        // We must make sure that, if the sequences have different lenghts, the first
        // sequence is be shorter than the second. This will allow the alignment method
        // to execute faster and use less resources if the difference is significant.
        const dSequenceSlice& first  = in.sequence[in.pair[blockId].first ];
        const dSequenceSlice& second = in.sequence[in.pair[blockId].second];

        pairwise::Score result = fullAlignment(
            first.getSize() > second.getSize() ? second : first
        ,   first.getSize() > second.getSize() ? first : second
        ,   &in.glcache.getBuffer()[cacheSize * blockId]
        ,   in.table.getRaw()
        ,   in.penalty
        );

        onlythread(0) out.getBuffer()[blockId] = result;
    }
}

/**
 * Selects pairs to be processed in device.
 * @param missing The pairs still available to be processed.
 * @param selected The selected pair indeces.
 * @return The maximum length of selected sequences.
 */
size_t pairwise::Needleman::select(std::vector<pairwise::Workpair>& missing, std::set<ptrdiff_t>& selected) const
{
    const DeviceProperties& dprop = device::properties();

    size_t pairmem, totalmem = dprop.totalGlobalMem - 2048;
    size_t maxlen = 0, maxpairs = dprop.maxGridSize[0];

    std::vector<bool> usedSequence(this->list.getCount(), false);
    selected.clear();
    
    // We must know the length of the longest sequence still waiting to be aligned. The
    // length is needed so we can allocate enough memory for our alignment method to work.
    // With this measure, we recommend that sequences have approximately the same length,
    // so very little to no resources are wasted.
    for(pairwise::Workpair& pair : missing)
        maxlen = max(maxlen, max(
            this->list[pair.first ].getLength()
        ,   this->list[pair.second].getLength()
        ));

    // We must also calculate the amount of memory requested by an alignment to occur. We
    // limit the number of parallel alignments on the amount of global memory available.
    for(size_t i = 0, n = missing.size(); i < n && selected.size() < maxpairs; ++i) {
        pairmem = 2 * sizeof(pairwise::dSequenceSlice) + sizeof(Block) * (
            (usedSequence[missing[i].first ] ? 0 : this->list[missing[i].first ].getSize())
        +   (usedSequence[missing[i].second] ? 0 : this->list[missing[i].second].getSize())
        ) + sizeof(pairwise::Workpair) + sizeof(pairwise::Score) * (maxlen + 1);

        if(pairmem < totalmem) {
            usedSequence[missing[i].first ] = true;
            usedSequence[missing[i].second] = true;
            selected.insert(i);
            totalmem -= pairmem;
        }
    }

    return maxlen;
}

/**
 * Loads input data to device.
 * @param list The list of available sequences.
 * @param pair The list of all pairs to process.
 * @param selected The indeces of pairs to process at the moment.
 * @param maxlen The maximum length of a sequence.
 * @param in The input structure.
 */
Buffer<pairwise::Score> deviceLoad
    (   const pairwise::SequenceList& list
    ,   const std::vector<pairwise::Workpair>& pair
    ,   const std::set<ptrdiff_t>& selected
    ,   const size_t maxlen
    ,   needleman::Input& in                         )
{
    std::set<ptrdiff_t> usedSequence;

    for(ptrdiff_t i : selected) {
        usedSequence.insert(pair[i].first );
        usedSequence.insert(pair[i].second);
    }

    pairwise::Workpair *gpuPairs;
    pairwise::Score *gpuScore, *gpuBuffer;
    std::vector<pairwise::Workpair> pairList;

    for(ptrdiff_t i : selected)
        pairList.push_back({*usedSequence.find(pair[i].first), *usedSequence.find(pair[i].second)});

    device::malloc(gpuScore, sizeof(pairwise::Score) * pairList.size());
    device::malloc(gpuBuffer, sizeof(pairwise::Score) * pairList.size() * maxlen);
    device::malloc(gpuPairs, sizeof(pairwise::Workpair) * pairList.size());
    device::memcpy(gpuPairs, pairList.data(), sizeof(pairwise::Workpair) * pairList.size());

    in.pair = {gpuPairs, pairList.size(), device::free<pairwise::Workpair>};
    in.glcache = {gpuBuffer, pairList.size() * maxlen, device::free<pairwise::Score>};
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
    std::set<ptrdiff_t> selected;
    std::vector<pairwise::Workpair> missing = this->pair;

#ifdef pw_prefer_shared_mem
    cudacall(cudaFuncSetCacheConfig(needleman::run, cudaFuncCachePreferShared));
#else
    cudacall(cudaFuncSetCacheConfig(needleman::run, cudaFuncCachePreferL1));
#endif

    while(!missing.empty()) {
        size_t maxlen = this->select(missing, selected);
        needleman::Input in = {this->penalty, this->table};
        Buffer<pairwise::Score> out = deviceLoad(this->list, this->pair, selected, maxlen, in);

        // Here, we call our kernel and allocate our Needleman-Wunsch line buffer in shared memory.
        // We recommend that the batch size be exactly 480 as this is a *magic* number: it is a
        // multiple of both the number of characters in a sequence block and the number of threads
        // in a warp. Block size must not be higher than 32.
        needleman::run<<<selected.size(), blockSize, sizeof(pairwise::Score) * batchSize>>>(in, out);

        for(auto i = selected.rbegin(); i != selected.rend(); ++i)
            missing.erase(missing.begin() + *i);

        device::sync();
        //this->recover(selected, out);
    }
}
