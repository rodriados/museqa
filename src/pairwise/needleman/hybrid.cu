/**
 * Multiple Sequence Alignment hybrid needleman file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <set>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>

#include "msa.hpp"
#include "cuda.cuh"
#include "node.hpp"
#include "buffer.hpp"
#include "pointer.hpp"
#include "encoder.hpp"
#include "database.hpp"
#include "exception.hpp"

#include "pairwise/database.cuh"
#include "pairwise/pairwise.cuh"
#include "pairwise/needleman.cuh"

using namespace pairwise;

/*
 * Algorithm configuration values. These values interfere directly into
 * the algorithm's execution, thus, they shall be modified with caution.
 */
namespace {
    enum : uint32_t { nw_batches = 15 };
    enum : uint32_t { nw_max_device_memory_usage = 80 };
    enum : uint32_t { nw_threads_block = cuda::warpSize };
    enum : bool     { nw_prefer_shared = true };
};

static constexpr int32_t blockSize = nw_threads_block;
static constexpr int32_t batchSize = nw_threads_block * nw_batches;

/**
 * Identifies a work unit to process.
 * @since 0.1.1
 */
struct Workpair
{
    Pair pair;                  /// The pair's sequence identifiers.
    uint32_t cacheOffset;       /// The cache offset available for this work unit.
};

/**
 * Groups up all required input values for module execution.
 * @since 0.1.1
 */
struct Input
{
    pairwise::Database db;      /// The database of sequences available for alignment.
    Buffer<Workpair> jobs;      /// The work units to process.
    Buffer<Score> cache;        /// The allocated cache for all current work units.
};

// TODO: Whenever a thread finishes a line calculation, make it execute the first not
// yet calculated line.

/*
 * Dynamically allocated shared memory pointer. This variable has its contents
 * allocated dynamic at kernel call runtime.
 */
extern __shared__ Score line[];

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
__device__ Score alignSlice
    (   const uint8_t *decoded
    ,   const Pointer<ScoringTable>& table
    ,   const int8_t penalty
    ,   Score done
    ,   Score left
    ,   const uint8_t letter                )
{
    uint8_t actual;
    Score val = left;

    // To align the slice with a block of the other sequence, we use the whole thread block
    // to calculate part of its line. The threads work in a diagonal fashion upon the line.
    // The size of batch shall be large enough so the waiting times at the start and finish
    // of the method is worth it.
    for(int32_t sliceOffset = -threadIdx.x; sliceOffset < batchSize; ++sliceOffset)
        if(sliceOffset >= 0) {
            // If the column to be processed at the moment represents the end of sequence,
            // then there is nothing left to do in here.
            if((actual = decoded[sliceOffset]) == encoder::end)
                break;

            // The new value of the line is obtained from the maximum value between the
            // previous line or column plus the corresponding penalty or gain. If the line
            // represents the end of sequence, the line will be simply copied.
            val = letter != encoder::end
                ? max(done + (*table)[actual][letter], max(left - penalty, line[sliceOffset] - penalty))
                : line[sliceOffset];

            done = line[sliceOffset];
            left = line[sliceOffset] = val;
        }

    return val;
}

/**
 * Aligns two sequences and returns the score of similarity between them.
 * @param one The first sequence to align.
 * @param two The second sequence to align.
 * @param column A global memory cache for storing column values.
 * @param table The scoring table to use.
 * @param penalty The gap penalty.
 * @return The score between sequences.
 */
__device__ Score alignComplete
    (   const SequenceSlice& one
    ,   const SequenceSlice& two
    ,   Score * const column
    ,   const Pointer<ScoringTable>& table
    ,   const int8_t penalty                )
{
    __shared__ uint8_t decoded[batchSize];

    // The 0-th column and line of the alignment matrix must be initialized by using
    // successive gap penalties. As the columns will not be recalculated each iteration,
    // we must then manually calculate its initial state.
    for(size_t lineOffset = threadIdx.x, n = one.getLength(); lineOffset < n; lineOffset += blockSize)
        column[lineOffset] = - penalty * (lineOffset + 1);

    __syncthreads();

    for(size_t columnOffset = 0; columnOffset < two.getLength(); columnOffset += batchSize) {
        // For each slice, we need to set up the 0-th line of the alignment matrix. We
        // achieve this by calculating the penalties represented in the line. Also, as
        // we are already iterating over the line, we decode the slice of the sequence
        // taht will be processed in this iteration.
        #pragma unroll
        for(size_t i = threadIdx.x; i < batchSize; i += blockSize) {
            decoded[i] = columnOffset + i < two.getLength()
                ? two[columnOffset + i]
                : encoder::end;
            line[i] = - penalty * (columnOffset + i + 1);
        }

        __syncthreads();

        // We will align each slice of the first sequence with the entirety of the second
        // sequence. The 0-th column of each slice will be obtained from the last column
        // of the previous slice. For the first slice, the 0-th column was previously calculated.
        for(size_t lineOffset = threadIdx.x, n = one.getLength(); lineOffset < n; lineOffset += blockSize) {
            Score done = lineOffset > 0
                ? column[lineOffset - 1]
                : - penalty * columnOffset;
            Score left = column[lineOffset];
            uint8_t letter = one[lineOffset];

            column[lineOffset] = alignSlice(decoded, table, penalty, done, left, letter);
        }

        __syncthreads();
    }

    // The final result is the result obtained by the last column of the last line in the
    // alignment matrix. As the algorithm stops at sequence end, no extra penalties are needed.
    return column[one.getLength() - 1];
}

/** 
 * Performs the needleman sequence aligment algorithm in parallel.
 * @param table The scoring table to use in alignment.
 * @param in The input data requested by the algorithm.
 * @param out The output data produced by the algorithm.
 */
__launch_bounds__(nw_threads_block, 4)
__global__ void kernel(const Pointer<ScoringTable> table, Input in, Buffer<Score> out)
{
    if(blockIdx.x < in.jobs.getSize()) {
        // We must make sure that, if the sequences have different lengths, the first
        // sequence is bigger than the second. This will allow the alignment method to
        // fully use its allocated cache.
        const SequenceSlice& seq1 = in.db[in.jobs[blockIdx.x].pair.id[0]];
        const SequenceSlice& seq2 = in.db[in.jobs[blockIdx.x].pair.id[1]];

        Score result = alignComplete(
            seq1.getSize() > seq2.getSize() ? seq1 : seq2
        ,   seq1.getSize() > seq2.getSize() ? seq2 : seq1
        ,   &in.cache[in.jobs[blockIdx.x].cacheOffset]
        ,   table
        ,   -(*table)[24][0]
        );

        if(!threadIdx.x)
            out[blockIdx.x] = result;
    }
}

/**
 * Loads all necessary data to device, so that the selected jobs can readily start up.
 * @param db The sequences available for alignment.
 * @param used The map of already "loaded" sequences.
 * @param jobs The case's target workpair.
 * @param cacheSize The total size of cache this case requires.
 * @param in The target input instance to load.
 */
static void upload
    (   const ::Database& db
    ,   const std::set<ptrdiff_t>& used
    ,   const std::vector<Workpair>& jobs
    ,   const size_t cacheSize
    ,   Input& in                                   )
{
    const size_t batch = jobs.size();

    std::unordered_map<ptrdiff_t, uint16_t> map;
    Buffer<Workpair> workpairs {jobs};
    uint16_t index = 0;

    for(ptrdiff_t id : used)
        map[id] = index++;

    for(size_t i = 0; i < batch; ++i)
        workpairs[i].pair = {
            map[jobs[i].pair.id[0]]
        ,   map[jobs[i].pair.id[1]]
        };

    in.db = pairwise::Database{db, used}.toDevice();
    in.jobs = Buffer<Workpair> {cuda::allocate<Workpair>(batch), batch};
    in.cache = Buffer<Score> {cuda::allocate<Score>(cacheSize), cacheSize};

    cuda::copy<Workpair>(in.jobs.getBuffer(), workpairs.getBuffer(), batch);
}

/**
 * Calculates the memory usage for a given work case.
 * @param db The sequences available for alignment.
 * @param used The map of already "loaded" sequences.
 * @param pair The case's target pair.
 * @param cachesz The size of cache this case requires.
 * @return The total memory requested for this case execution.
 */
inline size_t calcMemUsage(const ::Database& db, const std::vector<bool> used, const Pair& pair, size_t *cachesz)
{
    *cachesz = std::max(db[pair.id[0]].getLength(), db[pair.id[1]].getLength());

    return 2 * sizeof(SequenceSlice) + sizeof(encoder::EncodedBlock) * (
            (used[pair.id[0]] ? 0 : db[pair.id[0]].getSize())
        +   (used[pair.id[1]] ? 0 : db[pair.id[1]].getSize())
        ) + sizeof(Workpair) + sizeof(Score) * (*cachesz + 1);
}

/**
 * Selects pairs to be processed in device.
 * @param db The sequences available for alignment.
 * @param pairs The sequence pairs to align.
 * @param done The number of already processed pairs.
 * @param in The target input instance for loading.
 * @return The number of workpairs loaded to device.
 */
static size_t prepare
    (   const ::Database& db
    ,   const Buffer<Pair>& pairs
    ,   const size_t done
    ,   Input& in                       )
{
    const cuda::device::Properties prop = cuda::device::getProperties();

    size_t memory = cuda::device::freeMemory() * (nw_max_device_memory_usage / 100.0) - 2048;
    size_t maxbatch = prop.maxGridSize[0];
    size_t pairmem, cachesz, batch = 0;
    size_t cacheDispl = 0;

    std::vector<Workpair> jobs;
    std::vector<bool> usedSequence(db.getCount(), false);
    std::set<ptrdiff_t> selectedSequence;

    jobs.reserve(100);

    for(size_t i = done, n = pairs.getSize(); i < n && batch < maxbatch; ++i, ++batch) {
        if((pairmem = calcMemUsage(db, usedSequence, pairs[i], &cachesz)) > memory)
            break;

        jobs.push_back({pairs[i], cacheDispl});
        usedSequence[pairs[i].id[0]] = usedSequence[pairs[i].id[1]] = true;
        selectedSequence.insert(pairs[i].id[0]);
        selectedSequence.insert(pairs[i].id[1]);

        memory -= pairmem;
        cacheDispl += cachesz;
    }

    upload(db, selectedSequence, jobs, cacheDispl, in);

    return batch;
}

/**
 * Executes the hybrid Needleman-Wunsch algorithm for the pairwise step.
 * @param db The sequences available for alignment.
 * @param pairs The workpairs to align.
 * @param table The scoring table to use.
 * @return The score of aligned pairs.
 */
static Buffer<Score> run(const ::Database& db, const Buffer<Pair>& pairs, const Pointer<ScoringTable>& table)
{
    size_t done = 0, batch = 0;
    const size_t total = pairs.getSize();

    Input in;
    Buffer<Score> score {total}, out;

    watchdog("pairwise", 0, total, "aligning pairs");
    cuda::kernel::preference(kernel, nw_prefer_shared ? cuda::cache::shared : cuda::cache::l1);
    
    while(done < total) {
        batch = prepare(db, pairs, done, in);
        out = Buffer<Score> {cuda::allocate<Score>(batch), batch};

        if(!batch) throw Exception("not all pairs fit in device memory.");

        // Here, we call our kernel and allocate our Needleman-Wunsch line buffer in shared memory.
        // We recommend that the batch size be exactly 480 a multiple of both the number of characters
        // in an encoded sequence block and the number of threads in a warp. We also recommend that
        // the block size must not be higher than a warp, but you can change this by tweaking the
        // *nw_threads_block* configuration value.
        kernel<<<batch, blockSize, sizeof(Score) * batchSize>>>(table, in, out);
        cuda::copy<Score>(&score[done], out.getBuffer(), batch);
        done += batch;

        watchdog("pairwise", done, total, "aligning pairs");
    }

    return score;
}

/**
 * Executes the needleman algorithm for the pairwise step. This method
 * is responsible for distributing and gathering workload from different
 * cluster nodes.
 */
Buffer<Score> needleman::hybrid(const Configuration& config)
{
    Buffer<Score> score;
    Buffer<Pair> pairs;

    onlymaster pairs = needleman::generate(config.db.getCount());
               pairs = needleman::scatter(pairs);

    Pointer<ScoringTable> table = scoring::toDevice(config.table);
    onlyslaves score = ::run(config.db, pairs, table);

    return needleman::gather(score);
}
