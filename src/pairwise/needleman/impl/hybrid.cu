/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Hybrid implementation for the pairwise module's needleman algorithm.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include <map>
#include <set>
#include <vector>
#include <cstdint>

#include "cuda.cuh"
#include "node.hpp"
#include "utils.hpp"
#include "buffer.hpp"
#include "encoder.hpp"
#include "pointer.hpp"
#include "database.hpp"
#include "sequence.hpp"
#include "exception.hpp"

#include "pairwise/database.cuh"
#include "pairwise/pairwise.cuh"
#include "pairwise/needleman/needleman.cuh"

namespace
{
    using namespace museqa;
    using namespace pairwise;

    /**
     * Identifies a pair of sequences to process as a unit.
     * @since 0.1.1
     */
    struct job
    {
        pair payload;           /// The target pair of sequence identifiers to be processed.
        ptrdiff_t cache_offset; /// The cache offset available for processing this pair.
    };

    /**
     * Groups all required input values up for module execution.
     * @since 0.1.1
     */
    struct input
    {
        pairwise::database db;  /// The database of sequences available for alignment.
        buffer<score> cache;    /// The allocated cache for all current jobs.
        buffer<job> jobs;       /// The list of jobs to process.
    };

    /*
     * Algorithm configuration parameters. These values interfere directly into
     * the algorithm's execution, thus, they shall be modified with caution.
     */
    enum : size_t { num_batch = 15 };
    enum : size_t { block_size = cuda::warp_size * 5 };
    enum : size_t { batch_size = cuda::warp_size * num_batch };

    /*
     * Dynamically allocated shared memory pointer. This variable has its contents
     * allocated dynamic at kernel call runtime.
     */
    extern __shared__ volatile score line[];

    /**
     * Aligns a sequence to a slice of the other one.
     * @param offset The slice's first column offset.
     * @param column The columns' cache input and output.
     * @param one The sequence to be aligned with slice.
     * @param two The decoded target sequence slice to align.
     * @param table The scoring table to use.
     */
    __device__ void align_slice(
            int offset
        ,   score *__restrict__ column
        ,   const sequence_view& one
        ,   const encoder::unit *two
        ,   const scoring_table& table
        )
    {
        score last_value;
        size_t last_line = static_cast<size_t>(~0);

        // We will align each slice of the second sequence with the entirety of
        // the first sequence. The 0-th column of each slice will be obtained from
        // the last column of the previous slice. For the first slice, the 0-th
        // column was previously calculated.
        for(size_t line_offset = 0; line_offset < one.length(); line_offset += blockDim.x) {
            const size_t current_line = line_offset + threadIdx.x;

            encoder::unit unit[2];
            score done, left, value;

            // Checks whether the currently selected line is valid and initializes
            // the variables for its processing.
            if(current_line < one.length()) {
                done = current_line ? column[current_line - 1] : line[0] + table.penalty();
                left = column[current_line];
                unit[0] = one[current_line];
            }

            __syncthreads();

            // Saves the previously processed line's value in the column's cache.
            // We only do this here to not infere with the accesses done above.
            if(last_line < one.length()) {
                column[last_line] = last_value;
            }

            // To align the slice with a block of the other sequence, we use the
            // whole thread block to calculate part of its line. The threads work
            // in a diagonal fashion upon the line. The size of processing batches
            // shall be large enough so the waiting times at the start and finish
            // of the method become minimal or negligible.
            for(size_t slice_offset = 0; slice_offset < batch_size + blockDim.x; ++slice_offset) {
                const size_t current_column = slice_offset - threadIdx.x;

                if(current_line < one.length() && 0 <= current_column && current_column < batch_size) {
                    // If the column to be processed at the moment represents the
                    // end of sequence, then there is nothing left to do.
                    if((unit[1] = two[current_column]) != sequence::padding) {
                        value = line[current_column];

                        // The new value of the line is obtained from the maximum
                        // value between the previous line or column plus the corresponding
                        // penalty or gain. If the line represents the end of sequence,
                        // the line will be simply copied.
                        if(unit[0] != sequence::padding) {
                            const auto insertd = left - table.penalty();
                            const auto removed = value - table.penalty();
                            const auto matched = done + table[{unit[0], unit[1]}];
                            value = utils::max(matched, utils::max(insertd, removed));
                        }

                        done = line[current_column];
                        left = line[current_column] = value;
                    }
                }

                __syncthreads();
            }

            // Saves the current value and line index so it can be saved on the
            // column cache in the next algorithm's step.
            last_line = current_line;
            last_value = value;
        }

        // Checks whether there still are values to be saved in the column cache
        // after the conclusion of the slice's execution.
        if(last_line < one.length()) {
            column[last_line] = last_value;
        }
    }

    /**
     * Aligns two sequences using Needleman-Wunsch algorithm.
     * @param one The first sequence to align.
     * @param two The second sequence to align.
     * @param table The scoring table to use.
     * @param column A global memory cache for storing column values.
     * @return The alignment score.
     */
    __device__ score align_pair(
            const sequence_view& one
        ,   const sequence_view& two
        ,   const scoring_table& table
        ,   score *__restrict__ column
        )
    {
        __shared__ encoder::unit decoded[batch_size];

        // The 0-th column and line of the alignment matrix must be initialized
        // by using successive gap penalties. As the columns will not be recalculated
        // each iteration, we must then manually calculate its initial state.
        for(size_t line_offset = threadIdx.x; line_offset < one.length(); line_offset += blockDim.x)
            column[line_offset] = (line_offset + 1) * -table.penalty();

        __syncthreads();

        // Iterates over the shortest sequence. Using the classic Needleman-Wunsch
        // table drawing as an analogy, this sequence will be put in the horizontal
        // axis, and thus, its elements' indeces are the table's columns'.
        for(size_t column_offset = 0; column_offset < two.length(); column_offset += batch_size) {

            // For each slice, we need to set up the 0-th line of the alignment matrix.
            // We achieve this by calculating the penalties represented in the line.
            // Also, as we are already iterating over the line, we decode the slice
            // of the sequence that will be processed in this iteration.
            #pragma unroll
            for(size_t i = threadIdx.x; i < batch_size; i += blockDim.x) {
                decoded[i] = (column_offset + i) < two.length()
                    ? two[column_offset + i]
                    : sequence::padding;
                line[i] = (column_offset + i + 1) * -table.penalty();
            }

            __syncthreads();

            // We will align each slice of the first sequence with the entirety of
            // the second sequence. The 0-th column of each slice will be obtained
            // from the last column of the previous slice. For the first slice, the
            // 0-th column was previously calculated.
            align_slice(column_offset, column, one, decoded, table);

            __syncthreads();
        }

        // The final result is the result obtained in the last column of the last
        // line in the alignment matrix. As the algorithm stops at sequence end,
        // no extra penalties are needed.
        return threadIdx.x == 0 ? column[one.length() - 1] : 0;
    }

    /**
     * Performs the Needleman-Wunsch sequence aligment algorithm in parallel.
     * @param in The input data requested by the algorithm.
     * @param out The output data produced by the algorithm.
     * @param table The scoring table to use in alignment.
     */
    __launch_bounds__(block_size)
    __global__ void align_kernel(input in, buffer<score> out, const scoring_table table)
    {
        __shared__ scoring_table::raw_type mem_table;
        __shared__ scoring_table shared_table;

        // Let's use our special scoring table constructor, available only in device
        // code, to copy the scoring table into shared memory.
        new (&shared_table) scoring_table {pointer<decltype(mem_table)>::weak(&mem_table), table};

        for(size_t i = blockIdx.x; i < in.jobs.size(); i += gridDim.x) {
            const sequence_view& one = in.db[in.jobs[i].payload.id[0]];
            const sequence_view& two = in.db[in.jobs[i].payload.id[1]];

            // We must make sure that, if the sequences have different lengths,
            // the first sequence is bigger than the second. This will allow the
            // alignment method to fully use its allocated cache.
            auto result = align_pair(
                    one.size() > two.size() ? one : two
                ,   one.size() > two.size() ? two : one
                ,   shared_table
                ,   &in.cache[in.jobs[i].cache_offset]
                );

            // We don't need every spawned thread to write the result to the output,
            // as only the first one contains the requested result.
            if(threadIdx.x == 0) {
                out[i] = result;
            }
        }
    }

    /**
     * Calculates the memory usage for a given work case.
     * @param db The sequences available for alignment.
     * @param used The map of already "loaded" sequences.
     * @param target The work case's target pair.
     * @param pair_cache The cache size required by the work case.
     * @return The total memory requested for this work case execution.
     */
    static size_t required_memory(
            const museqa::database& db
        ,   const std::set<ptrdiff_t>& used
        ,   const pair& target
        ,   size_t pair_cache
        )
    {
        // Calculating the total amount of memory required for cache while processing
        // the current work pair. The cache consists of a score type array.
        size_t total_mem = sizeof(score) * pair_cache;

        // The amount of memory required by the sequences themselves are defined
        // by their sizes and whether they are already "loaded" or not.
        total_mem += sizeof(encoder::block) * (
                (used.find(target.id[0]) == used.end()) * db[target.id[0]].contents.size()
            +   (used.find(target.id[1]) == used.end()) * db[target.id[1]].contents.size()
            );

        // Besides the cache and the sequences, a work pair also requires memory
        // for its final result and describing structures' instances.
        return total_mem + 2 * sizeof(sequence_view) + sizeof(job) + sizeof(score);
    }

    /**
     * Calculates the total amount of cache needed for processing the given pair.
     * @param cache_size The amount of cache allocated previously for the block.
     * @param db The sequences available for alignment.
     * @param target The work case's target pair.
     * @return The additional amount of cache needed for processing the pair.
     */
    static size_t needed_cache(const size_t cache_size, const museqa::database& db, const pair& target)
    {
        auto total = utils::max(db[target.id[0]].contents.length(), db[target.id[1]].contents.length());
        return utils::max(total - cache_size, 0UL);
    }

    /**
     * Loads all necessary data to device, so that the selected jobs can start.
     * @param db The sequences available for alignment.
     * @param used The map of already "loaded" sequences.
     * @param jobs The batch's target jobs.
     * @param cache_size The total cache size these jobs require.
     * @return The new loaded input instance.
     */
    static input load_input(
            const museqa::database& db
        ,   const std::set<ptrdiff_t>& used
        ,   const buffer<job>& jobs
        ,   const size_t cache_size
        )
    {
        input target;
        const size_t count = jobs.size();

        std::map<ptrdiff_t, seqref> transform;
        auto jobs_buffer = buffer<job>::copy(jobs);
        seqref index = 0;

        for(const auto& selected : used)
            transform[selected] = index++;

        for(size_t i = 0; i < count; ++i)
            jobs_buffer[i].payload = {
                transform[jobs[i].payload.id[0]]
            ,   transform[jobs[i].payload.id[1]]
            };

        target.db = pairwise::database(db.only(used)).to_device();
        target.jobs = buffer<job>::make(cuda::allocator::device, count);
        target.cache = buffer<score>::make(cuda::allocator::device, cache_size);

        cuda::memory::copy(target.jobs.raw(), jobs_buffer.raw(), count);

        return target;
    }

    /**
     * Prepares the input for the algorithm's device kernel.
     * @param pairs The sequence pairs to align.
     * @param db The sequences available for alignment.
     * @param done The number of already processed pairs.
     * @return Input object instance with the selected pairs.
     */
    static input make_input(const buffer<pair>& pairs, const museqa::database& db, const size_t done)
    {
        const size_t count = pairs.size();

        size_t job_count  = 0;
        size_t mem_limit  = cuda::device::free_memory();
        size_t max_blocks = cuda::device::blocks(count);

        auto sequences    = std::set<ptrdiff_t>();
        auto block_cache  = std::vector<size_t>(max_blocks);

        for(size_t i = done, n = 0; i < count; ++i, n = (n + 1) % max_blocks) {
            // As this block might have already been used to calculate a previous
            // pair, we should try reusing its old cache.
            size_t pair_cache = needed_cache(block_cache[n], db, pairs[i]);
            size_t pair_mem = required_memory(db, sequences, pairs[i], pair_cache);

            // If the amount of memory requested by the current pair is not available,
            // then our input is already in its full capacity. We assume sequences
            // are roughly the same size, so trying to find another pair that might
            // fit in the remaining space will be mostly unsuccessful.
            if(pair_mem > mem_limit)
                break;

            sequences.insert(pairs[i].id[0]);
            sequences.insert(pairs[i].id[1]);

            block_cache[n] += pair_cache;
            mem_limit -= pair_mem;
            ++job_count;
        }

        size_t cache_offset = 0;
        auto jobs = buffer<job>::make(job_count);

        // Calculates the cache offset for each kernel block's use. This is essential
        // to avoid cache collisions while processing pairs simultaneously.
        for(size_t i = 0; i < max_blocks; ++i) {
            auto length = block_cache[i];
            block_cache[i] = cache_offset;
            cache_offset += length;
        }

        // Builds the jobs instances, by assigning each job a pair to process and
        // a cache offset to work with, which relates to the block it'll run in.
        for(size_t i = 0; i < job_count; ++i)
            jobs[i] = {pairs[done + i], block_cache[i]};

        return load_input(db, sequences, jobs, cache_offset);
    }

    /**
     * Executes the hybrid Needleman-Wunsch algorithm for the pairwise step.
     * @param pairs The sequence pairs to align in the current node.
     * @param db The sequences available for alignment.
     * @param table The scoring table to use.
     * @return The score of aligned pairs.
     */
    static auto align(const buffer<pair>& pairs, const museqa::database& db, const scoring_table& table)
    -> buffer<score>
    {
        const size_t count = pairs.size();
        auto result = buffer<score>::make(count);

        cuda::kernel::preference(align_kernel, cuda::cache::shared);

        for(size_t done = 0; done < count; ) {
            input in = make_input(pairs, db, done);
            auto out = buffer<score>::make(cuda::allocator::device, in.jobs.size());

            enforce(in.jobs.size(), "not enough memory in device");
            size_t blocks = cuda::device::blocks(in.jobs.size());

            // Here, we call our kernel and allocate our Needleman-Wunsch line buffer
            // in shared memory. We recommend that the batch size be a multiple
            // of both the number of characters in an encoded sequence block and
            // the number of threads in a warp. You can change the block size by
            // tweaking the *block_size* configuration value.
            align_kernel<<<blocks, block_size, sizeof(score) * batch_size>>>(in, out, table);
            cuda::memory::copy(result.raw() + done, out.raw(), in.jobs.size());
            done += in.jobs.size();
        }

        return result;
    }

    /**
     * The hybrid needleman algorithm object. This algorithm uses hybrid
     * parallelism to run the Needleman-Wunsch algorithm.
     * @since 0.1.1
     */
    struct hybrid : public needleman::algorithm
    {
        /**
         * Executes the hybrid needleman algorithm for the pairwise step. This method is
         * responsible for distributing and gathering workload from different cluster nodes.
         * @param context The algorithm's context.
         * @return The module's result value.
         */
        auto run(const context& ctx) const -> distance_matrix override
        {
            buffer<score> result;
            size_t nsequences = ctx.db.count();

            onlyslaves {
                auto pairs = this->generate(nsequences);
                const scoring_table table = ctx.table.to_device();
                result = align(pairs, ctx.db, table);
            }

            return distance_matrix {this->gather(result), nsequences};
        }
    };
}

namespace museqa
{
    /**
     * Instantiates a new hybrid needleman instance.
     * @return The new algorithm instance.
     */
    extern auto pairwise::needleman::hybrid() -> pairwise::algorithm *
    {
        return new ::hybrid;
    }
}
