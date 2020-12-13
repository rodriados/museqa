/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the pairwise-specialized sequence database.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include <vector>
#include <cstdint>

#include "cuda.cuh"
#include "encoder.hpp"
#include "database.hpp"
#include "sequence.hpp"

#include "pairwise/database.cuh"

namespace museqa
{
    /**
     * Sets up the view pointers responsible for keeping track of internal sequences.
     * @param seq The merged sequence to have its internal parts splitted.
     * @param db The database to have its sequences mapped.
     */
    auto pairwise::database::init(underlying_type& merged, const museqa::database& db) -> entry_buffer
    {
        auto const count = db.count();
        auto result = entry_buffer::make(count);

        for(size_t i = 0, j = 0; i < count; ++i) {
            result[i] = sequence_view {merged, ptrdiff_t(j), db[i].contents.size()};
            j += db[i].contents.size();
        }

        return result;
    }

    /**
     * Merges all sequences in given database to a single contiguous sequence.
     * @param db The database to have its sequences merged.
     * @return The merged sequences blocks.
     */
    auto pairwise::database::merge(const museqa::database& db) -> underlying_type
    {
        std::vector<encoder::block> merged;

        for(const auto& entry : db)
            merged.insert(merged.end(), entry.contents.begin(), entry.contents.end());

        return underlying_type::copy(merged);
    }

    /**
     * Transfers this database instance to the compute-capable device.
     * @return The database instance allocated in device.
     */
    auto pairwise::database::to_device() const -> pairwise::database
    {
        const size_t total_blocks = this->size();
        const size_t total_views  = this->count();

        auto blocks = underlying_type::make(cuda::allocator::device, total_blocks);
        auto views  = entry_buffer::make(cuda::allocator::device, total_views);
        auto helper = entry_buffer::make(total_views);

        for(size_t i = 0; i < total_views; ++i)
            helper[i] = sequence_view {blocks, m_views[i]};

        cuda::memory::copy(blocks.raw(), this->raw(), total_blocks);
        cuda::memory::copy(views.raw(), helper.raw(), total_views);

        return {blocks, views};
    }
}
