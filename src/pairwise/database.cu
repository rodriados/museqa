/**
 * Multiple Sequence Alignment pairwise database file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <vector>

#include <cuda.cuh>
#include <encoder.hpp>
#include <pointer.hpp>
#include <database.hpp>
#include <sequence.hpp>

#include <pairwise/database.cuh>

/**
 * Sets up the view pointers responsible for keeping track of internal sequences.
 * @param seq The merged sequence to have its internal parts splitted.
 * @param db The database to have its sequences mapped.
 */
auto pairwise::database::init(underlying_type& merged, const ::database& db) -> entry_buffer
{
    const size_t count = db.count();
    auto result = entry_buffer::make(count);

    for(size_t i = 0, j = 0; i < count; ++i) {
        result[i] = {merged, ptrdiff_t(j), db[i].size()};
        j += db[i].size();
    }

    return result;
}

/**
 * Merges all sequences in given database to a single contiguous sequence.
 * @param db The database to have its sequences merged.
 * @return The merged sequences blocks.
 */
auto pairwise::database::merge(const ::database& db) -> underlying_type
{
    const size_t count = db.count();
    std::vector<encoder::block> merged;

    for(size_t i = 0; i < count; ++i)
        merged.insert(merged.end(), db[i].begin(), db[i].end());

    return underlying_type::copy(merged);
}

/**
 * Transfers this database instance to the compute-capable device.
 * @return The database instance allocated in device.
 */
auto pairwise::database::to_device() const -> pairwise::database
{
    const size_t nblocks = this->size();
    const size_t nviews = this->count();

    auto dblocks = underlying_type::make(cuda::memory::global<encoder::block[]>(), nblocks);
    auto dviews = entry_buffer::make(cuda::memory::global<element_type[]>(), nviews);
    auto helper = entry_buffer::make(nviews);

    for(size_t i = 0; i < nviews; ++i)
        helper[i] = {dblocks, mviews[i]};

    cuda::memory::copy(dblocks.raw(), this->raw(), nblocks);
    cuda::memory::copy(dviews.raw(), helper.raw(), nviews);

    return {dblocks, dviews};
}
