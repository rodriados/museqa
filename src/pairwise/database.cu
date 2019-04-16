/**
 * Multiple Sequence Alignment pairwise database file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <set>
#include <vector>

#include "cuda.cuh"
#include "encoder.hpp"
#include "pointer.hpp"
#include "database.hpp"
#include "sequence.hpp"

#include "pairwise/database.cuh"

/**
 * Initializes a contiguous database from a common database instance.
 * @param db The database to be transformed.
 */
pairwise::Database::Database(const ::Database& db)
:   Sequence {merge(db)}
,   view {db.getCount()}
{
    init(db);
}

/**
 * Initializes a contiguous database from the subset of common database instance.
 * @param db The database to be transformed.
 * @param selected The subset of elements to be in new database.
 */
pairwise::Database::Database(const ::Database& db, const std::set<ptrdiff_t>& selected)
:   view {selected.size()}
{
    ::Database selectdb = db.only(selected);
    Sequence::operator=(merge(selectdb));

    init(selectdb);
}

/**
 * Initializes a contiguous database from the subset of common database instance.
 * @param db The database to be transformed.
 * @param selected The subset of elements to be in new database.
 */
pairwise::Database::Database(const ::Database& db, const std::vector<ptrdiff_t>& selected)
:   view {selected.size()}
{
    ::Database selectdb = db.only(selected);
    Sequence::operator=(merge(selectdb));

    init(selectdb);
}

/**
 * Transfers this database instance to the compute-capable device.
 * @return The database instance allocated in device.
 */
pairwise::Database pairwise::Database::toDevice() const
{
    const size_t dbsize = this->getSize();
    const size_t dbcount = this->getCount();

    using Block = encoder::EncodedBlock;
    using View = SequenceView;

    Buffer<Block> device_seq = Buffer<Block> {cuda::allocate<Block>(dbsize), dbsize};
    Buffer<View> device_view = Buffer<View> {cuda::allocate<View>(dbcount), dbcount};

    Buffer<View> transformed {dbcount};

    for(size_t i = 0; i < dbcount; ++i)
        transformed[i] = {device_seq, view[i]};

    cuda::copy<Block>(device_seq.getBuffer(), this->getBuffer(), dbsize);
    cuda::copy<View>(device_view.getBuffer(), transformed.getBuffer(), dbcount);

    return {device_seq, device_view};
}

/**
 * Sets up the view pointers responsible for keeping track of internal sequences.
 * @param db The database to have its sequences mapped.
 */
void pairwise::Database::init(const ::Database& db)
{
    for(size_t i = 0, j = 0, n = getCount(); i < n; ++i) {
        view[i] = {*this, static_cast<ptrdiff_t>(j), db[i].getSize()};
        j += db[i].getSize();
    }
}

/**
 * Merges all sequences in given database to a single contiguous sequence.
 * @param db The database to have its sequences merged.
 * @return The merged sequences blocks.
 */
std::vector<encoder::EncodedBlock> pairwise::Database::merge(const ::Database& db)
{
    std::vector<encoder::EncodedBlock> merged;

    for(size_t i = 0, n = db.getCount(); i < n; ++i)
        merged.insert(merged.end(), db[i].getBuffer(), db[i].getBuffer() + db[i].getSize());

    return merged;
}
