/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for a database of sequences to be aligned.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include <iterator>

#include "database.hpp"

namespace museqa
{
    /**
     * Keeps track of the global count of anonymous sequences, so they don't take
     * the chance of inadvertently overriding sequences with equal IDs.
     * @since 0.1.1
     */
    int database::anonymous = 0;

    /**
     * Adds all elements from another database into this instance.
     * @param db The database to merge into this instance.
     */
    void database::merge(const database& db)
    {
        const size_t last = m_entries.size();
        m_entries.insert(m_entries.end(), db.begin(), db.end());
        update_keys(last);
    }

    /**
     * Moves all elements from another database into this instance.
     * @param db The database to merge into this instance.
     */
    void database::merge(database&& db)
    {
        const size_t last = m_entries.size();
        m_entries.insert(m_entries.end(), std::make_move_iterator(db.begin()), std::make_move_iterator(db.end()));
        db.m_indeces.clear();
        update_keys(last);
    }

    /**
     * Updates the indeces keys from the given starting index.
     * @param index The entry index from which updates must start.
     */
    void database::update_keys(size_t index)
    {
        for(size_t total = count(); index < total; ++index) {
            const auto& entry = m_entries[index];
            m_indeces[entry.description] = index;
        }
    }
}
