/**
 * Multiple Sequence Alignment sequences database header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include <format.hpp>
#include <sequence.hpp>
#include <exception.hpp>

namespace msa
{
    /**
     * Stores a list of sequences read from possibly different sources. The added
     * sequences can only be accessed via their respective identity or iterator.
     * @since 0.1.1
     */
    class database
    {
        public:
            using element_type = sequence;          /// The type of database's elements.

        protected:
            /**
             * Groups up a sequence's contents with its respective description.
             * @since 0.1.1
             */
            struct entry_type
            {
                std::string description;
                element_type contents;
            };

        protected:
            using underlying_type = std::vector<entry_type>;    /// The database's underlying type.

        protected:
            underlying_type m_db;                   /// The map of database's elements.

        public:
            inline database() = default;
            inline database(const database&) = default;
            inline database(database&&) = default;

            inline database& operator=(const database&) = default;
            inline database& operator=(database&&) = default;

            /**
             * Gives access to a specific entry in database.
             * @param offset The requested entry in given offset.
             * @return The retrieved entry.
             */
            inline const entry_type& operator[](ptrdiff_t offset) const
            {
                enforce(offset >= 0 && size_t(offset) < count(), "database offset out of range");
                return m_db[offset];
            }

            /**
             * Adds a new entry to database. The sequence's description will not
             * be, in any way, checked for uniqueness. Thus, if a sequence with
             * the same description is already known, it will be duplicated.
             * @param description The sequence's description.
             * @param elem The element to be added to database.
             */
            inline void add(const std::string& description, const element_type& elem)
            {
                m_db.push_back({description, elem});
            }

            /**
             * Adds a new anonymous element to database.
             * @param elem The element to be added to database.
             */
            inline void add(const element_type& elem)
            {
                add(fmt::format("anonymous#%d", count() + 1), elem);
            }

            /**
             * Adds many elements to database
             * @param elems The list of elements to be added into database.
             */
            inline void add(const std::vector<element_type>& elems)
            {
                for(const element_type& elem : elems)
                    add(elem);
            }

            /**
             * Adds all elements from another database into this instance.
             * @param db The database to merge into this instance.
             */
            inline void merge(const database& db)
            {
                m_db.insert(m_db.end(), db.begin(), db.end());
            }

            /**
             * Merges all elements from another database by moving into this instance.
             * @param db The database to be moved into this instance.
             */
            inline void merge(database&& db)
            {
                m_db.insert(m_db.end(), std::make_move_iterator(db.begin()), std::make_move_iterator(db.end()));
            }

            /**
             * Creates a new database with only a set of selected elements.
             * @param entries The entries to be included in new database.
             * @return The new database containing only the selected elements.
             */
            inline database only(const std::set<ptrdiff_t>& entries) const
            {
                return database {*this, entries};
            }

            /**
             * Removes an element from database.
             * @param offset Element offset to be removed from database.
             */
            inline void remove(ptrdiff_t offset)
            {
                m_db.erase(m_db.begin() + offset);
            }

            /**
             * Removes many elements from database.
             * @param entries Element entries to be removed from database.
             */
            inline void remove(const std::set<ptrdiff_t>& entries)
            {
                for(ptrdiff_t entry : entries)
                    remove(entry);
            }

            /**
             * Allows the database to be iterated from its beginning.
             * @return The database's iterator.
             */
            inline auto begin() noexcept -> underlying_type::iterator
            {
                return m_db.begin();
            }

            /**
             * Allows the database to be iterated without being modified.
             * @return The database's const iterator.
             */
            inline auto begin() const noexcept -> const underlying_type::const_iterator
            {
                return m_db.begin();
            }

            /**
             * Informs the database's final iterator point.
             * @return The iterator to the point after last element in database.
             */
            inline auto end() noexcept -> underlying_type::iterator
            {
                return m_db.end();
            }

            /**
             * Informs the database's final const iterator point.
             * @return The const iterator to the point after last element in database.
             */
            inline auto end() const noexcept -> const underlying_type::const_iterator
            {
                return m_db.end();
            }

            /**
             * Informs the number of entries in database.
             * @return The number of entries in database.
             */
            inline size_t count() const noexcept
            {
                return m_db.size();
            }

        private:
            /**
             * Creates a new instance from selected elements of another database.
             * @param db The database to copy elements from.
             * @param entries The selected entries to copy.
             */
            inline explicit database(const database& db, const std::set<ptrdiff_t>& entries)
            {
                for(ptrdiff_t entry : entries)
                    m_db.push_back(db[entry]);
            }
    };
}