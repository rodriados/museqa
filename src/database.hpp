/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a database of sequences to be aligned.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "format.hpp"
#include "sequence.hpp"
#include "exception.hpp"

namespace museqa
{
    /**
     * Stores a list of sequences read from possibly different sources. The added
     * sequences can only be accessed via their respective identity or iterator.
     * @since 0.1.1
     */
    class database
    {
        public:
            using element_type = sequence;                      /// The type of database's elements.

        public:
            /**
             * Exposes an entry sequence with its description.
             * @since 0.1.1
             */
            using entry_type = struct {
                std::string description;
                element_type contents;
            };

        protected:
            using underlying_type = std::vector<entry_type>;
            using mapping_type = std::map<std::string, ptrdiff_t>;

        private:
            static int anonymous;                               /// Unique global IDs for anonymous sequences.

        protected:
            underlying_type m_entries;                          /// The database's element storage.
            mapping_type m_indeces;                             /// The elements' indeces mapping.

        public:
            inline database() noexcept = default;
            inline database(const database&) = default;
            inline database(database&&) = default;

            inline database& operator=(const database&) = default;
            inline database& operator=(database&&) = default;

            /**
             * Gives access to a specific entry in database via its offset.
             * @param offset The requested entry in given offset.
             * @return The retrieved entry.
             */
            inline const entry_type& operator[](ptrdiff_t offset) const
            {
                enforce(offset >= 0 && size_t(offset) < count(), "database offset out of range");
                return m_entries[offset];
            }

            /**
             * Gives access to a specific entry in database via its key.
             * @param key The requested key to be retrieved from database.
             * @return The retrieved entry.
             */
            inline const entry_type& operator[](const std::string& key) const
            {
                const auto entry = m_indeces.find(key);
                enforce(entry != m_indeces.end(), "cannot find key in database");
                return m_entries[entry->second];
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
                m_indeces[description] = m_entries.size();
                m_entries.push_back({description, elem});
            }

            /**
             * Adds a new anonymous element to database.
             * @param elem The element to be added to database.
             */
            inline void add(const element_type& elem)
            {
                add(fmt::format("anonymous#%d", ++anonymous), elem);
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
             * Creates a new database with only a set of selected elements.
             * @tparam T The type of key used to select elements in database.
             * @param keys The sequence keys to be included in new database.
             * @return The new database containing only the selected elements.
             */
            template <typename T>
            inline database only(const std::set<T>& keys) const
            {
                return database {*this, keys};
            }

            /**
             * Allows the database to be iterated from its beginning.
             * @return The database's iterator.
             */
            inline auto begin() noexcept -> underlying_type::iterator
            {
                return m_entries.begin();
            }

            /**
             * Allows the database to be iterated without being modified.
             * @return The database's const iterator.
             */
            inline auto begin() const noexcept -> const underlying_type::const_iterator
            {
                return m_entries.begin();
            }

            /**
             * Informs the database's final iterator point.
             * @return The iterator to the point after last element in database.
             */
            inline auto end() noexcept -> underlying_type::iterator
            {
                return m_entries.end();
            }

            /**
             * Informs the database's final const iterator point.
             * @return The const iterator to the point after last element in database.
             */
            inline auto end() const noexcept -> const underlying_type::const_iterator
            {
                return m_entries.end();
            }

            /**
             * Informs the number of entries in database.
             * @return The number of entries in database.
             */
            inline size_t count() const noexcept
            {
                return m_entries.size();
            }

            void merge(const database& db);
            void merge(database&&);

        private:
            /**
             * Creates a new instance from selected elements of another database.
             * @param db The database to copy elements from.
             * @param entries The selected entries to copy.
             */
            template <typename T>
            inline explicit database(const database& db, const std::set<T>& entries)
            {
                for(const auto& entry : entries)
                    add(db[entry]);
            }

            /**
             * Adds a new entry to database. If an entry with equal description
             * already exists in the database, it'll be overriden.
             * @param entry The entry to be added to database.
             */
            inline void add(const entry_type& entry)
            {
                add(entry.description, entry.contents);
            }

            void update_keys(size_t);
    };
}
