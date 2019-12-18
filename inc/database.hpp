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
#include <iterator>

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
            using key_type = std::string;               /// The element's identity key type.
            using element_type = sequence;              /// The type of database's elements.

        protected:
            using underlying_type = std::map<key_type, element_type>;   /// The database's underlying type.

        protected:
            underlying_type m_map;                      /// The map of database's elements.

        public:
            inline database() = default;
            inline database(const database&) = default;
            inline database(database&&) = default;

            inline database& operator=(const database&) = default;
            inline database& operator=(database&&) = default;

            /**
             * Gives access to a specific sequence in database.
             * @param key The requested sequence identified by key.
             * @return The requested sequence.
             */
            inline const element_type& operator[](const key_type& key) const
            {
                auto pair = m_map.find(key);
                enforce(pair != m_map.end(), "unable to find key '%s' in database", key);
                return pair->second;
            }

            /**
             * Adds a new entry to database. If the key can already be found in
             * the database, then this method has no effect.
             * @param key The element's identifying key.
             * @param elem The element to be added to database.
             */
            inline void add(const key_type& key, const element_type& elem)
            {
                m_map.insert({key, elem});
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
             * @param db The database to merge into this.
             */
            inline void merge(const database& db)
            {
                m_map.insert(db.begin(), db.end());
            }

            /**
             * Merges all elements from another database by moving into this instance.
             * @param db The database to be moved into this.
             */
            inline void merge(database&& db)
            {
                m_map.insert(std::make_move_iterator(db.begin()), std::make_move_iterator(db.end()));
            }

            /**
             * Creates a new database with only a set of selected elements.
             * @param keys The keys to be included in new database.
             * @return The new database containing only the selected elements.
             */
            inline database only(const std::set<key_type>& keys) const
            {
                return database {*this, keys};
            }

            /**
             * Removes an element from database.
             * @param key Element key to be removed from database.
             */
            inline void remove(const key_type& key)
            {
                m_map.erase(key);
            }

            /**
             * Removes many elements from database.
             * @param keys Element keys to be removed from database.
             */
            inline void remove(const std::set<key_type>& keys)
            {
                for(const key_type& key : keys)
                    remove(key);
            }

            /**
             * Allows the database to be iterated from its beginning.
             * @return The database's iterator.
             */
            inline auto begin() noexcept -> underlying_type::iterator
            {
                return m_map.begin();
            }

            /**
             * Allows the database to be iterated without being modified.
             * @return The database's const iterator.
             */
            inline auto begin() const noexcept -> const underlying_type::iterator
            {
                return m_map.begin();
            }

            /**
             * Informs the database's final iterator point.
             * @return The iterator to the point after last element in database.
             */
            inline auto end() noexcept -> underlying_type::iterator
            {
                return m_map.end();
            }

            /**
             * Informs the database's final const iterator point.
             * @return The const iterator to the point after last element in database.
             */
            inline auto end() const noexcept -> const underlying_type::iterator
            {
                return m_map.end();
            }

            /**
             * Informs the number of entries in database.
             * @return The number of entries in database.
             */
            inline size_t count() const noexcept
            {
                return m_map.size();
            }

        private:
            /**
             * Creates a new instance from selected elements of another database.
             * @param db The database to copy elements from.
             * @param keys The selected keys to copy.
             */
            inline explicit database(const database& db, const std::set<key_type>& keys)
            {
                for(const key_type& key : keys)
                    add(key, db[key]);
            }
    };
}