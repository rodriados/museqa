/**
 * Multiple Sequence Alignment sequences database header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef DATABASE_HPP_INCLUDED
#define DATABASE_HPP_INCLUDED

#include <map>
#include <set>
#include <string>
#include <vector>

#include <sequence.hpp>
#include <exception.hpp>

/**
 * Allows sequences to be stored alongside their respective properties.
 * @since 0.1.1
 */
struct database_entry
{
    std::string description;    /// The sequence description.
    sequence raw_sequence;      /// The sequence content.
};

/**
 * Stores a list of sequences read from possibly different sources. The
 * sequences may be identified by description or inclusion index.
 * @since 0.1.1
 */
class database
{
    public:
        using element_type = database_entry;    /// The type of data stored in the database.

    protected:
        std::vector<element_type> mlist;        /// The list of sequence entries in database.

    public:
        inline database() = default;
        inline database(const database&) = default;
        inline database(database&&) = default;

        inline database& operator=(const database&) = default;
        inline database& operator=(database&&) = default;

        /**
         * Gives access to a specific sequence in database.
         * @param offset The requested sequence offset.
         * @return The requested sequence.
         */
        inline const sequence& operator[](ptrdiff_t offset) const
        {
            return entry(offset).raw_sequence;
        }

        /**
         * Adds a new entry to database.
         * @param seq The sequence to be added to database.
         */
        inline void add(const sequence& seq)
        {
            add("unnamed #" + std::to_string(count() + 1), seq);
        }

        /**
         * Adds a new entry to database.
         * @param entry The entry to be added to database.
         */
        inline void add(const element_type& entry)
        {
            mlist.push_back(entry);
        }

        /**
         * Adds a new entry to database.
         * @param description The sequence description.
         * @param seq The sequence to be added to database.
         */
        inline void add(const std::string& description, const sequence& seq)
        {
            mlist.push_back({description, seq});
        }

        /**
         * Adds all entries from another database.
         * @param db The database to merge into this.
         */
        inline void add_many(const database& db)
        {
            add_many(db.mlist);
        }

        /**
         * Adds many entries to database
         * @param vector The list of sequences to add.
         */
        inline void add_many(const std::vector<sequence>& vector)
        {
            for(const sequence& seq : vector)
                add(seq);
        }

        /**
         * Adds many entries to database
         * @param vector The list of entries to add.
         */
        inline void add_many(const std::vector<element_type>& vector)
        {
            mlist.insert(mlist.end(), vector.begin(), vector.end());
        }

        /**
         * Adds many entries to database from a map.
         * @param map The map of entries to add.
         */
        inline void add_many(const std::map<std::string, sequence>& map)
        {
            for(const auto& pair : map)
                add(pair.first, pair.second);
        }

        /**
         * Creates a copy of the current database removing some selected elements.
         * @param excluded The elements to be removed from copy.
         * @return The new database with selected elements removed.
         */
        inline database except(const std::set<ptrdiff_t>& excluded) const
        {
            database db {*this};
            db.remove_many(excluded);
            return db;
        }

        /**
         * Creates a copy of the current database removing some selected elements.
         * @param excluded The elements to be removed from copy.
         * @return The new database with selected elements removed.
         */
        inline database except(const std::vector<ptrdiff_t>& excluded) const
        {
            return except(std::set<ptrdiff_t> {excluded.begin(), excluded.end()});
        }

        /**
         * Creates a new database only with the selected database entries.
         * @param selected The indeces to be included in new database.
         * @return The new database with only the selected elements.
         */
        inline database only(const std::set<ptrdiff_t>& selected) const
        {
            database db;

            for(const auto& it : selected)
                db.add(entry(it));

            return db;
        }

        /**
         * Creates a new database only with the selected database entries.
         * @param selected The indeces to be included in new database.
         * @return The new database with only the selected elements.
         */
        inline database only(const std::vector<ptrdiff_t>& selected) const
        {
            return only(std::set<ptrdiff_t> {selected.begin(), selected.end()});
        }

        /**
         * Removes an element from database.
         * @param offset Element to be removed from database.
         */
        inline void remove(ptrdiff_t offset)
        {
            enforce(size_t(offset) < count(), "database offset out of range");
            mlist.erase(mlist.begin() + offset);
        }

        /**
         * Removes many elements from database.
         * @param selected Elements to be removed.
         */
        inline void remove_many(const std::set<ptrdiff_t>& selected)
        {
            for(auto it = selected.rbegin(); it != selected.rend(); ++it)
                remove(*it);
        }

        /**
         * Removes many elements from database.
         * @param selected Elements to be removed.
         */
        inline void remove_many(const std::vector<ptrdiff_t>& selected)
        {
            remove_many(std::set<ptrdiff_t> {selected.begin(), selected.end()});
        }

        /**
         * Informs the number of entries in database.
         * @return The number of entries in database.
         */
        inline size_t count() const noexcept
        {
            return mlist.size();
        }

        /**
         * Retrieves an entry from database.
         * @param offset The entry offset index.
         * @return The requested entry.
         */
        inline const element_type& entry(ptrdiff_t offset) const
        {
            enforce(size_t(offset) < count(), "database offset out of range");
            return mlist.at(offset);
        }
};

#endif