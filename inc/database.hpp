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

#include "sequence.hpp"
#include "exception.hpp"

/**
 * Allows sequences to be stored alongside their respective properties.
 * @since 0.1.1
 */
struct DatabaseEntry
{
    std::string description;    /// The sequence description.
    Sequence sequence;          /// The sequence content.
};

/**
 * Stores a list of sequences read from possibly different sources. The
 * sequences may be identified by description or inclusion index.
 * @since 0.1.1
 */
class Database
{
    protected:
        std::vector<DatabaseEntry> list;    /// The list of sequence entries in database.

    public:
        inline Database() = default;
        inline Database(const Database&) = default;
        inline Database(Database&&) = default;

        inline Database& operator=(const Database&) = default;
        inline Database& operator=(Database&&) = default;

        /**
         * Gives access to a specific sequence in database.
         * @param offset The requested sequence offset.
         * @return The requested sequence.
         */
        inline const Sequence& operator[](ptrdiff_t offset) const
        {
            return getEntry(offset).sequence;
        }

        /**
         * Informs the number of entries in database.
         * @return The number of entries in database.
         */
        inline size_t getCount() const noexcept
        {
            return list.size();
        }

        /**
         * Retrieves an entry from database.
         * @param offset The entry offset index.
         * @return The requested entry.
         */
        inline const DatabaseEntry& getEntry(ptrdiff_t offset) const
        {
            enforce(static_cast<size_t>(offset) < getCount(), "database offset out of range");
            return list.at(offset);
        }

        /**
         * Adds a new entry to database.
         * @param sequence The sequence to be added to database.
         */
        inline void add(const Sequence& sequence)
        {
            add("unnamed #" + std::to_string(getCount() + 1), sequence);
        }

        /**
         * Adds a new entry to database.
         * @param entry The entry to be added to database.
         */
        inline void add(const DatabaseEntry& entry)
        {
            list.push_back(entry);
        }

        /**
         * Adds a new entry to database.
         * @param description The sequence description.
         * @param sequence The sequence to be added to database.
         */
        inline void add(const std::string& description, const Sequence& sequence)
        {
            list.push_back({description, sequence});
        }

        /**
         * Adds all entries from another database.
         * @param dbase The database to merge into this.
         */
        inline void addMany(const Database& dbase)
        {
            addMany(dbase.list);
        }

        /**
         * Adds many entries to database
         * @param vector The list of sequences to add.
         */
        inline void addMany(const std::vector<Sequence>& vector)
        {
            for(const Sequence& sequence : vector)
                add(sequence);
        }

        /**
         * Adds many entries to database
         * @param vector The list of entries to add.
         */
        inline void addMany(const std::vector<DatabaseEntry>& vector)
        {
            list.insert(list.end(), vector.begin(), vector.end());
        }

        /**
         * Adds many entries to database from a map.
         * @param map The map of entries to add.
         */
        inline void addMany(const std::map<std::string, Sequence>& map)
        {
            for(const auto& pair : map)
                add(pair.first, pair.second);
        }

        /**
         * Creates a copy of the current database removing some selected elements.
         * @param excluded The elements to be removed from copy.
         * @return The new database with selected elements removed.
         */
        inline Database except(const std::set<ptrdiff_t>& excluded) const
        {
            Database db {*this};
            db.removeMany(excluded);
            return db;
        }

        /**
         * Creates a copy of the current database removing some selected elements.
         * @param excluded The elements to be removed from copy.
         * @return The new database with selected elements removed.
         */
        inline Database except(const std::vector<ptrdiff_t>& excluded) const
        {
            return except(std::set<ptrdiff_t> {excluded.begin(), excluded.end()});
        }

        /**
         * Creates a new database only with the selected database entries.
         * @param selected The indeces to be included in new database.
         * @return The new database with only the selected elements.
         */
        inline Database only(const std::set<ptrdiff_t>& selected) const
        {
            Database db;

            for(const auto& it : selected)
                db.add(getEntry(it));

            return db;
        }

        /**
         * Creates a new database only with the selected database entries.
         * @param selected The indeces to be included in new database.
         * @return The new database with only the selected elements.
         */
        inline Database only(const std::vector<ptrdiff_t>& selected) const
        {
            return only(std::set<ptrdiff_t> {selected.begin(), selected.end()});
        }

        /**
         * Removes an element from database.
         * @param offset Element to be removed from database.
         */
        inline void remove(ptrdiff_t offset)
        {
            enforce(static_cast<size_t>(offset) < getCount(), "database offset out of range");
            list.erase(list.begin() + offset);
        }

        /**
         * Removes many elements from database.
         * @param selected Elements to be removed.
         */
        inline void removeMany(const std::set<ptrdiff_t>& selected)
        {
            for(auto it = selected.rbegin(); it != selected.rend(); ++it)
                remove(*it);
        }

        /**
         * Removes many elements from database.
         * @param selected Elements to be removed.
         */
        inline void removeMany(const std::vector<ptrdiff_t>& selected)
        {
            removeMany(std::set<ptrdiff_t> {selected.begin(), selected.end()});
        }
};

#endif