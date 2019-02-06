/**
 * Multiple Sequence Alignment sequences database header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef DATABASE_HPP_INCLUDED
#define DATABASE_HPP_INCLUDED

#include <map>
#include <string>
#include <vector>

#include "sequence.hpp"

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
        uint32_t index = 0;                 /// The description index for anonymous sequences.

    public:
        Database() = default;
        Database(const Database&) = default;
        Database(Database&&) = default;

        Database& operator=(const Database&) = default;
        Database& operator=(Database&&) = default;

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
         * Adds a new entry to database.
         * @param sequence The sequence to be added to database.
         */
        inline void add(const Sequence& sequence)
        {
            std::string description = std::string("#") + std::to_string(++index);
            add(description, sequence);
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
            for(const DatabaseEntry& entry : vector)
                add(entry);
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
         * Informs the number of entries in database.
         * @return The number of entries in database.
         */
        inline size_t getCount() const
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
            return list.at(offset);
        }
};

#endif