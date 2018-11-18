/**
 * Multiple Sequence Alignment fasta header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef FASTA_HPP_INCLUDED
#define FASTA_HPP_INCLUDED

#pragma once

#include <string>
#include <fstream>

#include "sequence.hpp"

/**
 * Represents a sequence read from a fasta file.
 * @since 0.1.alpha
 */
class FastaSequence : public Sequence
{
    private:
        const std::string description;  /// The sequence description.

    public:
        FastaSequence() = default;
        FastaSequence(const FastaSequence&) = default;
        FastaSequence(FastaSequence&&) = default;

        /**
         * Instantiates a new fasta sequence.
         * @param description The sequence description.
         * @param string The string containing this sequence's data.
         */
        inline FastaSequence(const std::string& description, const std::string& string)
        :   Sequence(string)
        ,   description(description) {}

        /**
         * Instantiates a new fasta sequence.
         * @param description The sequence description.
         * @param buffer The buffer containing this sequence's data.
         */
        inline FastaSequence(const std::string& description, const Sequence& buffer)
        :   Sequence(buffer)
        ,   description(description) {}

        /**
         * Instantiates a new fasta sequence.
         * @param description The sequence description.
         * @param buffer The buffer containing this sequence's data.
         * @param size The buffer's size.
         */
        inline FastaSequence(const std::string& description, const char *buffer, size_t size)
        :   Sequence(buffer, size)
        ,   description(description) {}

        virtual ~FastaSequence() noexcept = default;

        FastaSequence& operator=(const FastaSequence&) = default;
        FastaSequence& operator=(FastaSequence&&) = default;

        /**
         * Retrieves the description linked to the sequence.
         * @return The sequence's description.
         */
        inline const std::string& getDescription() const
        {
            return this->description;
        }
};

/**
 * Creates a list of sequences read from a fasta file. This sequence list
 * is responsible for keeping track of sequences within its scope. Once a
 * sequence is put into the list, it cannot leave.
 * @since 0.1.alpha
 */
class Fasta final
{
    protected:
        std::vector<FastaSequence> list;    /// The list of sequences read from file.

    public:
        Fasta() = default;
        Fasta(const Fasta&) = default;
        Fasta(Fasta&&) = default;

        Fasta(const std::string&);

        Fasta& operator=(const Fasta&) = default;
        Fasta& operator=(Fasta&&) = default;

        /**
         * Gives access to a specific sequence of the list.
         * @param offset The requested sequence offset.
         * @return The requested sequence.
         */
        inline const FastaSequence& operator[](ptrdiff_t offset) const
        {
            return this->list.at(offset);
        }
        
        /**
         * Informs the number of sequences in the list.
         * @return The list's number of sequences.
         */
        inline size_t getCount() const
        {
            return this->list.size();
        }

        void load(const std::string&);

        static void broadcast(Fasta&);

    private:
        bool extract(std::fstream&);
        void push(const std::string&, const std::string&);
        void push(const std::string&, const char *, size_t);
};

#endif