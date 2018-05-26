/**
 * Multiple Sequence Alignment fasta header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _FASTA_HPP_
#define _FASTA_HPP_

#include <fstream>
#include <string>

#include "sequence.hpp"

/**
 * Represents a sequence read from a fasta file.
 * @since 0.1.alpha
 */
class FastaSequence : public Sequence
{
    private:
        const std::string description;      /// The sequence description.

    public:
        FastaSequence(const std::string&, const std::string&);
        FastaSequence(const std::string&, const Buffer<char>&);
        FastaSequence(const std::string&, const char *, uint32_t);
        virtual ~FastaSequence() noexcept = default;

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
        std::vector<FastaSequence *> list;  /// The list of sequences read from file.

    public:
        Fasta() = default;
        ~Fasta() noexcept;

        /**
         * Gives access to a specific sequence of the list.
         * @return The requested sequence.
         */
        inline const FastaSequence& operator[](uint16_t offset) const
        {
            return *(this->list.at(offset));
        }

        /**
         * Informs the number of sequences in the list.
         * @return The list's number of sequences.
         */
        inline uint16_t getCount() const
        {
            return this->list.size();
        }

        uint16_t load(const std::string&);

    private:
        void broadcast();
        bool extract(std::fstream&);
        void push(const std::string&, const std::string&);
        void push(const std::string&, const Buffer<char>&);
        void push(const std::string&, const char *, uint32_t);
};

#endif