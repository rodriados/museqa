/**
 * Multiple Sequence Alignment fasta header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _FASTA_HPP_
#define _FASTA_HPP_

#include <fstream>

#include "msa.hpp"
#include "sequence.hpp"

/**
 * Holds all data extracted from a fasta file.
 * @since 0.1.alpha
 */
class Fasta final : protected SequenceList
{
    public:
        Fasta() = default;

        using SequenceList::getCount;
        using SequenceList::operator[];

        uint16_t read(const char *);
        
        using SequenceList::compact;
        using SequenceList::select;

    private:
        bool extract(std::fstream&);
        uint16_t bufferize(std::fstream&, char **);
}

#endif