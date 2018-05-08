/** @file fasta.hpp
 * @brief Parallel Multiple Sequence Alignment fasta header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _FASTA_HPP
#define _FASTA_HPP

#include "msa.h"

/** @class fasta_t
 * @brief Holds all data extracted from a fasta file.
 * @var nseq The number of extracted sequences.
 * @var seq The list of extracted sequences.
 */
class fasta_t
{
public:
    unsigned short nseq;
    sequence_t *seq;

public:
    fasta_t();
    ~fasta_t();
    int read(const char *);

private:
    int tobuffer(FILE *, char **);
    int loadsequence(FILE *);
};

#endif