/** @file fasta.hpp
 * @brief Parallel Multiple Sequence Alignment fasta header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _FASTA_HPP
#define _FASTA_HPP

#include "msa.h"

/** @struct fasta_t
 * @brief Holds all data extracted from a fasta file.
 * @var nseq The number of extracted sequences.
 * @var seq The list of extracted sequences.
 */
struct fasta_t {
    short nseq;
    sequence_t *seq;

    fasta_t();
    ~fasta_t();
    int load(const char *);

private:
    int tobuffer(FILE *, char **);
    int loadsequence(FILE *);
};

#endif