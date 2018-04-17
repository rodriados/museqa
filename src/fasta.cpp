/** @file fasta.cpp
 * @brief Parallel Multiple Sequence Alignment fasta file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdio>
#include <cctype>
#include <cstdlib>

#include "msa.h"
#include "fasta.hpp"

/** @fn fasta_t::fasta_t()
 * @brief Initializes the object.
 */
fasta_t::fasta_t()
{
    this->nseq = 0;
    this->seq = NULL;
}

/** @fn fasta_t::~fasta_t()
 * @brief Frees all memory allocated to sequences.
 */
fasta_t::~fasta_t()
{
    for(int i = 0; i < this->nseq; ++i)
        free(this->seq[i].data);

    free(this->seq);
}

/** @fn int fasta_t::tobuffer(FILE *, char **)
 * @brief Reads a sequence out of the file and puts it into a buffer.
 * @param ffile The file to read sequence from.
 * @param dest The destination address for the sequence.
 * @return Size of the read sequence.
 */
int fasta_t::tobuffer(FILE *ffile, char **dest)
{
    char buf[2049];

    if(feof(ffile) || ferror(ffile) || !fscanf(ffile, " >%2048[^\t\n]", buf))
        return 0;

    int size = 0;

    while(!feof(ffile) && fscanf(ffile, " %2048[^>\t\n]\n", buf))
        for(char *bc = buf; *bc; ++bc) {
            if(!isalpha(*bc))
                continue;

            if(size % 2048 == 0)
                *dest = (char *)realloc(*dest, sizeof(char) * (size + 2049));

            (*dest)[size++] = *bc;
        }

    return size;
}

/** @fn int fasta_t::loadsequence(FILE *)
 * @brief Reads a sequence from file.
 * @param ffile File to read sequence from.
 * @return Size of the read sequence.
 */
int fasta_t::loadsequence(FILE *ffile)
{
    int size;
    char *aux = NULL;

    if(!(size = tobuffer(ffile, &aux)))
        return 0;

    int i = this->nseq++;

    this->seq = (sequence_t *)realloc(this->seq, sizeof(sequence_t) * this->nseq);
    this->seq[i].length = size;
    this->seq[i].data = aux;

    return size;
}

/** @fn int fasta_t::load(const char *)
 * @brief Reads a file and allocates memory to all sequences contained in it.
 * @param fname The name of the file to be read.
 * @return Number of sequences read.
 */
int fasta_t::load(const char *fname)
{
    FILE *ffasta = fopen(fname, "r");

    __debugh("loading from %s", fname);

    if(ffasta == NULL)
        finish(INVALIDFILE);

    while(!feof(ffasta) && !ferror(ffasta))
        if(!this->loadsequence(ffasta))
            break;

    __debugh("loaded %d sequences", this->nseq);

    return this->nseq;
}