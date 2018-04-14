/** @file fasta.c
 * @brief Parallel Multiple Sequence Alignment fasta file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>

#include "msa.h"
#include "fasta.h"

extern msadata_t msa_data;

/** @fn int tobuffer(FILE *, char **)
 * @brief Reads a sequence out of the file and puts it into a buffer.
 * @param ffile The file to read sequence from.
 * @param dest The destination address for the sequence.
 * @return Size of the read sequence.
 */
int tobuffer(FILE *ffile, char **dest)
{
    char buf[513];

    if(feof(ffile) || ferror(ffile) || !fscanf(ffile, " >%[^\t\n]", buf))
        return 0;

    register int size = 0;

    while(fscanf(ffile, " %512[^>\t\n]s", buf) && !feof(ffile))
        for(char *bc = buf; *bc != '\0'; ++bc) {
            if(!isalpha(*bc))
                continue;

            if((size & 2047) == 0)
                *dest = realloc(*dest, sizeof(char) * (size + 2049));

            (*dest)[size++] = *bc;
        }

    (*dest)[size] = '\0';
    return size;
}

/** @fn int loadsequence(FILE *)
 * @brief Reads a sequence from file.
 * @param ffile File to read sequence from.
 * @return Size of the read sequence.
 */
int loadsequence(FILE *ffile)
{
    int size;
    char *aux = NULL;

    if(!(size = tobuffer(ffile, &aux)))
        return 0;

    int i = msa_data.nseq++;

    msa_data.seq = realloc(
        msa_data.seq,
        sizeof(sequence_t) * msa_data.nseq
    );

    msa_data.seq[i].length = size;
    msa_data.seq[i].data = aux;

    return size;
}

/** @fn int loadfasta(const char *)
 * @brief Reads a file and allocates memory to all sequences contained in it.
 * @param fname The name of the file to be read.
 * @return Number of sequences read.
 */
int loadfasta(const char *fname)
{
    FILE *ffasta = fopen(fname, "r");

    if(ffasta == NULL)
        finish(INVALIDFILE);

    while(!feof(ffasta) && !ferror(ffasta))
        loadsequence(ffasta);

    __debugh("loaded %d sequences\n", msa_data.nseq);

    return msa_data.nseq;
}

/** @fn void freefasta()
 * @brief Frees all memory allocated to sequences.
 */
void freefasta()
{
    for(int i = 0; i < msa_data.nseq; ++i)
        free(msa_data.seq[i].data);

    free(msa_data.seq);
}
