/** @file distribute.cpp
 * @brief Parallel Multiple Sequence Alignment pairwise distribute file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstring>
#include <mpi.h>

#include "msa.h"
#include "fasta.hpp"
#include "pairwise.cuh"

/** @fn void pairwise_t::load(const fasta_t *fasta)
 * @brief Broadcasts all content read from file.
 * @param fasta Data structure read from file.
 */
void pairwise_t::load(const fasta_t *fasta)
{
    int bfsize = 0;

    this->nseq = fasta->nseq;

    MPI_Bcast(&this->nseq, 1, MPI_SHORT, __master, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    this->seq = new position_t[this->nseq];

    __onlymaster {
        for(int i = 0; i < fasta->nseq; ++i) {
            this->seq[i].offset = bfsize;
            bfsize += this->seq[i].length = fasta->seq[i].length;
        }
    }

    MPI_Bcast(&bfsize, 1, MPI_UNSIGNED, __master, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    this->seqchar = new char[bfsize];

    __onlymaster {
        for(int i = 0, off = 0; i < fasta->nseq && off < bfsize; ++i) {
            memcpy(this->seqchar + off, fasta->seq[i].data, fasta->seq[i].length);
            off += fasta->seq[i].length;
        }
    }

    MPI_Bcast(this->seq, this->nseq, MPI_2INT, __master, MPI_COMM_WORLD);
    MPI_Bcast(this->seqchar, bfsize, MPI_CHAR, __master, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

/** @fn void pairwise_t::scatter()
 * @brief Scatters sequences and working pairs to slave nodes.
 */
void pairwise_t::scatter()
{
    int *count = new int [mpi_data.size];
    int *displ = new int [mpi_data.size];
    workpair_t *workpair;

    int total = this->nseq * (this->nseq - 1) / 2.0;
    int div = total / mpi_data.size;
    int rem = total % mpi_data.size;

    __onlymaster __debugh("created %d sequence pairs", total);

    for(int i = 0, offset = 0; i < mpi_data.size; ++i) {
        displ[i] = offset;
        count[i] = div + (i < rem);
        offset += count[i];
    }

    workpair = new workpair_t [total];

    for(int i = 0, wp = 0; i < this->nseq; ++i)
        for(int j = i + 1; j < this->nseq; ++j, ++wp) {
            workpair[wp].seq[0] = i;
            workpair[wp].seq[1] = j;
        }

    this->npair = count[mpi_data.rank];
    this->pair  = new workpair_t [this->npair];

    for(int i = 0, j = displ[mpi_data.rank]; i < count[mpi_data.rank]; ++i, ++j)
        this->pair[i] = workpair[j];

    delete[] count;
    delete[] displ;
    delete[] workpair;
}
