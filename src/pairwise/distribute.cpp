/** @file distribute.cpp
 * @brief Parallel Multiple Sequence Alignment pairwise distribute file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstring>
#include <mpi.h>

#include "msa.h"
#include "fasta.hpp"

#include "distribute.hpp"
#include "pairwise.hpp"

extern pairwise_t pairdata;

namespace pairwise
{
/** @fn void pairwise::sync(const fasta_t& fasta)
 * @brief Broadcasts all content read from file.
 * @param fasta Data structure read from file.
 */
void sync(const fasta_t& fasta)
{
    int bufsize = 0;
    pairdata.nseq = fasta.nseq;

    MPI_Bcast(&pairdata.nseq, 1, MPI_SHORT, __master, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    pairdata.seq = new position_t[pairdata.nseq];

    __onlymaster {
        for(int i = 0; i < fasta.nseq; ++i) {
            pairdata.seq[i].offset = bufsize;
            bufsize += pairdata.seq[i].length = fasta.seq[i].length;
        }
    }

    MPI_Bcast(&bufsize, 1, MPI_UNSIGNED, __master, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    pairdata.data = new char[bufsize];

    __onlymaster {
        for(int i = 0, off = 0; i < fasta.nseq && off < bufsize; ++i) {
            memcpy(pairdata.data + off, fasta.seq[i].data, fasta.seq[i].length);
            off += fasta.seq[i].length;
        }
    }

    MPI_Bcast(pairdata.data, bufsize, MPI_CHAR, __master, MPI_COMM_WORLD);
    MPI_Bcast(pairdata.seq, pairdata.nseq, MPI_2INT, __master, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

/** @fn void pairwise::scatter()
 * @brief Scatters sequences and working pairs to slave nodes.
 */
void scatter()
{
    int total  = pairdata.nseq * (pairdata.nseq - 1) / 2.0;
    int *count = new int[mpi_data.size];
    int *displ = new int[mpi_data.size];

    int div = total / mpi_data.size;
    int rest = total % mpi_data.size;
    workpair_t *workpairs;

    for(int i = 0, off = 0; i < mpi_data.size; ++i) {
        displ[i] = off;
        count[i] = div + (i < rest);
        off += count[i];
    }

    __onlymaster {
        workpairs = new workpair_t[total];
        __debugh("created %d sequence pairs", total);

        for(int i = 0, wp = 0; i < pairdata.nseq; ++i)
            for(int j = i + 1; j < pairdata.nseq; ++j, ++wp) {
                workpairs[wp].seq[0] = i;
                workpairs[wp].seq[1] = j;
            }
    }

    pairdata.npair = count[mpi_data.rank];
    pairdata.pair = new workpair_t[pairdata.npair];

    MPI_Datatype workpair_type;
    MPI_Type_contiguous(2, MPI_SHORT, &workpair_type);
    MPI_Type_commit(&workpair_type);

    MPI_Scatterv(
        workpairs, count, displ, workpair_type,
        pairdata.pair, count[mpi_data.rank], workpair_type,
        __master, MPI_COMM_WORLD
    );
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Type_free(&workpair_type);

    free(count);
    free(displ);
    __onlymaster free(workpairs);
}

/** @fn pairwise::clean()
 * @brief Cleans up all dynamicaly allocated data for pairwise.
 */
void clean()
{
    delete[] pairdata.seq;
    delete[] pairdata.pair;
}

}