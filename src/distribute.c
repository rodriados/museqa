/** @file distribute.c
 * @brief Parallel Multiple Sequence Alignment distribute file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "msa.h"
#include "distribute.h"

extern mpidata_t mpi_data;
extern msadata_t msa_data;

/** @fn void sendto(int, int, int)
 * @brief Prepares and sends data to slave node.
 * @param rank Slave node to send data to.
 * @param npair The total number of working pairs.
 * @param chunk The maximum number of working pairs to send to node.
 */
void sendto(int rank, int npair, int chunk)
{
    int off = chunk * (rank - 1);
    int bsz = 0, pos = 0;
    short nseq = 0;

    short *aux = calloc(msa_data.nseq, sizeof(short));
    short *new = calloc(msa_data.nseq, sizeof(short));
    unsigned *len = calloc(msa_data.nseq, sizeof(unsigned));

    npair = npair - off < chunk
        ? npair - off
        : chunk;

    for(int i = 0, j = off; i < npair; ++i, ++j)
        for(int k = 0; k < 2; ++k) {
            if(!aux[msa_data.pair[j].seq[k]]) {
                new[nseq] = msa_data.pair[j].seq[k];
                len[nseq] = msa_data.seq[msa_data.pair[j].seq[k]].length;
                bsz += sizeof(char) * (len[nseq] + 1);
                aux[msa_data.pair[j].seq[k]] = ++nseq;
            }

            msa_data.pair[j].seq[k] = aux[msa_data.pair[j].seq[k]] - 1;
        }

    bsz += sizeof(int) * (1 + nseq) + sizeof(short) * (1 + npair * 2);
    MPI_Send(&bsz, 1, MPI_INT, rank, M_SIZE, MPI_COMM_WORLD);

    aux = realloc(aux, bsz);
    MPI_Pack(&nseq,               1,         MPI_SHORT, aux, bsz, &pos, MPI_COMM_WORLD);
    MPI_Pack(&npair,              1,         MPI_INT,   aux, bsz, &pos, MPI_COMM_WORLD);
    MPI_Pack(&msa_data.pair[off], npair * 2, MPI_SHORT, aux, bsz, &pos, MPI_COMM_WORLD);
    MPI_Pack(len,                 nseq,      MPI_INT,   aux, bsz, &pos, MPI_COMM_WORLD);

    for(int i = 0; i < nseq; ++i)
        MPI_Pack(msa_data.seq[new[i]].data, len[i] + 1, MPI_CHAR, aux, bsz, &pos, MPI_COMM_WORLD);

    MPI_Send(aux, bsz, MPI_PACKED, rank, M_DATA, MPI_COMM_WORLD);

    free(aux);
    free(new);
    free(len);
}

/** @fn void collect(int)
 * @brief Receives and sets up data from master node.
 * @param rank Master node to receive data from.
 */
void collect(int rank)
{
    int bsz, nseq, npair, pos = 0;
    MPI_Status s;

    MPI_Recv(&bsz,  1, MPI_INT,    rank, M_SIZE, MPI_COMM_WORLD, &s);
    
    char *buf = malloc(bsz);
    MPI_Recv(buf, bsz, MPI_PACKED, rank, M_DATA, MPI_COMM_WORLD, &s);

    MPI_Unpack(buf, bsz, &pos, &nseq,         1,         MPI_SHORT, MPI_COMM_WORLD);
    MPI_Unpack(buf, bsz, &pos, &npair,        1,         MPI_INT,   MPI_COMM_WORLD);

    msa_data.npair = npair;
    msa_data.pair  = malloc(sizeof(workpair_t) * npair);
    unsigned *len  = malloc(sizeof(int) * nseq);

    MPI_Unpack(buf, bsz, &pos, msa_data.pair, npair * 2, MPI_SHORT, MPI_COMM_WORLD);
    MPI_Unpack(buf, bsz, &pos, len,           nseq,      MPI_INT,   MPI_COMM_WORLD);

    msa_data.nseq = nseq;
    msa_data.seq  = malloc(sizeof(sequence_t) * nseq);

    for(int i = 0; i < nseq; ++i) {
        msa_data.seq[i].length = len[i];
        msa_data.seq[i].data = malloc(sizeof(char) * (len[i] + 1));
        MPI_Unpack(buf, bsz, &pos, msa_data.seq[i].data, len[i] + 1, MPI_CHAR, MPI_COMM_WORLD);
    }

    free(buf);
    free(len);
}

/** @fn void distribute()
 * @brief Scatters sequences and working pairs to slave nodes.
 */
void distribute()
{
    int count = msa_data.nseq * (msa_data.nseq - 1) / 2.0;
    int chunk = ceil(count / (float)(mpi_data.nproc - 1));

    __debugh("scatter %d sequence pairs for each node\n", chunk);

    double time  = MPI_Wtime();
    msa_data.pair = malloc(sizeof(workpair_t) * count);

    for(int i = 0, wp = 0; i < msa_data.nseq; ++i)
        for(int j = i + 1; j < msa_data.nseq; ++j, ++wp) {
            msa_data.pair[wp].seq[0] = i;
            msa_data.pair[wp].seq[1] = j;
        }

    for(int rank = 1; rank < mpi_data.nproc; ++rank)
        sendto(rank, count, chunk);

    __debugh("time spent by distribution: %lfs\n", MPI_Wtime() - time);

    free(msa_data.pair);
}
