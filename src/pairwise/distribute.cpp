/** @file distribute.cpp
 * @brief Parallel Multiple Sequence Alignment pairwise distribute file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
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
    this->clength = 0;

    __onlymaster {
        this->nseq  = fasta->nseq;
        this->npair = fasta->nseq * (fasta->nseq - 1) / 2.0;

        __debugh("generating %d sequence pairs", this->npair);
    }

    MPI_Bcast(&this->nseq,  1, MPI_SHORT, __master, MPI_COMM_WORLD);
    MPI_Bcast(&this->npair, 1, MPI_INT,   __master, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    this->seq = new position_t [this->nseq];

    __onlymaster {
        for(int i = 0; i < fasta->nseq; ++i) {
            this->seq[i].offset = this->clength;
            this->seq[i].length = fasta->seq[i].length;

            this->clength += fasta->seq[i].length;
        }
    }

    MPI_Bcast(this->seq, this->nseq, MPI_2INT, __master, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    __onlymaster {
        this->seqchar = (char *)malloc(sizeof(char) * this->clength);

        for(uint32_t i = 0, off = 0; i < fasta->nseq && off < this->clength; ++i) {
            memcpy(this->seqchar + off, fasta->seq[i].data, fasta->seq[i].length);
            off += fasta->seq[i].length;
        }
    }

    __onlyslaves {
        for(int i = 0; i < this->nseq; ++i)
            this->seq[i].offset = ~0;
    }
}

/** @fn void pairwise_t::daemon()
 * @brief Creates threads to respond the nodes' requests.
 */
void pairwise_t::daemon()
{
    std::thread *threads = new std::thread [mpi_data.size];

    for(int i = 1; i < mpi_data.size; ++i)
        threads[i] = std::thread(daemon::run, this, i);

    for(int i = 1; i < mpi_data.size; ++i)
        threads[i].join();

    delete[] threads;
}

/** @fn void daemon::run(const pairwise_t *, int)
 * @brief Sets up the thread to listen to a node's requests.
 * @param pw The pairwise object to be sent to the nodes.
 * @param rank The node rank this thread should respond to.
 */
void daemon::run(const pairwise_t *pw, int rank)
{
    int count;
    MPI_Status status;

    while(true) {
        MPI_Recv(&count, 1, MPI_INT, rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if(status.MPI_TAG == END || count == 0)
            return;

        if(status.MPI_TAG != SYN)
            continue;

        short *seq = new short [count];

        MPI_Recv(seq, count, MPI_SHORT, rank, PLD, MPI_COMM_WORLD, &status);
        response(pw, rank, count, seq);

        delete[] seq;
    }
}

/** @fn void daemon::response(const pairwise_t *, int, int, short[])
 * @brief Sends the requested data to the requesting node.
 * @param pw The pairwise object to be sent to the nodes.
 * @param rank The node rank this thread should respond to.
 * @param nseq The number of requested sequences.
 * @param seq The list of requested sequences.
 */
void daemon::response(const pairwise_t *pw, int rank, int nseq, short seq[])
{
    int buffersize = 0;
    position_t *transformed = new position_t [nseq];

    for(int i = 0; i < nseq; ++i) {
        transformed[i].offset = buffersize;
        transformed[i].length = pw->seq[seq[i]].length;
        buffersize += transformed[i].length;
    }

    char *buffer = new char [buffersize];

    for(int i = 0, off = 0; i < nseq; ++i) {
        memcpy(buffer + off, pw->seqchar + pw->seq[seq[i]].offset, pw->seq[seq[i]].length);
        off += pw->seq[seq[i]].length;
    }

    MPI_Send(&buffersize,     1, MPI_INT,  rank, BSZ, MPI_COMM_WORLD);
    MPI_Send(transformed,  nseq, MPI_2INT, rank, POS, MPI_COMM_WORLD);
    MPI_Send(buffer, buffersize, MPI_CHAR, rank, CHR, MPI_COMM_WORLD);

    delete[] transformed;
    delete[] buffer;
}

/** @fn char *daemon::request(std::vector<uint16_t>&, position_t *, int&)
 * @brief Requests a list of sequences from the master node.
 * @param seqlist The list of requested sequences.
 * @param position The retrieved sequence positions.
 * @param bfisze The retrieved buffer size.
 * @return The retrieved buffer.
 */
char *daemon::request(std::vector<uint16_t>& seqlist, position_t *position, int& bfsize)
{
    MPI_Status status;
    int nseq = seqlist.size();

    printf("daemon::request<%d>(", mpi_data.rank);

    for(uint16_t i : seqlist)
        printf("%d ", i);

    printf("\b)\n");

    MPI_Send(&nseq,             1, MPI_INT,   __master, SYN, MPI_COMM_WORLD);
    MPI_Send(seqlist.data(), nseq, MPI_SHORT, __master, PLD, MPI_COMM_WORLD);

    MPI_Recv(&bfsize,     1, MPI_INT,  __master, BSZ, MPI_COMM_WORLD, &status);
    MPI_Recv(position, nseq, MPI_2INT, __master, POS, MPI_COMM_WORLD, &status);

    char *buffer = new char [bfsize];    
    MPI_Recv(buffer, bfsize, MPI_CHAR, __master, CHR, MPI_COMM_WORLD, &status);

    return buffer;
}

/** @fn void daemon::destroy()
 * @brief Destroys a daemon thread responsible for a slave node's requests.
 */
void daemon::destroy()
{
    int null = 0;

    MPI_Send(&null, 1, MPI_INT, __master, END, MPI_COMM_WORLD);
}
