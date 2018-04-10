/*! \file msa.cpp
 * \brief Parallel Multiple Sequence Alignment main file.
 * \author Rodrigo Siqueira <rodriados@gmail.com>
 * \copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <mpi.h>

#include "msa.hpp"
#include "interface.hpp"

struct msadata gldata;

using namespace std;

/*! \fn main(int, char **)
 * Starts, manages and finishes the software's execution.
 * \param argc Number of arguments sent by command line.
 * \param argv The arguments sent by command line.
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &gldata.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &gldata.nproc);

    if(gldata.rank == 0) {
        argparse(argc, argv);
        cout << gldata.fname << endl;
        //loadfasta(gldata.fname);
        //distribute();
    } else {
        //pairwise();
    }

    MPI_Finalize();
    return 0;
}

/*! \fn finish(int = 0)
 * Aborts the execution and exits the software.
 * \param code Error code to send to operational system.
 */
void finish(int code)
{
    MPI_Finalize();
    exit(code);
}
