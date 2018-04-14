/** @file msa.cpp
 * @brief Parallel Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <mpi.h>

#include "msa.h"
#include "fasta.h"
#include "distribute.h"
#include "interface.hpp"

short verbose = 0;
mpidata_t mpi_data;
msadata_t msa_data;

extern clidata_t cli_data;

using namespace std;

/** @fn int main(int, char **)
 * @brief Starts, manages and finishes the software's execution.
 * @param argc Number of arguments sent by command line.
 * @param argv The arguments sent by command line.
 * @return The error code for the operating system.
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_data.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_data.nproc);

    if(mpi_data.rank == 0) {
        parsecli(argc, argv);
        loadfasta(cli_data.fname);
        distribute();
        freefasta();
    } else {
        collect(0);
        //pairwise();
        freefasta();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

/** @var char *error_str[]
 * @brief Lists the error messages to be shown when finishing.
 */
static const char *error_str[] = {
    ""                          // NOERROR
,   "no input file."            // NOFILE
,   "file could not be read."   // INVALIDFILE
};

/** @fn void finish(errornum_t)
 * @brief Aborts the execution and exits the software.
 * @param code Error code to send to operational system.
 */
void finish(errornum_t code)
{
    if(code)
        cerr << MSA ": fatal error: " << error_str[code] << endl;

    MPI_Finalize();
    exit(0);
}
