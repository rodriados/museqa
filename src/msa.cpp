/** @file msa.cpp
 * @brief Parallel Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <mpi.h>

#include "msa.h"
#include "interface.hpp"

struct mpi_data mpi_data;
struct msa_data msa_data;

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
        parse_cli(argc, argv);
        //fasta::load(msa_data.fname);
        //distribute();
    } else {
        //pairwise();
    }

    MPI_Finalize();
    return 0;
}

/** @fn void finish(int)
 * @brief Aborts the execution and exits the software.
 * @param code Error code to send to operational system.
 */
void finish(int code)
{
    MPI_Finalize();
    exit(code);
}
