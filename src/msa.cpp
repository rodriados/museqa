/** @file msa.cpp
 * @brief Parallel Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <mpi.h>

#include "msa.h"
#include "gpu.hpp"
#include "fasta.hpp"
#include "interface.hpp"

#include "pairwise/distribute.hpp"
#include "pairwise/pairwise.hpp"

mpidata_t mpi_data;
unsigned char verbose = 0;

extern clidata_t cli_data;

/** @fn int main(int, char **)
 * @brief Starts, manages and finishes the software's execution.
 * @param argc Number of arguments sent by command line.
 * @param argv The arguments sent by command line.
 * @return The error code for the operating system.
 */
int main(int argc, char **argv)
{
    fasta_t fasta;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_data.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_data.size);

    if(!gpu::check())
        finish(NOGPUFOUND);

    cli::parse(argc, argv);
    MPI_Barrier(MPI_COMM_WORLD);
    
    __onlymaster fasta.load(cli_data.fname);

    pairwise::sync(fasta);
    pairwise::scatter();
    pairwise::prepare();
    pairwise::pairwise();
    pairwise::clean();

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
,   "invalid argument."         // INVALIDARG
,   "no GPU device detected."   // NOGPUFOUND
,   "GPU runtime error."        // CUDAERROR
};

/** @fn void finish(errornum_t)
 * @brief Aborts the execution and kills all processes.
 * @param code Code of detected error.
 */
void finish(errornum_t code)
{
    if(code) {
        std::cerr << MSA ": fatal error: " << error_str[code] << std::endl;
    }

    MPI_Abort(MPI_COMM_WORLD, MPI_SUCCESS);
    exit(0);
}
