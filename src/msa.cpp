/** 
 * Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <mpi.h>

#include "msa.hpp"
#include "input.hpp"
#include "device.cuh"

/*
 * Declaring global variables.
 */
Input cmdinput;
NodeInfo nodeinfo;
bool verbose = false;

/**
 * Starts, manages and finishes the software's execution.
 * @param argc Number of arguments sent by command line.
 * @param argv The arguments sent by command line.
 * @return The error code for the operating system.
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &nodeinfo.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nodeinfo.size);

    if(__isslave() && !Device::check())
        finalize(ErrorCode::NoGPU);

    cmdinput.parse(argc, argv);
    verbose = cmdinput.has(InputCommand::Verbose);
    MPI_Barrier(MPI_COMM_WORLD);
    
    /*fasta_t *fasta = new fasta_t;
    pairwise_t *pairwise = new pairwise_t;

    __onlymaster fasta->read(cli_data.fname);    
                 pairwise->load(fasta);
    __onlymaster pairwise->daemon();
    __onlyslaves pairwise->pairwise();

    MPI_Barrier(MPI_COMM_WORLD);

    delete fasta;

    //__onlymaster pairwise->gather();
    delete pairwise;

    MPI_Barrier(MPI_COMM_WORLD);*/

    MPI_Finalize();
    return 0;
}

/*
 * Lists the error messages to be shown when finishing.
 */
static const char *error_str[] = {
    ""                              // Success
,   "no input file."                // NoFile
,   "input file is invalid."        // InvalidFile
,   "invalid argument."             // InvalidArg
,   "no GPU device detected."       // NoGPU
,   "GPU runtime error."            // CudaError
};

/**
 * Aborts the execution and kills all processes.
 * @param code Code of detected error.
 */
void finalize(ErrorCode code)
{
    __onlymaster {
        if(code != ErrorCode::Success)
            std::cerr
                << __bold MSA __reset ": "
                << __bold __redfg "fatal error" __reset ": "
                << error_str[code] << std::endl;
    }

    MPI_Finalize();
    exit(0);
}
