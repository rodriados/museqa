/** 
 * Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <string>
#include <mpi.h>

#include "msa.hpp"
#include "input.hpp"
#include "fasta.hpp"
#include "device.cuh"

#include "pairwise.cuh"

/*
 * Declaring global variables.
 */
Input clidata;
NodeInfo nodeinfo;

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

    if(__isslave && !Device::check())
        finalize(ErrorCode::NoGPU);

    clidata.parse(argc, argv);  
    clidata.checkhelp();
    MPI_Barrier(MPI_COMM_WORLD);

    Fasta fasta;
    Pairwise pairwise;

    fasta.load(clidata.get(ParamCode::File));    
    //pairwise.process(fasta);
    
    /*MPI_Barrier(MPI_COMM_WORLD);

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
,   "input file is invalid."        // InvalidFile
,   "no GPU device detected."       // NoGPU
,   "GPU runtime error."            // CudaError
};

/**
 * Aborts the execution and kills all processes.
 * @param code Code of detected error.
 */
[[noreturn]]
void finalize(ErrorCode code)
{
    __onlymaster {
        if(code != ErrorCode::Success)
            std::cerr
                << __bold MSA __reset ": "
                << __bold __redfg "fatal error" __reset ": "
                << error_str[static_cast<int>(code)] << std::endl;
    }

    MPI_Finalize();
    exit(0);
}
