/** 
 * Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <string>

#include "msa.hpp"
#include "input.hpp"
#include "fasta.hpp"
#include "device.cuh"
#include "cluster.hpp"

#include "pairwise.cuh"

/*
 * Declaring global variables.
 */
int node::rank = 0;
int node::size = 0;

/**
 * Starts, manages and finishes the software's execution.
 * @param argc Number of arguments sent by command line.
 * @param argv The arguments sent by command line.
 * @return The error code for the operating system.
 */
int main(int argc, char **argv)
{
    cluster::init(&argc, &argv);

    if(node::isslave() && !device::check())
        finalize(ErrorCode::NoGPU);

    clidata.parse(argc, argv);  
    clidata.checkhelp();
    cluster::synchronize();

    Fasta fasta;
    fasta.load(clidata.get(ParamCode::File));
    cluster::synchronize();
    
    Pairwise pairwise(fasta);
    pairwise.process();
    
    /*MPI_Barrier(MPI_COMM_WORLD);

    delete fasta;

    //__onlymaster pairwise->gather();
    delete pairwise;

    MPI_Barrier(MPI_COMM_WORLD);*/

    cluster::finalize();
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
    onlymaster {
        if(code != ErrorCode::Success)
            std::cerr
                << style(bold, __msa__) ": "
                << style(bold, fg(red, "fatal error")) ": "
                << error_str[static_cast<int>(code)] << std::endl;
    }

    cluster::finalize();
    exit(0);
}
