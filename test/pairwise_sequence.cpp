#include <iostream>

#include "src/msa.hpp"
#include "src/fasta.hpp"
#include "src/cluster.hpp"

#include "src/pairwise/sequence.cuh"

#include <string>
#include "src/sequence.hpp"

/*
 * Declaring global variables.
 */
int node::rank = 0;
int node::size = 0;

//extern void call(const pairwise::DeviceSequenceList&);

int main(int argc, char **argv)
{
    cluster::init(argc, argv);

    Fasta ffile("test.fasta");

    pairwise::SequenceList list(ffile);

    /*for(unsigned i = 0; i < list.getCount(); ++i)
        std::cout << list[i] << std::endl;*/

    uint16_t selectlist[] = {0,2,4,7};
    pairwise::dSequenceList compressed = list.select(selectlist, 4);

    //std::cout << compressed << std::endl;

    onlymaster for(unsigned i = 0; i < compressed.getCount(); ++i)
        std::cout << *(reinterpret_cast<const pairwise::dSequence*>(&compressed[i])) << std::endl;

    /*pairwise::DeviceSequenceList dlist = compressed.toDevice();

    call(dlist);*/

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
