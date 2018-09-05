/** 
 * Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <string>
#include <vector>

#include "msa.hpp"
#include "input.hpp"
#include "fasta.hpp"
#include "device.cuh"
#include "cluster.hpp"
#include "benchmark.hpp"

#include "pairwise.hpp"

/**
 * Starts, manages and finishes the software's execution.
 * @param argc Number of arguments sent by command line.
 * @param argv The arguments sent by command line.
 * @return The error code for the operating system.
 */
int main(int argc, char **argv)
{
    cmd.init({
        {"h", "help",     "Displays this help menu." }
    ,   {"v", "version",  "Displays the software version." }
    ,   {"b", "verbose",  "Activates verbose mode." }
    ,   {"m", "multigpu", "Try to use multiple devices in a single host."}
    ,   {"f", "file",     "File to be processed.", "filename"}
    ,   {"x", "matrix",   "Choose the scoring matrix to use.", "matrix"}
    }, {"filename"});

    cluster::init(argc, argv);

    if(node::isSlave() && !device::exists())
        finalize(DeviceError::noGPU());

    cmd.parse(argc, argv);
    cluster::sync();

    Benchmark bm;

    Fasta fasta(cmd.get("filename"));
    cluster::sync(); bm.step();

    Pairwise pwise = Pairwise::run(fasta);
    cluster::sync(); bm.step();

    /*NJoining nj = NJoining::run(pwise);
    cluster::sync(); bm.step(); */

    cluster::finalize();

    onlymaster {
        const std::vector<std::string> steps = {"distribution", "pairwise"};

        for(int i = 0, n = bm.getCount(); i < n; ++i)
            std::cerr << style(bold, "[msa:report] ") << steps[i] << " step evaluated in "
                << bm.getStep(i) << " seconds." << std::endl;

        std::cerr << style(bold, "[msa:report] ") "total elapsed time: " << bm.elapsed() << " seconds." << std::endl;
    }

    return 0;
}

/**
 * Aborts the execution and kills all processes.
 * @param error Error detected during execution.
 */
[[noreturn]]
void finalize(Error error)
{
    if(node::isMaster() && !error.msg.empty()) {
        std::cerr
            << style(bold, MSA) ": " s_bold c_red_fg "fatal error: " s_reset << error.msg << std::endl
            << "execution has terminated." << std::endl;
    }

    cluster::finalize();
    exit(0);
}
