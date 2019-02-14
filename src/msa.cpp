/** 
 * Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "msa.hpp"
#include "mpi.hpp"
#include "cuda.cuh"
#include "parser.hpp"
#include "cmdline.hpp"
#include "encoder.hpp"
#include "database.hpp"
#include "exception.hpp"
#include "stopwatch.hpp"

#include "pairwise.hpp"

/**
 * The list of command line options available. This list might be increased
 * depending on the options available for the different steps of the software.
 * @since 0.1.1
 */
static const std::vector<cmdline::Option> options = {
    {"m", "multigpu",  "Try to use multiple devices in a single host."}
,   {"x", "matrix",    "Choose the scoring matrix to use in pairwise.", true}
,   {"1", "pw-alg",    "Choose the algorithm to use in pairwise.", true}
};

/**
 * Cluster communicator responsible for monitoring whether all nodes are
 * executing successfully.
 * @since 0.1.1
 */
static mpi::Communicator monitor;

namespace msa
{
    static Database db;         /// The database of sequences to align.
    static Pairwise pw;         /// The pairwise step manager.

    /**
     * Parses all files given via command line and shares with all nodes.
     * @param db The database to load sequences into.
     */
    static void load(Database& db)
    {
        std::vector<size_t> sizeList;
        std::vector<encoder::EncodedBlock> blockList;

        onlymaster db.addMany(parser::parseMany(cmdline::getPositional()));

        onlymaster for(size_t i = 0, n = db.getCount(); i < n; ++i) {
            sizeList.push_back(db[i].getSize());
            blockList.insert(blockList.end(), db[i].getBuffer(), db[i].getBuffer() + db[i].getSize());
        }

        mpi::broadcast(sizeList);
        mpi::broadcast(blockList);

        onlyslaves for(size_t i = 0, j = 0, n = sizeList.size(); i < n; ++i) {
            db.add({&blockList[j], sizeList[i]});
            j += sizeList[i];
        }
    }

    /**
     * Runs the first step in the multiple sequence alignment heuristic: the
     * pairwise alignment. This step will align all possible pairs of sequences
     * and give a score to each of these pairs.
     * @param pw The pairwise manager instance.
     * @param db The database of sequences to be aligned.
     * @return 0.1.1
     */
    static void pwrun(Pairwise& pw, const Database& db)
    {
        pw.run({
            db
        ,   cmdline::get("pw-alg", "needleman")
        ,   cmdline::get("matrix", "blosum62")
        });
    }

    /**
     * Reports success and the execution time of a step.
     * @param name The name of executed step.
     * @param duration The execution time duration.
     */
    inline void report(const std::string& name, const stopwatch::duration::Seconds& duration)
    {
        onlymaster {
            std::cout << s_bold "[report] " c_green_fg << name << " done in " << duration << " seconds" << std::endl;
        }
    }

    /**
     * Runs the application. This function measures the application's total
     * execution time as well as all its steps.
     * @see watch
     */
    static void run()
    {
        try {
            report("total", stopwatch::run([]() {
                report("loading", stopwatch::run(load, db));
                report("pairwise", stopwatch::run(pwrun, pw, db));
            }));
        }

        catch(Exception e) {
            error(e.what());
            halt(1);
        }

        halt(0);
    }

    /**
     * Observes and monitors worker processes and quits in case of failure.
     * @return The error code obtained from processes.
     */
    static int watch()
    {
        int code = 0;

        monitor = mpi::communicator::clone();

        onlymaster for(int i = 0; i < node::size; ++i) {
            mpi::receive(code, mpi::any, 0xff01, monitor);
            if(code) break;
        }

        mpi::broadcast(code, node::master, monitor);
        mpi::communicator::free(monitor);

        return code;
    }
};

/**
 * Starts, manages and finishes the software's execution.
 * @param argc Number of arguments sent by command line.
 * @param argv The arguments sent by command line.
 * @return The error code for the operating system.
 */
int main(int argc, char **argv)
{
    mpi::init(argc, argv);

    if(node::size < 2)
        error("At least 2 nodes are needed.");

    onlyslaves if(!cuda::device::getCount())
        error("No compatible GPU device has been found.");

    cmdline::init(options);
    cmdline::parse(argc, argv);

    onlyslaves if(cmdline::has("multigpu"))
        cuda::device::setCurrent(node::rank - 1);

    mpi::barrier();

    std::thread worker {msa::run};
    worker.detach();

    int code = msa::watch();

    mpi::finalize();

    return code;
}

/**
 * Halts the software execution in all nodes with a code.
 * @param code The exit code.
 * @return The code to operational system.
 */
void halt(uint8_t code)
{
    if(monitor == mpi::communicator::null) {
        mpi::finalize();
        exit(code);
    }

    mpi::send(code, node::master, 0xff01, monitor);
}
