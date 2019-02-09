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
#include "database.hpp"
#include "exception.hpp"
#include "stopwatch.hpp"

#include "pairwise.hpp"

/*
 * The list of command line options available. This list might be increased
 * depending on the options available for the different steps of the software.
 */
static const std::vector<cmdline::Option> options = {
    {"m", "multigpu",  "Try to use multiple devices in a single host."}
,   {"x", "matrix",    "Choose the scoring matrix to use.", true}
};

/*
 * Cluster communicator responsible for monitoring whether all nodes are
 * executing successfully.
 */
static mpi::Communicator monitor;

/**
 * Reports success and the execution time of a step.
 * @param name The name of executed step.
 * @param duration The execution time duration.
 */
void report(const std::string& name, const stopwatch::duration::Seconds& duration)
{
    onlymaster
        std::cout << s_bold "[report] " c_green_fg << name << " done in " << duration << " seconds" << std::endl;
}

/**
 * Runs the application. This function measures the application's total
 * execution time as well as all its steps.
 */
void run()
{
    Database db;
    Pairwise pairwise;

    try {
        report("total", stopwatch::run([&db, &pairwise]() {
            report("parsing", stopwatch::run([&db]() { db.addMany(parser::parseMany(cmdline::getPositional())); }));
            //report("pairwise", stopwatch::run([&db, &pairwise]() { pairwise.run(db); mpi::barrier(); }));
            //report("njoining", timer.run([&app](){ njoining(app); }));
            //report("profile-align", timer.run([&app](){ profilealign(app); }));
        }));
    }

    catch(Exception e) {
        error(e.what());
    }

    halt(0);
}

/**
 * Observes and monitors worker processes and quits in case of failure.
 * @return The error code obtained from processes.
 */
int observer()
{
    int code = 0;

    monitor = mpi::communicator::clone();

    onlymaster for(int i = 0; i < node::size; ++i) {
        mpi::receive(code, mpi::any, 0xff01, monitor);
        if(code) break;
    }

    mpi::broadcast(code, node::master, monitor);
    mpi::communicator::free(monitor);
}

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
        error("at least 2 nodes are needed.");

    onlyslaves if(!cuda::device::getCount())
        error("no compatible GPU device has been found.");

    cmdline::init(options);
    cmdline::parse(argc, argv);

    onlyslaves if(cmdline::has("multigpu"))
        cuda::device::setCurrent(node::rank - 1);

    mpi::barrier();

    std::thread worker {run};
    worker.detach();

    int code = observer();

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
