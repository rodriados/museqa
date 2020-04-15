/** 
 * Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include <msa.hpp>
#include <mpi.hpp>
#include <cuda.cuh>
#include <parser.hpp>
#include <cmdline.hpp>
#include <encoder.hpp>
#include <database.hpp>
#include <benchmark.hpp>
#include <exception.hpp>

#include <pairwise.cuh>
//#include <phylogeny.cuh>

using namespace msa;

/**
 * The list of command line options available. This list might be increased
 * depending on the options available for the different steps of the software.
 * @since 0.1.1
 */
static const std::vector<cmdline::option> options = {
    {"multigpu",    {"-m", "--multigpu"},   "Try to use multiple devices in a single host."}
,   {"matrix",      {"-x", "--matrix"},     "Choose the scoring matrix to use in pairwise.", true}
,   {"watchdog",    {"-w", "--watchdog"},   "The file to which watchdog notifications are sent to", true}
,   {"pairwise",    {"-1", "--pairwise"},   "Choose the algorithm to use in pairwise module.", true}
,   {"phylogeny",   {"-2", "--phylogeny"},  "Choose the algorithm to use in phylogeny module.", true}
};

namespace msa
{
    namespace step
    {
        /**
         * Parses all files given via command line and shares with all nodes.
         * @return The database with all loaded sequences.
         */
        static auto load() -> database
        {
            database db;
            std::vector<size_t> size;
            std::vector<encoder::block> block;

            onlymaster for(size_t i = 0, n = cmdline::count(); i < n; ++i)
                db.merge(parser::parse(cmdline::get(i)));

            onlymaster for(const auto& entry : db) {
                size.push_back(entry.contents.size());
                block.insert(block.end(), entry.contents.begin(), entry.contents.end());
            }

            size = mpi::broadcast(size);
            block = mpi::broadcast(block);

            onlymaster watchdog::info("loaded a total of <bold>%d</> sequences", db.count());

            onlyslaves for(size_t i = 0, j = 0, n = size.size(); i < n; ++i) {
                db.add(sequence::copy(&block[j], size[i]));
                j += size[i];
            }

            return db;
        }

        /**
         * Runs the first step in the multiple sequence alignment heuristic: the
         * pairwise alignment. This step will align all possible pairs of sequences
         * and give a score to each of these pairs.
         * @param db The database of sequences to be aligned.
         * @return The pairwise manager instance.
         */
        static auto pairwise(const database& db) -> msa::pairwise::manager
        {
            return msa::pairwise::manager::run({
                    db
                ,   cmdline::get("pairwise", std::string ("default"))
                ,   cmdline::get("matrix", std::string ("default"))
                });
        }

        /**
         * Runs the second step in the multiple sequence alignment heuristic: the
         * pseudo-phylogenetic tree construction. This step will group sequences in
         * ways that they can later be definitively aligned.
         * @param pw The pairwise step instance.
         * @return The phylogeny module instance.
         */
        /*static auto phylogeny(const msa::pairwise::manager& pw) -> msa::phylogeny::manager
        {
            return msa::phylogeny::manager::run({
                    pw
                ,   cmdline::get("phylogeny", std::string ("default"))
                });
        }*/
    }

    /**
     * Prints a time report for given task.
     * @param taskname The name of completed task.
     * @param seconds The duration in seconds of given task.
     */
    static void report(const char *taskname, double seconds) noexcept
    {
        #if !__msa(runtime, cython)
            onlymaster watchdog::info("<bold green>%s</> done in <bold>%lf</> seconds", taskname, seconds);
        #endif
    }

    /**
     * Runs the application. This function measures the application's total
     * execution time as well as all its steps.
     * @see report
     */
    static void run() noexcept
    {
        report("total", benchmark::run([]() {
            database db;
            pairwise::manager pw;

            report("loading", benchmark::run(db, step::load));
            report("pairwise", benchmark::run(pw, step::pairwise, db));
            //report("phylogeny", benchmark::run(pg, step::phylogeny, pw));
        }));
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

    cmdline::init(options);
    cmdline::parse(argc, argv);

    bool failure = false;

    try {
        enforce(cmdline::count(), "no input files to align");
        enforce(node::count >= 2, "at least one slave node is needed");

        onlyslaves {
            if(!cuda::device::count())
                throw exception("no compatible device has been found");

            if(cmdline::has("multigpu"))
                cuda::device::select(node::rank - 1);
        }

        msa::run();
    } catch(const exception& except) {
        watchdog::error(except.what());
        failure = true;
    }

    mpi::finalize();

    return failure;
}
