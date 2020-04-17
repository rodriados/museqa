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
    {"multigpu",    {"-m", "--multigpu"},    "Try to use multiple devices in a single host."}
,   {"matrix",      {"-x", "--matrix"},      "Choose the scoring matrix to use in pairwise.", true}
,   {"pairwise",    {"-1", "--pairwise"},    "Choose the algorithm to use in pairwise module.", true}
,   {"phylogeny",   {"-2", "--phylogeny"},   "Choose the algorithm to use in phylogeny module.", true}
,   {"report",      {"-r", "--report-only"}, "Indicates whether should only print timing reports"}
};

namespace msa
{
    /**
     * Keeps the state of whether we should only print time report statuses.
     * @since 0.1.1
     */
    bool watchdog::report_only = false;

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
     * Runs the application. This function measures the application's total
     * execution time as well as all its steps.
     * @return Informs whether a failure has occurred during execution.
     */
    static bool run() noexcept
    try {
        watchdog::report("total", benchmark::run([]() {
            database db;
            pairwise::manager pw;

            watchdog::report("loading", benchmark::run(db, step::load));
            watchdog::report("pairwise", benchmark::run(pw, step::pairwise, db));
            //watchdog::report("phylogeny", benchmark::run(pg, step::phylogeny, pw));
        }));

        return 0;
    } catch(const exception& e) {
        watchdog::error(e.what());
        return 1;
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
    int code = 0;

    mpi::init(argc, argv);

    try {
        cmdline::init(options);
        cmdline::parse(argc, argv);

        watchdog::report_only = cmdline::has("report");

        enforce(cmdline::count(), "no input files to align");
        enforce(node::count >= 2, "at least one slave node is needed");

        onlyslaves if(!cuda::device::count())
            throw exception("no compatible device has been found");

        onlyslaves if(cmdline::has("multigpu"))
            cuda::device::select(node::rank - 1);

        code = msa::run();
    } catch(const exception& e) {
        watchdog::error(e.what());
        code = 1;
    }

    mpi::finalize();

    return code;
}
