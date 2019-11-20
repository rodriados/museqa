/** 
 * Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
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

/**
 * The list of command line options available. This list might be increased
 * depending on the options available for the different steps of the software.
 * @since 0.1.1
 */
static const std::vector<cmdline::option> options = {
    {"multigpu",    {"-m", "--multigpu"},   "Try to use multiple devices in a single host."}
,   {"matrix",      {"-x", "--matrix"},     "Choose the scoring matrix to use in pairwise.", true}
,   {"pairwise",    {"-1", "--pairwise"},   "Choose the algorithm to use in pairwise module.", true}
,   {"phylogeny",   {"-2", "--phylogeny"},  "Choose the algorithm to use in phylogeny module.", true}
};

namespace msa
{
    /**
     * Parses all files given via command line and shares with all nodes.
     * @return The database with all loaded sequences.
     */
    static auto load() -> database
    {
        database db;
        std::vector<size_t> lsize;
        std::vector<encoder::block> lblock;

        onlymaster for(size_t i = 0, n = cmdline::count(); i < n; ++i)
            db.add_many(parser::parse(cmdline::get(i)));

        onlymaster for(size_t i = 0, n = db.count(); i < n; ++i) {
            lsize.push_back(db[i].size());
            lblock.insert(lblock.end(), db[i].begin(), db[i].end());
        }

        mpi::broadcast(lsize);
        mpi::broadcast(lblock);

        onlymaster info("loaded a total of %d sequences", db.count());

        onlyslaves for(size_t i = 0, j = 0, n = lsize.size(); i < n; ++i) {
            db.add(sequence::copy(&lblock[j], lsize[i]));
            j += lsize[i];
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
    static auto rpairwise(const database& db) -> pairwise::manager
    {
        return pairwise::manager::run({
                db
            ,   cmdline::get("pairwise", std::string ("needleman"))
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
    /*static auto rphylogeny(const pairwise::manager& pw) -> phylogeny::manager
    {
        return phylogeny::manager::run({
                pw
            ,   cmdline::get("phylogeny", std::string ("default"))
            });
    }*/

    /**
     * Runs the application. This function measures the application's total
     * execution time as well as all its steps.
     * @see watch
     */
    static void run() noexcept
    try {
        database db;
        pairwise::manager pw;
        //phylogeny::manager pg;

        msa::report("total", benchmark::run([&]() {
            msa::report("loading", benchmark::run(db, load));
            msa::report("pairwise", benchmark::run(pw, rpairwise, db));
            //msa::report("phylogeny", benchmark::run(pg, rphylogeny, pw));
        }));
    }

    catch(const exception& e) {
        msa::error(e.what());
    }

    /**
     * Halts the whole software's execution and exits with given code.
     * @param code The exit code.
     * @since 0.1.1
     */
    [[noreturn]] void halt(uint8_t code) noexcept
    {
        mpi::finalize();
        exit(code);
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

    if(node::count < 2)
        msa::error("at least 2 nodes are needed");

    onlyslaves if(!cuda::device::count())
        msa::error("no compatible gpu device has been found");

    onlyslaves if(cmdline::has("multigpu"))
        cuda::device::select(node::rank - 1);

    msa::run();
    mpi::finalize();

    return 0;
}
