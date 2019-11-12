/** 
 * Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include "msa.hpp"
#include "mpi.hpp"
#include "cuda.cuh"
#include "parser.hpp"
#include "cmdline.hpp"
#include "encoder.hpp"
#include "database.hpp"
#include "benchmark.hpp"
#include "exception.hpp"

#include "pairwise.cuh"
#include "phylogeny.cuh"

/**
 * The list of command line options available. This list might be increased
 * depending on the options available for the different steps of the software.
 * @since 0.1.1
 */
static const std::vector<cmdline::Option> options = {
    {"m", "multigpu",   "Try to use multiple devices in a single host."}
,   {"x", "matrix",     "Choose the scoring matrix to use in pairwise.", true}
,   {"1", "pairwise",   "Choose the algorithm to use in pairwise module.", true}
,   {"2", "phylogeny",  "Choose the algorithm to use in phylogeny module.", true}
};

namespace msa
{
    static Database db;         /// The database of sequences to align.
    static Pairwise pw;         /// The pairwise module manager.
    static Phylogeny pg;        /// The phylogenetics module manager.

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
            blockList.insert(blockList.end(), db[i].begin(), db[i].end());
        }

        mpi::broadcast(sizeList);
        mpi::broadcast(blockList);

        onlyslaves for(size_t i = 0, j = 0, n = sizeList.size(); i < n; ++i) {
            db.add({&blockList[j], sizeList[i]});
            j += sizeList[i];
        }

        onlymaster info("loaded a total of %d sequences", db.getCount());
    }

    /**
     * Runs the first step in the multiple sequence alignment heuristic: the
     * pairwise alignment. This step will align all possible pairs of sequences
     * and give a score to each of these pairs.
     * @param pw The pairwise manager instance.
     * @param db The database of sequences to be aligned.
     */
    static void pwrun(const Pairwise& pw, const Database& db)
    {
        pw.run({
            db
        ,   cmdline::get<std::string>("pairwise", "needleman")
        ,   cmdline::get<std::string>("matrix", "blosum62")
        });
    }

    /**
     * Runs the second step in the multiple sequence alignment heuristic: the
     * pseudo-phylogenetic tree construction. This step will group sequences in
     * ways that they can later be definitively aligned.
     * @param pg The phylogeny module instance.
     * @param pw The pairwise step instance.
     */
    static void pgrun(const Phylogeny& pg, const Pairwise& pw)
    {
        pg.run({
            pw
        ,   cmdline::get<std::string>("phylogeny", "njoining")
        });
    }

    /**
     * Runs the application. This function measures the application's total
     * execution time as well as all its steps.
     * @see watch
     */
    static void run() noexcept
    {
        try {
            msa::report("total", benchmark::run([]() {
                msa::report("loading", benchmark::run(load, db));
                msa::report("pairwise", benchmark::run(pwrun, pw, db));
                msa::report("phylogeny", benchmark::run(pgrun, pg, pw));
            }));
        }

        catch(Exception e) {
            msa::error(e.what());
        }
    }

    /**
     * Halts the whole software's execution and exits with given code.
     * @param code The exit code.
     * @since 0.1.1
     */
    void halt(uint8_t code) noexcept
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

    if(node::size < 2)
        msa::error("at least 2 nodes are needed.");

    onlyslaves if(!cuda::device::getCount())
        msa::error("no compatible GPU device has been found.");

    onlyslaves if(cmdline::has("multigpu"))
        cuda::device::setCurrent(node::rank - 1);

    msa::run();
    mpi::finalize();

    return 0;
}
