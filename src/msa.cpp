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
#include "exception.hpp"
#include "stopwatch.hpp"

#include "pairwise.hpp"
//#include "phylogeny.hpp"

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
    static Pairwise pw;         /// The pairwise step manager.
    //static Phylogeny pg;         /// The phylogenetics step manager.

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

        onlymaster info("loaded a total of", db.getCount(), "sequences");
    }

    /**
     * Runs the first step in the multiple sequence alignment heuristic: the
     * pairwise alignment. This step will align all possible pairs of sequences
     * and give a score to each of these pairs.
     * @param pw The pairwise manager instance.
     * @param db The database of sequences to be aligned.
     */
    static void pwrun(Pairwise& pw, Database& db)
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
    /*static void pgrun(Phylogeny& pg, Pairwise& pw)
    {
        pg.run({
            pw
        ,   cmdline::get<std::string>("phylogeny", "njoining")
        });
    }*/

    /**
     * Reports success and the execution time of a step.
     * @param name The name of executed step.
     * @param duration The execution time duration.
     */
    inline void report(const std::string& name, const stopwatch::duration::Seconds& duration)
    {
        onlymaster msa::log(std::cout, s_bold "[report]" c_green_fg, name, s_reset, duration, "seconds");
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
//                report("phylogeny", stopwatch::run(pgrun, pg, pw));
            }));
        }

        catch(Exception e) {
            error(e.what());
        }
    }

    /**
     * Halts the software execution in all nodes with a code.
     * @param code The exit code.
     * @return The code to operational system.
     */
    void halt(uint8_t code)
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
        error("at least 2 nodes are needed.");

    onlyslaves if(!cuda::device::getCount())
        error("no compatible GPU device has been found.");

    onlyslaves if(cmdline::has("multigpu"))
        cuda::device::setCurrent(node::rank - 1);

    msa::run();
    mpi::finalize();

    return 0;
}
