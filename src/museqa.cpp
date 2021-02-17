/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The software's entry point.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include "io.hpp"
#include "mpi.hpp"
#include "cuda.cuh"
#include "museqa.hpp"
#include "encoder.hpp"
#include "database.hpp"
#include "benchmark.hpp"
#include "exception.hpp"

#include "bootstrap.hpp"
#include "pairwise.cuh"
#include "phylogeny.cuh"
#include "pgalign.cuh"

using namespace museqa;

/**
 * The list of command line options available. This list might be increased
 * depending on the options available for the different steps of the software.
 * @since 0.1.1
 */
static const std::vector<terminal::option> options = {
    {"multigpu",      {"-m", "--multigpu"},      "Use multiple devices in a single host if possible."}
,   {"report-only",   {"-r", "--report-only"},   "Print only timing reports and nothing else."}
,   {"gpu-id",        {"-d", "--device"},        "Picks the GPU to be used on hosts with more than one.", true}
,   {"scoring-table", {"-s", "--scoring-table"}, "The scoring table name or file to align sequences with.", true}
,   {"pairwise",      {"-1", "--pairwise"},      "Picks the algorithm to use within the pairwise module.", true}
,   {"phylogeny",     {"-2", "--phylogeny"},     "Picks the algorithm to use within the phylogeny module.", true}
,   {"pgalign",       {"-3", "--pgalign"},       "Picks the algorithm to use within the profile-aligner.", true}
};

namespace museqa
{
    /**
     * The global state instance. Here we define the target environment in which
     * the software is running and has been compiled to.
     * @since 0.1.1
     */
    state global_state {
      #if defined(__museqa_environment)
        static_cast<env>(__museqa_environment)
      #else
        env::production
      #endif
    };

    namespace heuristic
    {
        template <typename T>
        struct timer : public pipeline::middleware<T>
        {
            /**
             * Executes the pipeline module's logic.
             * @param io The pipeline's IO service instance.
             * @param pipe The previous module's conduit instance.
             * @return The resulting conduit to send to the next module.
             */
            auto run(const io::manager& io, pipeline::pipe& pipe) const -> pipeline::pipe override
            {
                pipeline::pipe mresult;

                const auto lambda = [&]() { return std::move(this->next(io, pipe)); };
                const auto duration = benchmark::run(mresult, lambda);

                watchdog::report(this->name(), duration);
                return mresult;
            }
        };

        /**
         * Initializes and bootstraps the pipeline. Loads all sequence database
         * files given via command line and feeds to following module.
         * @since 0.1.1
         */
        struct bootstrap : public museqa::module::bootstrap
        {
            /**
             * Executes the pipeline module's logic.
             * @param io The pipeline's IO service instance.
             * @param pipe The previous module's conduit instance.
             * @return The resulting conduit to send to the next module.
             */
            auto run(const io::manager& io, pipeline::pipe& pipe) const -> pipeline::pipe override
            {
                auto mresult = museqa::module::bootstrap::run(io, pipe);
                auto conduit = pipeline::convert<bootstrap>(mresult);

                onlymaster if(conduit->total > 0)
                    watchdog::info("loaded a total of <bold>%d</> sequences", conduit->total);

                return mresult;
            }

            /**
             * Returns an string identifying the module's name.
             * @return The module's name.
             */
            inline auto name() const -> const char * override
            {
                return "loading";
            }
        };

        /**
         * Executes the heuristic's pairwise alignment module. This pairwise alignment
         * module produces a distance matrix of sequences in relation to all others.
         * @since 0.1.1
         */
        struct pairwise : public museqa::module::pairwise
        {
            /**
             * Executes the pipeline module's logic.
             * @param io The pipeline's IO service instance.
             * @param pipe The previous module's conduit instance.
             * @return The resulting conduit to send to the next module.
             */
            auto run(const io::manager& io, pipeline::pipe& pipe) const -> pipeline::pipe override
            {
                onlymaster {
                    auto algoname = io.cmd.get("pairwise", "default");
                    auto tablename = io.cmd.get("scoring-table", "default");

                    auto previous = pipeline::convert<pairwise::previous>(pipe);

                    watchdog::info("chosen pairwise algorithm <bold>%s</>", algoname);
                    watchdog::info("chosen pairwise scoring table <bold>%s</>", tablename);
                    watchdog::init("pairwise", "aligning <bold>%llu</> pairs", utils::nchoose(previous->total));
                }

                auto mresult = museqa::pairwise::module::run(io, pipe);
                onlymaster watchdog::finish("pairwise", "aligned all sequence pairs");

                return mresult;
            }
        };

        /**
         * Executes the heuristic's phylogeny module. This module produces a pseudo-
         * phylogenetic tree, which will then be used to guide the final alignment.
         * @since 0.1.1
         */
        struct phylogeny : public museqa::module::phylogeny
        {
            /**
             * Executes the pipeline module's logic.
             * @param io The pipeline's IO service instance.
             * @param pipe The previous module's conduit instance.
             * @return The resulting conduit to send to the next module.
             */
            auto run(const io::manager& io, pipeline::pipe& pipe) const -> pipeline::pipe override
            {
                onlymaster {
                    auto algoname = io.cmd.get("phylogeny", "default");

                    watchdog::info("chosen phylogeny algorithm <bold>%s</>", algoname);
                    watchdog::init("phylogeny", "producing phylogenetic tree");
                }

                auto mresult = museqa::phylogeny::module::run(io, pipe);
                onlymaster watchdog::finish("phylogeny", "phylogenetic tree produced");

                return mresult;
            }
        };

        /**
         * Executes the heuristic's profile-aligner module. This module produces
         * the final global alignment of all sequences given as input.
         * @since 0.1.1
         */
        struct pgalign : public museqa::module::pgalign
        {
            /**
             * Executes the pipeline module's logic.
             * @param io The pipeline's IO service instance.
             * @param pipe The previous module's conduit instance.
             * @return The resulting conduit to send to the next module.
             */
            auto run(const io::manager& io, pipeline::pipe& pipe) const -> pipeline::pipe override
            {
                onlymaster {
                    auto algoname = io.cmd.get("pgalign", "default");

                    watchdog::info("chosen profile-aligner algorithm <bold>%s</>", algoname);
                    watchdog::init("pgalign", "aligning sequence profiles");
                }

                auto mresult = museqa::pgalign::module::run(io, pipe);
                onlymaster watchdog::finish("pgalign", "sequences alignment is completed");

                return mresult;
            }
        };
    }

    /**
     * Definition of the heuristic's pipeline stages. These are the modules to run
     * for executing this project's proposed heuristics.
     * @since 0.1.1
     */
    using runner = pipeline::runner<
            heuristic::timer<heuristic::bootstrap>
        ,   heuristic::timer<heuristic::pairwise>
        ,   heuristic::timer<heuristic::phylogeny>
        ,   heuristic::timer<heuristic::pgalign>
        >;

    /**
     * Runs the application's heuristic's pipeline. This function measures the application's
     * total execution time and reports it to the watchdog process.
     * @param io The IO module service instance.
     * @return Informs whether a failure has occurred during execution.
     */
    static void run(const io::manager& io)
    {
        auto runner = museqa::runner {};
        auto lambda = [&io, &runner]() { runner.run(io); };

        onlyslaves if(global_state.local_devices > 0) {
            const auto rank  = node::rank - 1;
            const auto count = global_state.local_devices;
            const auto gpuid = io.cmd.get<unsigned>("gpu-id", cuda::device::init);
            cuda::device::select((global_state.use_multigpu ? gpuid + rank : gpuid) % count);
        }

        watchdog::report("total", benchmark::run(lambda));
    }
};

/**
 * Starts, manages and finishes the software's execution.
 * @param argc Number of arguments sent by command line.
 * @param argv The arguments sent by command line.
 * @return The error code for the operating system.
 */
int main(int argc, char **argv)
try {
    mpi::init(argc, argv);

    auto io = io::manager::make(options, argc, argv);

    enforce(!io.cmd.all().empty(), "no input files given");
    enforce(node::count >= 2, "at least one slave node is needed");

    onlyslaves try {
        global_state.local_devices = cuda::device::count();
    } catch(const cuda::exception&) {
        global_state.local_devices = 0;
    }

    global_state.report_only = io.cmd.has("report-only");
    onlyslaves global_state.use_multigpu = io.cmd.has("multigpu");
    global_state.use_devices = mpi::allreduce(global_state.local_devices, mpi::op::min);

    museqa::run(io);
    mpi::finalize();

    return 0;
} catch(const std::exception& e) {
    watchdog::error(e.what());
    mpi::finalize();

    return 1;
}
