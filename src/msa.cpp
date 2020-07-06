/** 
 * Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include <io.hpp>
#include <msa.hpp>
#include <mpi.hpp>
#include <cuda.cuh>
#include <encoder.hpp>
#include <database.hpp>
#include <benchmark.hpp>
#include <exception.hpp>

#include <bootstrap.hpp>
#include <pairwise.cuh>
#include <phylogeny.cuh>

using namespace msa;

/**
 * The list of command line options available. This list might be increased
 * depending on the options available for the different steps of the software.
 * @since 0.1.1
 */
static const std::vector<io::option> options = {
    {cli::multigpu,  {"-m", "--multigpu"},    "Use multiple devices in a single host."}
,   {cli::report,    {"-r", "--report-only"}, "Print only timing reports and nothing more."}
,   {cli::scoring,   {"-s", "--scoring"},     "Choose pairwise module's scoring matrix.", true}
,   {cli::pairwise,  {"-1", "--pairwise"},    "Choose pairwise module's algorithm.", true}
,   {cli::phylogeny, {"-2", "--phylogeny"},   "Choose phylogeny module's algorithm.", true}
};

namespace msa
{
    /**
     * The global state instance. Here we define the target environment in which
     * the software is running and has been compiled to.
     * @since 0.1.1
     */
    state global_state {
        #if __msa(production)
            environment::production,
        #elif __msa(testing)
            environment::testing,
        #elif __msa(debug)
            environment::debug,
        #else
            environment::dev,
        #endif
    };

    namespace step
    {
        /**
         * Aliases the common conduit type for all pipeline's modules.
         * @since 0.1.1
         */
        using pipe = pointer<pipeline::conduit>;

        /**
         * Initializes and bootstraps the pipeline. Loads all sequence database
         * files given via command line and feeds to following module.
         * @since 0.1.1
         */
        struct bootstrap : public msa::bootstrap::module
        {
            /**
             * Executes the pipeline module's logic.
             * @param io The pipeline's IO service instance.
             * @param pipe The previous module's conduit instance.
             * @return The resulting conduit to send to the next module.
             */
            auto run(const io::service& io, const step::pipe& pipe) const -> step::pipe override
            {
                auto mresult = msa::bootstrap::module::run(io, pipe);
                auto conduit = pipeline::convert<bootstrap>(*mresult);

                onlymaster if(conduit.total > 0)
                    watchdog::info("loaded a total of <bold>%d</> sequences", conduit.total);

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
        struct pairwise : public msa::pairwise::module
        {
            /**
             * Executes the pipeline module's logic.
             * @param io The pipeline's IO service instance.
             * @param pipe The previous module's conduit instance.
             * @return The resulting conduit to send to the next module.
             */
            auto run(const io::service& io, const step::pipe& pipe) const -> step::pipe override
            {
                onlymaster {
                    auto algoname = io.get<std::string>(cli::pairwise, "default");
                    auto tablename = io.get<std::string>(cli::scoring, "default");
                    const auto& conduit = pipeline::convert<pairwise::previous>(*pipe);

                    watchdog::info("chosen pairwise algorithm <bold>%s</>", algoname);
                    watchdog::info("chosen pairwise scoring table <bold>%s</>", tablename);
                    watchdog::init("pairwise", "aligning <bold>%llu</> pairs", utils::nchoose(conduit.total));
                }

                auto mresult = msa::pairwise::module::run(io, pipe);
                onlymaster watchdog::finish("pairwise", "aligned all sequence pairs");

                return mresult;
            }
        };

        /**
         * Executes the heuristic's phylogeny module. This module produces a pseudo-
         * phylogenetic tree, which will then be used to guide the final alignment.
         * @since 0.1.1
         */
        struct phylogeny : public msa::phylogeny::module
        {
            /**
             * Executes the pipeline module's logic.
             * @param io The pipeline's IO service instance.
             * @param pipe The previous module's conduit instance.
             * @return The resulting conduit to send to the next module.
             */
            auto run(const io::service& io, const step::pipe& pipe) const -> step::pipe override
            {
                onlymaster {
                    auto algoname = io.get<std::string>(cli::phylogeny, "default");

                    watchdog::info("chosen phylogeny algorithm <bold>%s</>", algoname);
                    watchdog::init("phylogeny", "producing phylogenetic tree");
                }

                auto mresult = msa::phylogeny::module::run(io, pipe);
                onlymaster watchdog::finish("phylogeny", "phylogenetic tree produced");

                return mresult;
            }
        };
    }

    /**
     * Definition of the heuristic's pipeline stages. These are the modules to run
     * for executing this project's proposed heuristics.
     * @since 0.1.1
     */
    using heuristic = pipeline::runner<
            step::bootstrap
        ,   step::pairwise
        ,   step::phylogeny
        >;

    /**
     * Definition of the heuristic's pipeline's runner. As we're interested on how
     * long our heuristics stages take to run, we will implement an special runner.
     * @since 0.1.1
     */
    class runner : public heuristic
    {
        protected:
            /**
             * Runs the pipeline's modules and reports how long they individually
             * take to execute and send each duration to the watchdog process.
             * @param modules The list pipeline's modules' instances to execute.
             * @param io The IO module service instance.
             * @return The pipeline's final module's result.
             */
            inline auto execute(const pipeline::module *modules[], const io::service& io) const
            -> heuristic::conduit
            {
                auto previous = heuristic::conduit {};
                auto lambda = [&](size_t i) { return std::move(modules[i]->run(io, previous)); };

                for(size_t i = 0; i < heuristic::count; ++i)
                    watchdog::report(modules[i]->name(), benchmark::run(previous, lambda, i));

                return previous;
            }
    };

    /**
     * Runs the application's heuristic's pipeline. This function measures the application's
     * total execution time and reports it to the watchdog process.
     * @param io The IO module service instance.
     * @return Informs whether a failure has occurred during execution.
     */
    static void run(const io::service& io)
    {
        auto runner = msa::runner {};
        auto lambda = [&io, &runner]() { runner.run(io); };

        onlyslaves if(global_state.use_multigpu && global_state.devices_available > 1)
            cuda::device::select(node::rank - 1);

        watchdog::report("total", benchmark::run(lambda));
    }
};

/**
 * Starts, manages and finishes the software's execution.
 * @param argc Number of arguments sent by command line.
 * @param argv The arguments sent by command line.
 * @return The error code for the operating system.
 */
auto main(int argc, char **argv) -> int
try {
    mpi::init(argc, argv);

    auto io = io::service::make(options, argc, argv);

    enforce(io.filecount(), "no input files given");
    enforce(node::count >= 2, "at least one slave node is needed");

    global_state.report_only = io.has(cli::report);
    onlyslaves global_state.use_multigpu = io.has(cli::multigpu);
    onlyslaves global_state.devices_available = cuda::device::count();

    msa::run(io);

    mpi::finalize();
    return 0;
} catch(const std::exception& e) {
    watchdog::error(e.what());

    mpi::finalize();
    return 1;
}
