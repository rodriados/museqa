/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the heuristic's phylogeny module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#include "io.hpp"
#include "pipeline.hpp"
#include "exception.hpp"

#include "phylogeny.cuh"

namespace museqa
{
    namespace module
    {
        namespace pg = museqa::phylogeny;

        /**
         * Execute the module's task when on a pipeline.
         * @param io The pipeline's IO service instance.
         * @return A conduit with the module's processed results.
         */
        auto phylogeny::run(const io::manager& io, const phylogeny::pipe& pipe) const -> phylogeny::pipe
        {
            auto algoname = io.cmd.get("phylogeny", "default");
            const auto conduit = pipeline::convert<phylogeny::previous>(*pipe);

            auto result = pg::run(conduit.distances, conduit.count, algoname);

            auto ptr = new phylogeny::conduit {conduit.db, result};
            return phylogeny::pipe {ptr};
        }

        /**
         * Checks whether command line arguments produce a valid module state.
         * @param io The pipeline's IO service instance.
         * @return Are the given command line arguments valid?
         */
        auto phylogeny::check(const io::manager& io) const -> bool
        {
            auto algoname = io.cmd.get("phylogeny", "default");
            enforce(pg::algorithm::has(algoname), "unknown phylogeny algorithm chosen: '%s'", algoname);

            return true;
        }
    }
}
