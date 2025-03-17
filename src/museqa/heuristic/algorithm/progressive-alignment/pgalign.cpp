/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the heuristic's profile-aligner module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#include "io.hpp"
#include "pipeline.hpp"
#include "exception.hpp"

#include "pgalign.cuh"

namespace museqa
{
    namespace module
    {
        namespace pa = museqa::pgalign;

        /**
         * Execute the module's task when on a pipeline.
         * @param io The pipeline's IO service instance.
         * @param pipe The previous module's conduit.
         * @return A conduit with the module's processed results.
         */
        auto pgalign::run(const io::manager& io, pipeline::pipe& pipe) const -> pipeline::pipe
        {
            auto algoname = io.cmd.get("pgalign", "default");
            auto previous = pipeline::convert<pgalign::previous>(pipe);

            auto result = pa::run(previous->db, previous->tree, previous->total, algoname);
            auto ptr = new pgalign::conduit {result};

            return pipeline::pipe {ptr};
        }

        /**
         * Checks whether command line arguments produce a valid module state.
         * @param io The pipeline's IO service instance.
         * @return Are the given command line arguments valid?
         */
        auto pgalign::check(const io::manager& io) const -> bool
        {
            auto algoname = io.cmd.get("pgalign", "default");
            enforce(pa::algorithm::has(algoname), "unknown pgalign algorithm chosen: '%s'", algoname);

            return true;
        }
    }
}
