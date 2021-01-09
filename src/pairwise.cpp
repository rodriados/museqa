/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the heuristic's pairwise module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#include <string>

#include "io.hpp"
#include "pipeline.hpp"
#include "exception.hpp"

#include "pairwise.cuh"

namespace museqa
{
    namespace module
    {
        namespace pw = museqa::pairwise;

        /**
         * Execute the module's task when on a pipeline.
         * @param io The pipeline's IO service instance.
         * @return A conduit with the module's processed results.
         */
        auto pairwise::run(const io::manager& io, const pairwise::pipe& pipe) const -> pairwise::pipe
        {
            auto algoname = io.cmd.get("pairwise", "default");
            auto tablename = io.cmd.get("scoring-table", "default");

            const auto conduit = pipeline::convert<pairwise::previous>(*pipe);

            auto table = pw::scoring_table::make(tablename);
            auto result = pw::run(*conduit.db, table, algoname);

            auto ptr = new pairwise::conduit {conduit.db, result};
            return pairwise::pipe {ptr};
        }

        /**
         * Checks whether command line arguments produce a valid module state.
         * @param io The pipeline's IO service instance.
         * @return Are the given command line arguments valid?
         */
        auto pairwise::check(const io::manager& io) const -> bool
        {
            auto algoname = io.cmd.get("pairwise", "default");
            enforce(pw::algorithm::has(algoname), "unknown pairwise algorithm chosen: '%s'", algoname);

            auto tablename = io.cmd.get("scoring-table", "default");
            enforce(pw::scoring_table::has(tablename), "unknown scoring table chosen: '%s'", tablename);
            
            return true;
        }
    }
}