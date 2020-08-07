/**
 * Multiple Sequence Alignment phylogeny module entry file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include <io.hpp>
#include <msa.hpp>
#include <exception.hpp>
#include <dispatcher.hpp>

#include <phylogeny/phylogeny.cuh>
#include <phylogeny/algorithm/njoining.cuh>

namespace msa
{
    namespace phylogeny
    {
        /**
         * Keeps the list of available algorithms and their respective factories.
         * @since 0.1.1
         */
        static const dispatcher<factory> factory_dispatcher = {
            {"default",              njoining::sequential_mat}
        ,   {"njoining",             njoining::sequential_sym}
        ,   {"njoining-matrix",      njoining::sequential_mat}
        ,   {"njoining-symmatrix",   njoining::sequential_sym}
        ,   {"njoining-sequential",  njoining::sequential_mat}
        ,   {"njoining-distributed", njoining::sequential_mat}
        };

        /**
         * Informs whether a given factory name exists in dispatcher.
         * @param name The name of algorithm to check existance of.
         * @return Does the chosen algorithm exist?
         */
        auto algorithm::has(const std::string& name) -> bool
        {
            return factory_dispatcher.has(name);
        }

        /**
         * Gets an algorithm factory by its name.
         * @param name The name of algorithm to retrieve.
         * @return The factory of requested algorithm.
         */
        auto algorithm::make(const std::string& name) -> const factory&
        try {
            return factory_dispatcher[name];
        } catch(const exception& e) {
            throw exception("unknown phylogeny algorithm '%s'", name);
        }

        /**
         * Informs the names of all available algorithms.
         * @return The list of available algorithms.
         */
        auto algorithm::list() noexcept -> const std::vector<std::string>&
        {
            return factory_dispatcher.list();
        }

        /**
         * Execute the module's task when on a pipeline.
         * @param io The pipeline's IO service instance.
         * @return A conduit with the module's processed results.
         */
        auto module::run(const io::service& io, const module::pipe& pipe) const -> module::pipe
        {
            auto algoname = io.get<std::string>(cli::phylogeny, "default");
            const auto conduit = pipeline::convert<module::previous>(*pipe);

            auto result = phylogeny::run(conduit.distances, conduit.total, algoname);

            auto ptr = new module::conduit {conduit.db, result};
            return module::pipe {ptr};
        }

        /**
         * Checks whether command line arguments produce a valid module state.
         * @param io The pipeline's IO service instance.
         * @return Are the given command line arguments valid?
         */
        auto module::check(const io::service& io) const -> bool
        {
            auto algoname = io.get<std::string>(cli::phylogeny, "default");
            enforce(algorithm::has(algoname), "unknown phylogeny algorithm chosen: '%s'", algoname);

            return true;
        }
    }
}
