/**
 * Multiple Sequence Alignment phylogeny module entry file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include <msa.hpp>
#include <exception.hpp>
#include <symmatrix.hpp>
#include <dispatcher.hpp>

#include <phylogeny/phylogeny.cuh>
#include <phylogeny/njoining.cuh>

namespace msa
{
    namespace phylogeny
    {
        /**
         * Keeps the list of available algorithms and their respective factories.
         * @since 0.1.1
         */
        static const dispatcher<factory> factory_dispatcher = {
            {"default",              njoining::sequential}
        ,   {"njoining",             njoining::sequential}
        ,   {"njoining-sequential",  njoining::sequential}
        ,   {"njoining-distributed", njoining::sequential}
        };

        /**
         * Gets an algorithm factory by its name.
         * @param name The name of algorithm to retrieve.
         * @return The factory of requested algorithm.
         */
        auto algorithm::retrieve(const std::string& name) -> const factory&
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
         * Generates a pseudo-phylogenetic tree from the distance matrix of all
         * possible input sequence pairs.
         * @param config The module's configuration.
         * @return The new module manager instance.
         */
        auto manager::run(const configuration& config) -> manager
        {
            const auto& algof = algorithm::retrieve(config.algorithm);

            onlymaster watchdog::info("chosen phylogeny algorithm <bold>%s</>", config.algorithm);
            onlymaster watchdog::init("phylogeny", "building phylogenetic tree");

            algorithm *worker = (algof)();

            context ctx {config.pw, config.pw.count()};
            manager result {worker->run(ctx)};

            onlymaster watchdog::finish("phylogeny", "phylogenetic tree built");

            delete worker;
            return result;
        }
    }
}
