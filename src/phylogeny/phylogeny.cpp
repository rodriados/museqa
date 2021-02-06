/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the heuristics' phylogeny module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include "museqa.hpp"

#include "exception.hpp"
#include "dispatcher.hpp"

#include "phylogeny/phylogeny.cuh"
#include "phylogeny/njoining/njoining.cuh"

namespace museqa
{
    namespace phylogeny
    {
        /**
         * Keeps the list of available algorithms and their respective factories.
         * @since 0.1.1
         */
        static const dispatcher<factory> factory_dispatcher = {
            {"default",                     njoining::best}
        ,   {"njoining",                    njoining::best}
        ,   {"hybrid",                      njoining::hybrid_symmetric}
        ,   {"linear",                      njoining::hybrid_linear}
        ,   {"symmetric",                   njoining::hybrid_symmetric}
        ,   {"njoining-hybrid",             njoining::hybrid_symmetric}
        ,   {"njoining-linear",             njoining::hybrid_linear}
        ,   {"njoining-symmetric",          njoining::hybrid_symmetric}
        ,   {"sequential",                  njoining::sequential_symmetric}
        ,   {"distributed",                 njoining::sequential_symmetric}
        ,   {"njoining-sequential",         njoining::sequential_symmetric}
        ,   {"njoining-sequential-linear",  njoining::sequential_linear}
        ,   {"njoining-distributed",        njoining::sequential_symmetric}
        ,   {"njoining-distributed-linear", njoining::sequential_linear}
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
    }
}
