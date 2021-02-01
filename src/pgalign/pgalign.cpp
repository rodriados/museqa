/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the heuristics' profile-aligner module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include "exception.hpp"
#include "dispatcher.hpp"

#include "pgalign/pgalign.cuh"
#include "pgalign/myers/myers.cuh"

namespace museqa
{
    namespace pgalign
    {
        /**
         * Keeps the list of available algorithms and their respective factories.
         * @since 0.1.1
         */
        static const dispatcher<factory> factory_dispatcher = {
            {"default",             myers::sequential}
        ,   {"myers",               myers::sequential}
        ,   {"sequential",          myers::sequential}
        ,   {"myers-sequential",    myers::sequential}
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
            throw exception("unknown pgalign algorithm '%s'", name);
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
