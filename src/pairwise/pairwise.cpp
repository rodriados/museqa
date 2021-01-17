/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the heuristics' pairwise module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include "utils.hpp"
#include "buffer.hpp"
#include "exception.hpp"
#include "dispatcher.hpp"

#include "pairwise/pairwise.cuh"
#include "pairwise/needleman/needleman.cuh"

namespace museqa
{
    namespace pairwise
    {
        /**
         * Keeps the list of available algorithms and their respective factories.
         * @since 0.1.1
         */
        static const dispatcher<factory> factory_dispatcher = {
            {"default",               needleman::hybrid}
        ,   {"hybrid",                needleman::hybrid}
        ,   {"needleman",             needleman::hybrid}
        ,   {"needleman-hybrid",      needleman::hybrid}
        ,   {"sequential",            needleman::sequential}
        ,   {"distributed",           needleman::sequential}
        ,   {"needleman-sequential",  needleman::sequential}
        ,   {"needleman-distributed", needleman::sequential}
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
            throw exception("unknown pairwise algorithm '%s'", name);
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
         * Generates all working pairs for a given number of elements.
         * @param num The total number of elements.
         * @return The generated sequence pairs.
         */
        auto algorithm::generate(size_t num) const -> buffer<pair>
        {
            const auto total = utils::nchoose(num);
            auto pairs = buffer<pair>::make(total);

            // We generate our sequence pairs in such a way that a pair will always
            // be at the same offset in array, independently of the number of sequences.
            for(size_t i = 1, c = 0; i < num; ++i)
                for(size_t j = 0; j < i; ++j, ++c)
                    pairs[c] = pair {seqref(i), seqref(j)};

            return pairs;
        }
    }
}
