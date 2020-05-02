/**
 * Multiple Sequence Alignment pairwise module entry file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include <msa.hpp>
#include <utils.hpp>
#include <buffer.hpp>
#include <exception.hpp>
#include <dispatcher.hpp>

#include <pairwise/pairwise.cuh>
#include <pairwise/needleman.cuh>

namespace msa
{
    namespace pairwise
    {
        /**
         * Keeps the list of available algorithms and their respective factories.
         * @since 0.1.1
         */
        static const dispatcher<factory> factory_dispatcher = {
            {"default",               needleman::hybrid}
        ,   {"needleman",             needleman::hybrid}
        ,   {"needleman-hybrid",      needleman::hybrid}
        ,   {"needleman-sequential",  needleman::sequential}
        ,   {"needleman-distributed", needleman::sequential}
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
        auto algorithm::generate(size_t num) -> buffer<pair>&
        {
            const auto total = utils::nchoose(num);
            auto pairbuf = buffer<pair>::make(total);

            for(size_t i = 0, c = 0; i < num - 1; ++i)
                for(size_t j = i + 1; j < num; ++j, ++c)
                    pairbuf[c] = {seqref(i), seqref(j)};

            return pairs = pairbuf;
        }

        /**
         * Aligns every sequence in given database pairwise, thus calculating a
         * similarity score for every different permutation of sequence pairs.
         * @param config The module's configuration.
         * @return The new module manager instance.
         */
        auto manager::run(const configuration& config) -> manager
        {
            using utils::nchoose;

            const auto& algof = algorithm::retrieve(config.algorithm);
            const auto table  = scoring_table::make(config.table);

            onlymaster watchdog::info("chosen pairwise algorithm <bold>%s</>", config.algorithm);
            onlymaster watchdog::info("chosen pairwise scoring table <bold>%s</>", config.table);
            onlymaster watchdog::init("pairwise", "aligning <bold>%llu</> pairs", nchoose(config.db.count()));

            algorithm *worker = (algof)();

            context ctx {config.db, table};
            manager result {worker->run(ctx), config.db.count()};

            onlymaster watchdog::finish("pairwise", "aligned all sequence pairs");

            delete worker;
            return result;
        }
    }
}
