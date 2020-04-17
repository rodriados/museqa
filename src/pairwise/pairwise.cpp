/**
 * Multiple Sequence Alignment pairwise module file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <map>
#include <string>
#include <vector>

#include <msa.hpp>
#include <utils.hpp>
#include <buffer.hpp>
#include <exception.hpp>

#include <pairwise/pairwise.cuh>
#include <pairwise/needleman.cuh>

namespace msa
{
    namespace pairwise
    {
        /**
         * The list of all available algorithms' names.
         * @since 0.1.1
         */
        static const std::vector<std::string> algo_name = {
            "default"
        ,   "needleman"
        ,   "needleman-hybrid"
        ,   "needleman-sequential"
        ,   "needleman-distributed"
        };

        /**
         * Keeps the list of available algorithms and their respective factories.
         * @since 0.1.1
         */
        static const std::map<std::string, factory> dispatcher = {
            {algo_name[0], needleman::hybrid}
        ,   {algo_name[1], needleman::hybrid}
        ,   {algo_name[2], needleman::hybrid}
        ,   {algo_name[3], needleman::sequential}
        ,   {algo_name[4], needleman::sequential}
        };

        /**
         * Gets an algorithm factory by its name.
         * @param name The name of algorithm to retrieve.
         * @return The factory of requested algorithm.
         */
        auto algorithm::retrieve(const std::string& name) -> factory
        {
            const auto& selected = dispatcher.find(name);
            enforce(selected != dispatcher.end(), "unknown pairwise algorithm '%s'", name);
            return selected->second;
        }

        /**
         * Informs the names of all available algorithms.
         * @return The list of available algorithms.
         */
        auto algorithm::list() noexcept -> const std::vector<std::string>&
        {
            return algo_name;
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
