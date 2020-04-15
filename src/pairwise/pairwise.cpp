/**
 * Multiple Sequence Alignment pairwise module file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <map>
#include <string>

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
         * Keeps the list of available algorithms and their respective factories.
         * @since 0.1.1
         */
        static const std::map<std::string, factory> dispatcher = {
            {"default",                 needleman::hybrid}
        ,   {"needleman",               needleman::hybrid}
        ,   {"needleman-hybrid",        needleman::hybrid}
        ,   {"needleman-sequential",    needleman::sequential}
        ,   {"needleman-distributed",   needleman::sequential}
        };

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
            const auto& selected = dispatcher.find(config.algorithm);

            enforce(selected != dispatcher.end(), "unknown pairwise algorithm <bold>%s</>", config.algorithm);

            onlymaster watchdog::info("chosen pairwise algorithm <bold>%s</>", config.algorithm);    
            onlymaster watchdog::init("pairwise", "aligning <bold>%llu</> pairs", utils::nchoose(config.db.count()));

            algorithm *worker = (selected->second)();
            manager result {worker->run(config), config.db.count()};

            onlymaster watchdog::finish("pairwise", "aligned all sequence pairs");
            
            delete worker;

            return result;
        }
    }
}