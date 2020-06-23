/**
 * Multiple Sequence Alignment pairwise module entry file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include <io.hpp>
#include <msa.hpp>
#include <utils.hpp>
#include <buffer.hpp>
#include <database.hpp>
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

            for(size_t i = 0, c = 0; i < num - 1; ++i)
                for(size_t j = i + 1; j < num; ++j, ++c)
                    pairs[c] = pair {seqref(i), seqref(j)};

            return pairs;
        }

        /**
         * Execute the module's task when on a pipeline.
         * @param io The pipeline's IO service instance.
         * @return A conduit with the module's processed results.
         */
        auto module::run(const io::service& io, const module::pipe& pipe) const -> module::pipe
        {
            auto algoname = io.get<std::string>(cli::pairwise, "default");
            auto tablename = io.get<std::string>(cli::scoring, "default");
            const auto conduit = pipeline::convert<module>(*pipe);

            auto table = scoring_table::make(tablename);
            auto result = pairwise::run(conduit.db, table, algoname);

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
            auto algoname = io.get<std::string>(cli::pairwise, "default");
            auto tablename = io.get<std::string>(cli::scoring, "default");

            enforce(algorithm::has(algoname), "unknown pairwise algorithm chosen: '%s'", algoname);
            enforce(scoring_table::has(tablename), "unknown scoring table chosen: '%s'", tablename);
            
            return true;
        }
    }
}
