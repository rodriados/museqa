/** 
 * Multiple Sequence Alignment parser file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include <utils.hpp>
#include <database.hpp>
#include <exception.hpp>
#include <dispatcher.hpp>

#include <io/loader/database.hpp>
#include <io/loader/parser/fasta.hpp>

namespace msa
{
    namespace
    {
        /**
         * Aliases the target functor into the anonymous namespace.
         * @since 0.1.1
         */
        using functor = typename io::loader<database>::functor;

        /*
         * Keeps the list of available parsers and their respective file extensions
         * correspondence. Whenever a new parser is introduced, it must be listed.
         */
        static const dispatcher<functor> parser_dispatcher = {
            {"fa", io::parser::fasta}
        ,   {"fasta", io::parser::fasta}
        };
    }

    namespace io
    {
        /**
         * Retrives a parser from its identification name or file extension.
         * @param ext The file extension to get the corresponding parser of.
         * @return The retrieved parser functor.
         */
        auto loader<database>::factory(const std::string& ext) const -> functor
        try {
            return parser_dispatcher[ext];
        } catch(const exception& e) {
            throw exception("unknown database parser '%s'", ext);
        }

        /**
         * Checks whether the given file has any known parsers for target type.
         * @param filename The name of file to be validated.
         * @return Can the given filename be parsed?
         */
        auto loader<database>::validate(const std::string& filename) const noexcept -> bool
        {
            const auto ext = utils::extension(filename);
            return parser_dispatcher.has(ext);
        }

        /**
         * Informs the list of all available parsers.
         * @return The list of parsers names.
         */
        auto loader<database>::list() const noexcept -> const std::vector<std::string>&
        {
            return parser_dispatcher.list();
        }
    }
}
