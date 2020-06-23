/**
 * Multiple Sequence Alignment command line processing file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#include <map>
#include <string>
#include <vector>

#include <exception.hpp>
#include <io/cmdline.hpp>

using namespace msa;

namespace
{
    /**
     * Prepares a list of options to be parsed. If flags of options somehow overlap,
     * the last option in order will be the flag's corresponding option.
     * @param options The list of available options for this parser.
     * @return The map of prepared options.
     */
    static auto prepare(const std::vector<io::cmdline::option>& options) noexcept
    -> std::map<std::string, io::cmdline::option>
    {
        std::map<std::string, io::cmdline::option> opmap;
        
        for(const io::cmdline::option& current : options)
            if(current.opcode && !current.flags.empty())
                for(const std::string& flag : current.flags)
                    opmap[flag] = current;

        return opmap;
    }

    /**
     * Searches for an option via one of its flags.
     * @param needle The option being searched for.
     * @return The found option or an unknown option.
     */
    static auto find(
            const std::map<std::string, io::cmdline::option>& opmap
        ,   const std::string& needle
        ) noexcept
    -> const io::cmdline::option&
    {
        static io::cmdline::option unknown {};

        const auto result = opmap.find(needle);
        return result != opmap.end() ? result->second : unknown;
    }
}

namespace msa
{
    namespace io
    {
        /**
         * Parses the command line arguments according to the given options.
         * @param options The list of options available to user through command line.
         * @param argc The number of command line arguments.
         * @param argv The command line arguments.
         */
        auto cmdline::parse(const std::vector<cmdline::option>& options, int argc, char **argv) -> void
        {
            const auto opmap = prepare(options);

            for(int i = 1; i < argc; ++i) {
                const cmdline::option& current = find(opmap, argv[i]);

                if(!current.opcode || current.flags.empty()) {
                    m_posargs.push_back(argv[i]);
                    continue;
                }

                if(current.variadic) {
                    enforce(i + 1 < argc, "missing argument value for option '%s'", argv[i]);
                    ++i;
                }

                m_parsed[current.opcode] = argv[i];
            }
        }
    }
}