/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the command line input parser module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include <map>
#include <string>

#include "terminal.hpp"
#include "exception.hpp"

namespace museqa
{
    /**
     * Parses the command line arguments according to the given options.
     * @param options The list of options available to user through command line.
     * @param argc The number of command line arguments.
     * @param argv The command line arguments.
     * @return The parsed command line arguments instance.
     */
    auto terminal::parse(const std::vector<terminal::option>& options, int argc, char **argv) -> terminal::parser
    {
        terminal::parser parser;
        std::map<std::string, terminal::option> option_map;

        for(const terminal::option& current : options) {
            for(const std::string& flag : current.flags)
                option_map[flag] = current;
        }

        for(int i = 1; i < argc; ++i) {
            const auto selected = option_map.find(argv[i]);

            if(selected == option_map.end()) {
                parser.m_positional.push_back(argv[i]);
                continue;
            } 

            const auto& target = selected->second;
            const auto& name = target.name;

            if(target.is_variadic)
                enforce(++i < argc, "missing argument value for option '%s'", selected->first);

            parser.m_parsed[name].push_back(argv[i]);
        }

        return parser;
    }
}
