/**
 * Multiple Sequence Alignment command line processing file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <string>
#include <vector>
#include <unordered_map>

#include <cmdline.hpp>
#include <exception.hpp>

/**
 * The global command line parser singleton instance.
 * @since 0.1.1
 */
cmdline::parser cmdline::instance;

namespace cmdline
{
    /**
     * Prepares a list of options to be parsed. If flags of options somehow overlap,
     * the last option in order will be the flag's corresponding option.
     * @param options The list of available options for this parser.
     * @return The map of prepared options.
     */
    auto prepare(const std::vector<option>& options) noexcept
    -> std::unordered_map<std::string, option>
    {
        std::unordered_map<std::string, option> result;

        for(const option& current : options)
            if(!current.name.empty() && !current.flags.empty())
                for(const std::string& flag : current.flags)
                    result[flag] = current;

        return result;
    }

    /**
     * Searches for an option via one of its names.
     * @param needle The option being searched for.
     * @return The found option or an unknown option.
     */
    auto find(const parser& parser, const std::string& needle) noexcept -> const option&
    {
        static option unknown {};

        const auto& value = parser.config.find(needle);
        return value != parser.config.end() ? value->second : unknown;
    }

    /**
     * Parses the command line arguments through its options.
     * @param argc The number of command line arguments.
     * @param argv The command line arguments.
     */
    void parse(int argc, char **argv)
    {
        for(int i = 1; i < argc; ++i) {
            const option& current = find(instance, argv[i]);

            if(!current.name.empty() && !current.flags.empty()) {
                if(current.variadic) {
                    enforce(i + 1 < argc, "missing argument value for option '%s'", current.name);
                    instance.result[current.name] = argv[++i];
                    continue;
                }

                else {
                    instance.result[current.name] = argv[i];
                    continue;
                }
            }

            instance.positional.push_back(argv[i]);
        }
    }
}